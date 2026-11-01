"""
common.py - shared utilities for the UBCF + CBF hybrid model
definitions:

  * identical temporal 5-1-1 split                 -> TRAIN_DAYS / VAL_DAY / TEST_DAY
  * identical candidate & impression format        -> load_candidates()
  * no future information leakage                  -> build_click_log() only uses
                                                      clicks from the build window
  * identical missing-value treatment (train mean) -> impute_train_mean()
  * identical negative-sampling policy             -> sample_negatives()
  * same metrics: MRR, nDCG@5, nDCG@10, AUC -> evaluate()
  * same cold-start definitions                    -> add_cold_start_flags()
  * same output format                             -> save_scores() / load_scores()

Banned features (slate_length, article_recency, session_dow) are never computed here.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Configuration  (edit here, or override with environment variables)
# --------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("MIND_DATA_DIR", os.path.join(_HERE, "..", "..", "data", "raw"))  # contains train/ and valid/
OUT_DIR = os.environ.get("HYBRID_OUT_DIR", os.path.join(_HERE, "outputs"))  # score files + metrics
CACHE_DIR = os.path.join(OUT_DIR, "cache")
SPLITS = ("train", "valid")   # MIND-small train / dev folders under DATA_DIR
SEED = 42

# MIND-small train (9-14 Nov 2019) + dev (15 Nov 2019) = 7 days -> 5-1-1
TRAIN_DAYS = ["2019-11-09", "2019-11-10", "2019-11-11", "2019-11-12", "2019-11-13"]
VAL_DAY = "2019-11-14"
TEST_DAY = "2019-11-15"

# Day-block stacking: base models built on days 1-4 score day 5 -> stacker training rows.
# For val/test the base models are rebuilt on all 5 train days (same as standalone models).
PHASES = {
    "stack_train": {"build_days": TRAIN_DAYS[:-1], "score_days": [TRAIN_DAYS[-1]]},
    "val":         {"build_days": TRAIN_DAYS,      "score_days": [VAL_DAY]},
    "test":        {"build_days": TRAIN_DAYS,      "score_days": [TEST_DAY]},
}

NEG_K = 4            # group negative-sampling policy: K negatives per positive (stacker training only)
KS = (5, 10)         # nDCG@K cut-offs

# Shared output format: one row per impression-candidate pair
OUTPUT_COLS = ["imp_id", "user_id", "news_id", "label", "score", "available",
               "user_cold", "item_cold"]

os.makedirs(CACHE_DIR, exist_ok=True)


# --------------------------------------------------------------------------
# I/O helpers (parquet when pyarrow is installed, pickle otherwise)
# --------------------------------------------------------------------------
def _has_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


def save_df(df: pd.DataFrame, path_no_ext: str) -> str:
    if _has_parquet():
        path = path_no_ext + ".parquet"
        df.to_parquet(path, index=False)
    else:
        path = path_no_ext + ".pkl"
        df.to_pickle(path)
    return path


def load_df(path_no_ext: str) -> pd.DataFrame:
    if os.path.exists(path_no_ext + ".parquet"):
        return pd.read_parquet(path_no_ext + ".parquet")
    return pd.read_pickle(path_no_ext + ".pkl")


def _cached(path_no_ext: str) -> bool:
    return os.path.exists(path_no_ext + ".parquet") or os.path.exists(path_no_ext + ".pkl")


# --------------------------------------------------------------------------
# Raw data
# --------------------------------------------------------------------------
NEWS_COLS = ["news_id", "category", "subcategory", "title", "abstract", "url",
             "title_entities", "abstract_entities"]
BEH_COLS = ["impression_id", "user_id", "time", "history", "impressions"]


def load_news(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Train + dev news catalogue, de-duplicated. Missing abstracts are imputed with
    the title (same convention as 02_feature_engineering)."""
    parts = [pd.read_csv(f"{data_dir}/{s}/news.tsv", sep="\t", header=None,
                         names=NEWS_COLS, quoting=3) for s in SPLITS]
    news = pd.concat(parts).drop_duplicates("news_id").reset_index(drop=True)
    news["title"] = news["title"].fillna("")
    news["abstract_imputed"] = news["abstract"].fillna(news["title"])
    news["nidx"] = np.arange(len(news), dtype=np.int32)    # row index used by embeddings
    return news


def load_behaviors(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Train + dev behaviours with a globally unique imp_id (raw impression ids restart
    at 1 in dev) and the 5-1-1 split label."""
    cache = os.path.join(CACHE_DIR, "behaviors")
    if _cached(cache):
        return load_df(cache)
    parts = []
    for s in SPLITS:
        df = pd.read_csv(f"{data_dir}/{s}/behaviors.tsv", sep="\t", header=None, names=BEH_COLS)
        df["source"] = s
        parts.append(df)
    beh = pd.concat(parts, ignore_index=True)
    beh["time"] = pd.to_datetime(beh["time"], format="%m/%d/%Y %I:%M:%S %p")
    beh["history"] = beh["history"].fillna("").astype(str).str.split()
    beh = beh.sort_values(["time", "source", "impression_id"]).reset_index(drop=True)
    beh["imp_id"] = np.arange(len(beh), dtype=np.int64)
    beh["date"] = beh["time"].dt.strftime("%Y-%m-%d")
    beh["split"] = np.select(
        [beh["date"].isin(TRAIN_DAYS), beh["date"] == VAL_DAY, beh["date"] == TEST_DAY],
        ["train", "val", "test"], default="other")
    save_df(beh, cache)
    return beh


def load_candidates(beh: pd.DataFrame) -> pd.DataFrame:
    """Shared candidate format: one row per (impression, candidate article)."""
    cache = os.path.join(CACHE_DIR, "candidates")
    if _cached(cache):
        return load_df(cache)
    s = beh[["imp_id", "impressions"]].copy()
    s["impressions"] = s["impressions"].str.split()
    s = s.explode("impressions", ignore_index=True)
    parts = s["impressions"].str.rsplit("-", n=1, expand=True)
    cand = pd.DataFrame({
        "imp_id": s["imp_id"].to_numpy(dtype=np.int64),
        "news_id": parts[0].to_numpy(),
        "label": parts[1].astype(np.int32).to_numpy(),    # int8 overflows once summed per impression
    })
    cand = cand.merge(beh[["imp_id", "user_id", "time", "date", "split"]], on="imp_id", how="left")
    save_df(cand, cache)
    return cand


# --------------------------------------------------------------------------
# Leakage-safe click logs
# --------------------------------------------------------------------------
def build_click_log(beh: pd.DataFrame, cand: pd.DataFrame, build_days) -> pd.DataFrame:
    """All clicks a base model may learn from when built on `build_days`:
       (a) pre-period reading histories of users active in the build window
           (no timestamp in MIND -> stamped at the start of the build window, i.e. oldest)
       (b) impression clicks inside the build window (with their real timestamps).
    Returns one row per (user_id, news_id) with the latest click time."""
    b = beh[beh["date"].isin(build_days)]
    hist = (b.sort_values("time")[["user_id", "history"]]
              .drop_duplicates("user_id", keep="last")
              .explode("history").dropna()
              .rename(columns={"history": "news_id"}))
    hist["time"] = pd.Timestamp(min(build_days))
    clicks = cand[cand["date"].isin(build_days) & (cand["label"] == 1)][["user_id", "news_id", "time"]]
    log = pd.concat([hist[["user_id", "news_id", "time"]], clicks], ignore_index=True)
    log = log.sort_values("time").drop_duplicates(["user_id", "news_id"], keep="last")
    return log.reset_index(drop=True)


def scoring_histories(beh: pd.DataFrame, score_days) -> pd.DataFrame:
    """Reading history each user brings into the scoring window (earliest impression in the
    window, so no later information is used). Used to fold in users the build log lacks."""
    s = beh[beh["date"].isin(score_days)].sort_values("time")
    first = s.drop_duplicates("user_id", keep="first")[["user_id", "history"]]
    h = first.explode("history").dropna().rename(columns={"history": "news_id"})
    return h[["user_id", "news_id"]].reset_index(drop=True)


def user_query_log(build_log: pd.DataFrame, beh: pd.DataFrame, score_days, users) -> pd.DataFrame:
    """Clicks known for each scoring user at scoring time = build-window clicks + history
    they arrive with. History items carry the oldest timestamp of the build window."""
    users = pd.Index(pd.unique(users))
    known = build_log[build_log["user_id"].isin(users)]
    h = scoring_histories(beh, score_days)
    h = h[h["user_id"].isin(users)].copy()
    h["time"] = build_log["time"].min()
    q = pd.concat([known, h], ignore_index=True)
    q = q.sort_values("time").drop_duplicates(["user_id", "news_id"], keep="last")
    return q.reset_index(drop=True)


def phase_candidates(cand: pd.DataFrame, phase: str) -> pd.DataFrame:
    return cand[cand["date"].isin(PHASES[phase]["score_days"])].reset_index(drop=True)


# --------------------------------------------------------------------------
# Shared cold-start definitions
# --------------------------------------------------------------------------
def add_cold_start_flags(c: pd.DataFrame, beh: pd.DataFrame, build_log: pd.DataFrame,
                         score_days) -> pd.DataFrame:
    """Contract cold-start definitions (agree these with the group):
       user_cold = user has 0 clicked articles known at scoring time
                   (nothing in the build-window log AND an empty history on arrival)
       item_cold = article has 0 historical clicks in the build window
                   (never in any build-window history nor clicked in a build-window impression)
    """
    q = user_query_log(build_log, beh, score_days, c["user_id"])
    warm_users = set(q["user_id"].unique())
    clicked_items = set(build_log["news_id"].unique())
    c = c.copy()
    c["user_cold"] = (~c["user_id"].isin(warm_users)).astype(np.int8)
    c["item_cold"] = (~c["news_id"].isin(clicked_items)).astype(np.int8)
    return c


# --------------------------------------------------------------------------
# Shared missing-value treatment & negative sampling
# --------------------------------------------------------------------------
def impute_train_mean(train_df: pd.DataFrame, dfs: list, cols) -> dict:
    """Fill NaNs in `cols` of every frame in `dfs` (in place) with the TRAINING-set mean.
    Pair every imputed column with an availability flag so the model can tell
    'unknown' from 'average'. Returns the means used."""
    means = {c: float(train_df[c].mean()) if train_df[c].notna().any() else 0.0 for c in cols}
    for d in dfs:
        for c in cols:
            d[c] = d[c].fillna(means[c])
    return means


def sample_negatives(df: pd.DataFrame, k: int = NEG_K, seed: int = SEED) -> pd.DataFrame:
    """Group policy: keep every positive, sample up to k negatives per positive within the same
    impression. Applied to model TRAINING rows only - evaluation always uses full slates."""
    rng = np.random.default_rng(seed)
    d = df.copy()
    d["_r"] = rng.random(len(d))
    npos = d.groupby("imp_id")["label"].transform("sum")
    neg = d["label"] == 0
    d["_nr"] = np.inf
    d.loc[neg, "_nr"] = d[neg].groupby("imp_id")["_r"].rank(method="first")
    keep = (d["label"] == 1) | (d["_nr"] <= k * npos)
    return d[keep & (npos > 0)].drop(columns=["_r", "_nr"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Shared evaluation (tie-safe)
# --------------------------------------------------------------------------
def evaluate(df: pd.DataFrame, score_col: str = "score", ks=KS, seed: int = SEED,
             by_segment: bool = True) -> pd.DataFrame:
    """MRR, nDCG@k and AUC per impression on FULL slates, averaged.

    Ties are broken RANDOMLY. (The baseline sorted (score, label) tuples, which puts the
    clicked article first whenever scores tie - inflating MRR/nDCG for sparse scorers.)
    Impressions without a click are dropped; AUC is averaged over impressions with
    at least one positive and one negative.
    """
    need = ["imp_id", "label", score_col, "user_cold", "item_cold"]
    d = df[[c for c in need if c in df.columns]].copy()
    if d[score_col].isna().any():
        raise ValueError(f"{score_col} has NaNs - impute (train mean) before evaluating")
    rng = np.random.default_rng(seed)
    d["_r"] = rng.random(len(d))
    d = d.sort_values(["imp_id", score_col, "_r"], ascending=[True, False, True])
    d["rank"] = d.groupby("imp_id").cumcount() + 1

    g = d.groupby("imp_id")
    imp = pd.DataFrame({"npos": g["label"].sum(), "n": g.size()})
    imp = imp[imp["npos"] > 0]
    pos = d[d["label"] == 1]

    imp["MRR"] = 1.0 / pos.groupby("imp_id")["rank"].min()
    disc = 1.0 / np.log2(np.arange(2, 2 + max(int(imp["n"].max()), max(ks)) + 1))
    ideal_cum = np.concatenate([[0.0], np.cumsum(disc)])
    for k in ks:
        hit = pos[pos["rank"] <= k]
        dcg = (1.0 / np.log2(hit["rank"] + 1)).groupby(hit["imp_id"]).sum()
        idcg = ideal_cum[np.minimum(imp["npos"].to_numpy(), k)]
        imp[f"nDCG@{k}"] = dcg.reindex(imp.index).fillna(0.0).to_numpy() / idcg
    # AUC via Mann-Whitney on within-impression average ranks (ties get half credit)
    d["arank"] = g[score_col].rank(method="average")
    sum_pos_rank = d[d["label"] == 1].groupby("imp_id")["arank"].sum().reindex(imp.index)
    nneg = imp["n"] - imp["npos"]
    auc = (sum_pos_rank - imp["npos"] * (imp["npos"] + 1) / 2) / (imp["npos"] * nneg)
    imp["AUC"] = auc.where(nneg > 0)

    metric_cols = ["MRR"] + [f"nDCG@{k}" for k in ks] + ["AUC"]
    if not by_segment or "user_cold" not in d.columns:
        out = imp[metric_cols].mean().to_frame("All").T
        out.insert(0, "impressions", len(imp))
        return out

    ucold = d.groupby("imp_id")["user_cold"].first().reindex(imp.index)
    pos_item_cold = pos.groupby("imp_id")["item_cold"].max().reindex(imp.index)
    segs = {
        "All": np.ones(len(imp), bool),
        "Warm users": (ucold == 0).to_numpy(),
        "Cold users (0 clicks)": (ucold == 1).to_numpy(),
        "Clicked article seen before": (pos_item_cold == 0).to_numpy(),
        "Clicked article new (no prior clicks)": (pos_item_cold == 1).to_numpy(),
    }
    rows = []
    for name, m in segs.items():
        r = imp.loc[m, metric_cols].mean()
        r["impressions"] = int(m.sum())
        rows.append(r.rename(name))
    out = pd.DataFrame(rows)[["impressions"] + metric_cols]
    out["impressions"] = out["impressions"].astype(int)
    return out


# --------------------------------------------------------------------------
# Shared output format
# --------------------------------------------------------------------------
def save_scores(df: pd.DataFrame, model: str, phase: str, extra_cols=()) -> str:
    """Write a model's scores in the shared format (+ optional model-specific feature cols)."""
    missing = [c for c in OUTPUT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"missing output columns: {missing}")
    cols = OUTPUT_COLS + [c for c in extra_cols if c not in OUTPUT_COLS]
    out = df[cols].copy()
    out["available"] = out["available"].astype(np.int8)
    os.makedirs(OUT_DIR, exist_ok=True)
    return save_df(out, os.path.join(OUT_DIR, f"scores_{model}_{phase}"))


def load_scores(model: str, phase: str) -> pd.DataFrame:
    return load_df(os.path.join(OUT_DIR, f"scores_{model}_{phase}"))


def standalone_eval(model: str, score_col: str = "score") -> dict:
    """Evaluate a base model on its own: unavailable scores -> training-set mean
    (mean of available scores in the stack_train phase, i.e. scored train-period day)."""
    tr = load_scores(model, "stack_train")
    fill = float(tr.loc[tr["available"] == 1, score_col].mean()) if (tr["available"] == 1).any() else 0.0
    res = {}
    for ph in ("val", "test"):
        d = load_scores(model, ph)
        d[score_col] = d[score_col].where(d["available"] == 1, fill).fillna(fill)
        res[ph] = evaluate(d, score_col)
    return res


def decay_weights(times: pd.Series, ref_time: pd.Timestamp, tau_days) -> np.ndarray:
    """Exponential time decay exp(-age/tau). tau_days=None -> all clicks weigh 1."""
    if tau_days is None:
        return np.ones(len(times), dtype=np.float32)
    age = (ref_time - times).dt.total_seconds().to_numpy() / 86400.0
    return np.exp(-np.clip(age, 0, None) / tau_days).astype(np.float32)
