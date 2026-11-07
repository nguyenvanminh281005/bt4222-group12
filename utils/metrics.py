"""Equivalent ranking primitives and reporting cohorts."""

import numpy as np


def ndcg_at_k(labels, ranked_labels, k):
    top = np.asarray(ranked_labels)[:k]
    gain = (top / np.log2(np.arange(2, len(top) + 2))).sum()
    ideal = np.sort(np.asarray(labels))[::-1][:k]
    ideal_gain = (ideal / np.log2(np.arange(2, len(ideal) + 2))).sum()
    return float(gain / ideal_gain) if ideal_gain > 0 else 0.0


def history_groups(history_length):
    n = np.asarray(history_length)
    return np.select([n == 0, n < 5], ["cold_user", "sparse_user"], default="warm_user")


import pandas as pd


def evaluate_scores(
    df: pd.DataFrame,
    score_col: str = "score",
    ks=(5, 10),
    seed: int = 42,
    by_segment: bool = True,
    user_groups=None,
) -> pd.DataFrame:
    """MRR, nDCG@k and AUC per impression on FULL slates, averaged.

    Score ties use reproducible random ordering.
    Impressions without a click are dropped; AUC is averaged over impressions with
    at least one positive and one negative.
    """
    need = ["impression_id", "label", score_col, "item_cold"]
    d = df[[c for c in need if c in df.columns]].copy()
    if d[score_col].isna().any():
        raise ValueError(
            f"{score_col} has NaNs - impute (train mean) before evaluating"
        )
    rng = np.random.default_rng(seed)
    d["_r"] = rng.random(len(d))
    d = d.sort_values(["impression_id", score_col, "_r"], ascending=[True, False, True])
    d["rank"] = d.groupby("impression_id").cumcount() + 1

    g = d.groupby("impression_id")
    imp = pd.DataFrame({"npos": g["label"].sum(), "n": g.size()})
    imp = imp[imp["npos"] > 0]
    pos = d[d["label"] == 1]

    imp["MRR"] = 1.0 / pos.groupby("impression_id")["rank"].min()
    disc = 1.0 / np.log2(np.arange(2, 2 + max(int(imp["n"].max()), max(ks)) + 1))
    ideal_cum = np.concatenate([[0.0], np.cumsum(disc)])
    for k in ks:
        hit = pos[pos["rank"] <= k]
        dcg = (1.0 / np.log2(hit["rank"] + 1)).groupby(hit["impression_id"]).sum()
        idcg = ideal_cum[np.minimum(imp["npos"].to_numpy(), k)]
        imp[f"nDCG@{k}"] = dcg.reindex(imp.index).fillna(0.0).to_numpy() / idcg
    # AUC via Mann-Whitney on within-impression average ranks (ties get half credit)
    d["arank"] = g[score_col].rank(method="average")
    sum_pos_rank = (
        d[d["label"] == 1].groupby("impression_id")["arank"].sum().reindex(imp.index)
    )
    nneg = imp["n"] - imp["npos"]
    auc = (sum_pos_rank - imp["npos"] * (imp["npos"] + 1) / 2) / (imp["npos"] * nneg)
    imp["AUC"] = auc.where(nneg > 0)

    metric_cols = ["MRR"] + [f"nDCG@{k}" for k in ks] + ["AUC"]
    if not by_segment or "item_cold" not in d.columns:
        out = imp[metric_cols].mean().to_frame("All").T
        out.insert(0, "impressions", len(imp))
        return out

    ugroup = user_groups.reindex(imp.index)
    pos_item_cold = pos.groupby("impression_id")["item_cold"].max().reindex(imp.index)
    segs = {
        "All": np.ones(len(imp), bool),
        **{
            seg: (ugroup == seg).to_numpy()
            for seg in (
                "Cold users (0 history)",
                "Sparse users (1-4 history)",
                "Warm users (>=5 history)",
            )
        },
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
