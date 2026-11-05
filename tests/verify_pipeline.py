"""Synthetic pipeline smoke test; pretrained text vectors use a seeded fixture cache."""

import os, sys, json, ast, tempfile, contextlib, io
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
import numpy as np, pandas as pd, torch

torch.set_num_threads(1)
os.environ["MPLBACKEND"] = "Agg"
import matplotlib.pyplot as plt

plt.show = lambda: plt.close("all")


def execute(p, overrides=None):
    n = json.loads((ROOT / p).read_text(encoding="utf-8"))
    scope = {"display": lambda *a, **k: None, "__name__": "notebook"}
    for i, c in enumerate(n["cells"]):
        if c["cell_type"] != "code":
            continue
        s = "".join(c["source"])
        if overrides:
            s = overrides(s)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(s, f"{p}:cell{i}", "exec"), scope)
        except Exception:
            print("FAILED", p, "cell", i, s[:180])
            if "COLD_INTER" in scope:
                tr = scope["A"]["train"]
                rows = np.repeat(tr["hist_len"] == 0, tr["n"])
                print("COLD_INTER", scope["COLD_INTER"])
                print("actual", tr["cand_x"][rows][:3, len(scope["POP_COLS"]) :])
            raise
    print("PASS", p)
    return scope


def limited(s):
    class FixtureSettings(ast.NodeTransformer):
        def visit_Assign(self, node):
            self.generic_visit(node)
            names = {
                target.id for target in node.targets if isinstance(target, ast.Name)
            }
            if names & {"MAX_EPOCHS", "ABLATION_EPOCHS"}:
                node.value = ast.Constant(1)
            if names & {"RUN_ABLATION", "RUN_PERMUTATION_IMPORTANCE"}:
                node.value = ast.Constant(False)
            return node

        def visit_keyword(self, node):
            if node.arg == "epochs":
                node.value = ast.Constant(1)
            elif node.arg in {"skip_seeds", "skip_ablation"}:
                node.value = ast.Constant(True)
            return node

        def visit_Dict(self, node):
            self.generic_visit(node)
            for index, key in enumerate(node.keys):
                if isinstance(key, ast.Constant) and key.value == "epochs":
                    node.values[index] = ast.Constant(1)
            return node

    return ast.unparse(ast.fix_missing_locations(FixtureSettings().visit(ast.parse(s))))


with tempfile.TemporaryDirectory(prefix="group12-fixture-") as tmp:
    base = Path(tmp)
    raw = base / "raw"
    features = base / "features"
    features.mkdir()
    os.environ.update(
        MIND_DATA_DIR=str(raw),
        BT4222_BASE=str(base),
        BT4222_FEATURE_DIR=str(features),
        BT4222_OUTPUT_DIR=str(base / "outputs"),
    )
    ids = [f"N{i}" for i in range(1, 31)]
    entities = json.dumps([{"WikidataId": "Q1"}])
    rng = np.random.default_rng(42)
    for source in ["train", "dev"]:
        d = raw / source
        d.mkdir(parents=True)
        news = [
            [
                nid,
                ["sports", "business"][j % 2],
                "sub" + str(j % 4),
                f"news title article {j} sports business",
                "a detailed abstract article text",
                "https://example.test/" + nid,
                json.dumps([{"WikidataId": f"Q{j%7+1}"}]),
                json.dumps([{"WikidataId": f"Q{(j+2)%7+1}"}]),
            ]
            for j, nid in enumerate(ids)
        ]
        pd.DataFrame(news).to_csv(
            d / "news.tsv", sep="\t", header=False, index=False, quoting=3
        )
        (d / "entity_embedding.vec").write_text(
            "".join(
                f"Q{q}\t" + "\t".join(map(str, rng.normal(size=100))) + "\n"
                for q in range(1, 8)
            )
        )
        rows = []
        counter = 1
        for day in (range(6) if source == "train" else [6]):
            for k in range(60):
                hl = [0, 1, 4, 5, 10][k % 5]
                history = " ".join(ids[(k + j) % 20] for j in range(hl))
                candidates = " ".join(
                    f"{ids[(k+j)%30]}-{int(j in [0,2])}" for j in range(6)
                )
                stamp = pd.Timestamp("2019-11-09") + pd.Timedelta(
                    days=day, hours=k % 24, minutes=k
                )
                rows.append(
                    [
                        counter,
                        "U" + str(k % 12),
                        stamp.strftime("%m/%d/%Y %I:%M:%S %p"),
                        history,
                        candidates,
                    ]
                )
                counter += 1
        pd.DataFrame(rows).to_csv(
            d / "behaviors.tsv", sep="\t", header=False, index=False
        )
    glove = base / "glove"
    glove.mkdir()
    (glove / "glove.6B.300d.txt").write_text(
        "\n".join(
            w + " " + " ".join(map(str, rng.normal(size=300)))
            for w in ["news", "title", "article", "sports", "business"]
        )
    )
    emb = rng.normal(size=(31, 12)).astype(np.float32)
    emb[0] = 0
    emb[1:] /= np.linalg.norm(emb[1:], axis=1, keepdims=True)
    np.save(features / "article_content_emb.npy", emb)
    (features / "article_content_emb_ids.json").write_text(json.dumps(ids))
    execute("data_preparation/2_preprocessing.ipynb")
    from utils.data_utils import content_identity, load_preprocessed, project_paths
    _, fixture_news = load_preprocessed(project_paths(ROOT))
    fixture_texts = (fixture_news["title"] + ". " + fixture_news["abstract"]).tolist()
    (features / "article_content_emb_config.json").write_text(json.dumps({
        "identity": content_identity({"texts": fixture_texts,
            "text_model": "sentence-transformers/all-MiniLM-L6-v2", "normalize": True})
    }))
    feature_scope = execute("data_preparation/3_feature_engineering.ipynb")
    exported_content = np.load(features / "article_content_emb.npy")
    exported_kg = np.load(features / "kg_entity_emb.npy")
    assert np.all(exported_content[0] == 0) and np.all(exported_kg[0] == 0)
    assert np.all(feature_scope["user_emb"][0] == 0)
    assert np.all(feature_scope["user_aff"][0] == 0)
    for _, candidates in feature_scope["tables"].values():
        assert candidates["category_match"].isin([0, 1]).all()
    assert not (features / "cbf_content_emb.npy").exists()
    print("PASS canonical embedding, cold-profile and binary category-match contracts")
    from utils.data_utils import bind_feature_run
    bind_feature_run(project_paths(ROOT), __import__("utils.data_utils", fromlist=["validate_features"]).validate_features(project_paths(ROOT)))
    eda_scope = execute("data_preparation/1_eda.ipynb")
    for name in [
        "table5_numeric_statistics.csv",
        "table6_categorical_statistics.csv",
        "table7_text_kg_statistics.csv",
        "split_comparison.csv",
    ]:
        assert (eda_scope["TABLE_DIR"] / name).exists()
    assert (eda_scope["FIGURE_DIR"] / "category_match_has_history.png").exists()
    print("PASS computed interim-report EDA artifacts")
    for p in [
        "models/4_base_models/4_1_popularity.ipynb",
        "models/4_base_models/4_5_popularity_smoothed.ipynb",
        "models/4_base_models/4_2_ubcf.ipynb",
        "models/4_base_models/4_3_cbf.ipynb",
        "models/5_ubcf_cbf/5_1_ubcf.ipynb",
        "models/5_ubcf_cbf/5_2_cbf.ipynb",
        "models/5_ubcf_cbf/5_3_popularity.ipynb",
        "models/5_ubcf_cbf/5_4_stacking.ipynb",
        "models/6_mlp/6_1_mlp.ipynb",
        "models/6_mlp/6_2_mlp_attention.ipynb",
        "models/4_base_models/4_4_nrms.ipynb",
        "models/7_nrms/7_1_nrms_popularity_gated.ipynb",
    ]:
        scope = execute(p, limited)
        if p.endswith("7_1_nrms_popularity_gated.ipynb"):
            reference = scope["NRMS"](
                scope["pretrained_embeddings"], scope["news_title_lookup"]
            ).to(scope["device"])
            result = scope["evaluate_global_popularity_fallback"](reference)
            assert len(result) == 2 * len(scope["test_records"])
            assert set(result["variant"]) == {"nrms", "global_top20_fallback"}
            print("PASS global popularity fallback inference fixture")
    from utils.data_utils import project_paths, validate_features

    import shutil
    from utils.data_utils import load_preprocessed, bind_feature_run
    original_id = validate_features(project_paths(ROOT))
    moved = base / "moved_features"
    shutil.copytree(features, moved)
    relocated = dict(project_paths(ROOT), features=moved, raw=base / "absent_raw")
    assert validate_features(relocated) == original_id
    print("PASS relocated feature bundle without raw data")
    changed_source = raw / "train" / "behaviors.tsv"
    changed_source.write_text(changed_source.read_text() + "\n")
    assert validate_features(project_paths(ROOT)) == original_id
    try:
        load_preprocessed(project_paths(ROOT))
    except ValueError:
        print("PASS changed raw source rejects stale preprocessing")
    else:
        raise AssertionError("Changed raw inputs must invalidate preprocessing")
    damaged = moved / "kg_entity_emb.npy"
    with damaged.open("ab") as stream:
        stream.write(b"changed")
    try:
        validate_features(relocated)
    except ValueError:
        print("PASS changed exported artifact rejected")
    else:
        raise AssertionError("Changed feature contents must be rejected")
    try:
        bind_feature_run(project_paths(ROOT), "different-feature-id")
    except ValueError:
        print("PASS output directory rejects another feature bundle")
    else:
        raise AssertionError("Checkpoints must remain bound to their feature bundle")
    print(
        "All fixture workflows passed; training smoke tests use one epoch and disable optional retraining."
    )
