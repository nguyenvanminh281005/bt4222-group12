"""Execute the full pipeline locally, in README order, with one Jupyter kernel.

Usage (from the repository root):
    python tests/run_local_pipeline.py                 # all steps, executed copies
    python tests/run_local_pipeline.py --inplace       # save outputs into the notebooks
    python tests/run_local_pipeline.py --from 4_1_popularity
    python tests/run_local_pipeline.py --only 7_1_nrms 7_1_gated

Executed copies go to outputs/<RUN_NAME>/executed/. Finished steps are marked
with a .done file and skipped on restart (use --force to rerun them). 7_1 is
run twice (NRMS_VARIANT = "nrms", then "gated") so the paired comparison runs
in the second pass. Set BT4222_* variables beforehand to override locations;
otherwise each notebook's local setup block is used.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]

STEPS = [
    ("2_preprocessing", "data_preparation/2_preprocessing.ipynb", None),
    ("3_feature_engineering", "data_preparation/3_feature_engineering.ipynb", None),
    ("1_eda", "data_preparation/1_eda.ipynb", None),
    ("4_1_popularity", "models/4_base_models/4_1_popularity.ipynb", None),
    ("4_2_ubcf", "models/4_base_models/4_2_ubcf.ipynb", None),
    ("4_3_cbf", "models/4_base_models/4_3_cbf.ipynb", None),
    ("4_4_nrms", "models/4_base_models/4_4_nrms.ipynb", None),
    ("4_5_popularity_smoothed", "models/4_base_models/4_5_popularity_smoothed.ipynb", None),
    ("5_1_ubcf", "models/5_ubcf_cbf/5_1_ubcf.ipynb", None),
    ("5_2_cbf", "models/5_ubcf_cbf/5_2_cbf.ipynb", None),
    ("5_4_stacking", "models/5_ubcf_cbf/5_4_stacking.ipynb", None),
    ("6_1_mlp", "models/6_mlp/6_1_mlp.ipynb", None),
    ("6_2_mlp_attention", "models/6_mlp/6_2_mlp_attention.ipynb", None),
    ("7_1_nrms", "models/7_nrms/7_1_nrms_popularity_gated.ipynb", "nrms"),
    ("7_1_gated", "models/7_nrms/7_1_nrms_popularity_gated.ipynb", "gated"),
]


def set_variant(nb, variant):
    line = 'NRMS_VARIANT = "gated"'
    cells = [c for c in nb.cells if c.cell_type == "code" and line in c.source]
    assert len(cells) == 1, "NRMS_VARIANT line not found in 7_1"
    cells[0].source = cells[0].source.replace(line, f'NRMS_VARIANT = "{variant}"')


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="run_01")
    p.add_argument("--kernel", default="python3")
    p.add_argument("--inplace", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--from", dest="start")
    p.add_argument("--only", nargs="+")
    a = p.parse_args()

    names = [s[0] for s in STEPS]
    steps = STEPS
    if a.start:
        steps = STEPS[names.index(a.start):]
    if a.only:
        steps = [s for s in STEPS if s[0] in a.only]

    os.environ.setdefault("BT4222_OUTPUT_DIR", str(ROOT / "outputs" / a.run_name))
    os.environ.setdefault("MPLBACKEND", "Agg")
    out = Path(os.environ["BT4222_OUTPUT_DIR"]) / "executed"
    out.mkdir(parents=True, exist_ok=True)

    for name, rel, variant in steps:
        done = out / f"{name}.done"
        if done.exists() and not a.force:
            print(f"skip {name}", flush=True)
            continue
        src = ROOT / rel
        nb = nbformat.read(src, as_version=4)
        if variant:
            set_variant(nb, variant)
        print(f"=== {name}  {time.strftime('%H:%M:%S')}", flush=True)
        t = time.time()
        client = NotebookClient(
            nb, timeout=None, kernel_name=a.kernel,
            resources={"metadata": {"path": str(src.parent)}},
        )
        ok = True
        try:
            client.execute()
        except BaseException as e:  # keep the partial notebook for inspection
            ok = False
            print(f"FAILED {name}: {str(e)[-3000:]}", flush=True)
        dst = out / f"{name}.ipynb"
        if ok and a.inplace and not variant:
            dst = src
        elif ok and a.inplace and variant == "gated":
            dst = src  # the final 7_1 pass (with comparison) is kept in the source
        elif not ok:
            dst = out / f"{name}.failed.ipynb"
        dst.write_text(
            json.dumps(json.loads(nbformat.writes(nb)), indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if not ok:
            sys.exit(1)
        done.write_text("")
        print(f"OK {name}  {(time.time() - t) / 60:.1f} min -> {dst}", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
