"""Run selected data notebooks on real inputs and retain executed review copies."""

import argparse
import ast
import base64
import contextlib
import io
import json
import os
import time
import traceback
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "cpu_validation"
OUTPUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT / "matplotlib"))

import matplotlib.pyplot as plt


def run_notebook(relative_path):
    notebook = json.loads((ROOT / relative_path).read_text(encoding="utf-8"))
    scope = {"__name__": "notebook"}
    started = time.time()
    results = []
    active_cell = 0
    active_outputs = []
    stream = io.StringIO()
    stream_position = 0

    def flush_stream():
        nonlocal stream_position
        contents = stream.getvalue()
        if len(contents) > stream_position:
            active_outputs.append(
                {
                    "output_type": "stream",
                    "name": "stdout",
                    "text": contents[stream_position:].splitlines(True),
                }
            )
            stream_position = len(contents)

    def show():
        flush_stream()
        for number in plt.get_fignums():
            figure = plt.figure(number)
            figure.savefig(
                OUTPUT / f"{Path(relative_path).stem}_cell{active_cell}_{number}.png"
            )
            image = io.BytesIO()
            figure.savefig(image, format="png", bbox_inches="tight")
            active_outputs.append(
                {
                    "output_type": "display_data",
                    "metadata": {},
                    "data": {
                        "image/png": base64.b64encode(image.getvalue()).decode("ascii"),
                        "text/plain": [repr(figure)],
                    },
                }
            )
        plt.close("all")

    def display(*values):
        flush_stream()
        for value in values:
            if value is None:
                continue
            data = {"text/plain": [repr(value)]}
            if hasattr(value, "_repr_html_"):
                html = value._repr_html_()
                if html:
                    data["text/html"] = [html]
            active_outputs.append(
                {"output_type": "display_data", "metadata": {}, "data": data}
            )

    plt.show = show
    scope["display"] = display
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        active_cell = index
        print(f"RUN {relative_path} cell {index}", flush=True)
        stream = io.StringIO()
        stream_position = 0
        active_outputs = []
        cell_started = time.time()
        error = None
        try:
            with contextlib.redirect_stdout(stream):
                filename = f"{relative_path}:cell{index}"
                tree = ast.parse("".join(cell["source"]), filename=filename)
                expression = (
                    tree.body.pop()
                    if tree.body and isinstance(tree.body[-1], ast.Expr)
                    else None
                )
                exec(compile(tree, filename, "exec"), scope)
                if expression is not None:
                    display(
                        eval(
                            compile(ast.Expression(expression.value), filename, "eval"),
                            scope,
                        )
                    )
        except Exception:
            error = traceback.format_exc()
        cell["execution_count"] = len(results) + 1
        flush_stream()
        cell["outputs"] = active_outputs
        if error:
            cell["outputs"].append(
                {
                    "output_type": "error",
                    "ename": "ExecutionError",
                    "evalue": error.splitlines()[-1],
                    "traceback": error.splitlines(),
                }
            )
        results.append(
            {
                "cell": index,
                "seconds": round(time.time() - cell_started, 3),
                "error": error,
            }
        )
        if error:
            print(error, flush=True)
            break
    # Persist actual outputs with the source cells as well as the review copy.
    (ROOT / relative_path).write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    destination = OUTPUT / Path(relative_path).name
    destination.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report = {
        "notebook": relative_path,
        "seconds": round(time.time() - started, 3),
        "cells": results,
    }
    (OUTPUT / f"{Path(relative_path).stem}_run.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    if any(item["error"] for item in results):
        raise RuntimeError(f"Notebook failed; see {destination}")
    print(f"PASS {relative_path}: {report['seconds']} seconds", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stages",
        nargs="*",
        choices=["preprocessing", "eda", "popularity", "smoothed_popularity"],
        default=["preprocessing", "eda"],
    )
    args = parser.parse_args()
    os.chdir(ROOT)
    notebooks = {
        "preprocessing": "data_preparation/2_preprocessing.ipynb",
        "eda": "data_preparation/1_eda.ipynb",
        "popularity": "models/4_base_models/4_1_popularity.ipynb",
        "smoothed_popularity": "models/4_base_models/4_5_popularity_smoothed.ipynb",
    }
    for stage in args.stages:
        run_notebook(notebooks[stage])
