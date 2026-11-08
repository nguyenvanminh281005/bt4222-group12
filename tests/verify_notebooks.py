"""Validate notebook structure, Python syntax and project layout contracts."""

import ast
import builtins
import json
import symtable
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def undefined_globals(source, filename):
    table = symtable.symtable(source, filename, "exec")
    defined = {
        item.get_name()
        for item in table.get_symbols()
        if item.is_assigned() or item.is_imported()
    }
    referenced = set()

    def collect(scope):
        for item in scope.get_symbols():
            if item.is_global() and item.is_referenced():
                referenced.add(item.get_name())
        for child in scope.get_children():
            collect(child)

    collect(table)
    return referenced - defined - set(dir(builtins)) - {"display", "__file__"}


def main():
    active = [
        path
        for path in ROOT.rglob("*.ipynb")
        if not ({"references", "outputs", "data"} & set(path.relative_to(ROOT).parts))
        and ".ipynb_checkpoints" not in path.parts
        and not any(part.startswith(".") for part in path.relative_to(ROOT).parts)
    ]
    expected = {
        "1_eda.ipynb",
        "2_preprocessing.ipynb",
        "3_feature_engineering.ipynb",
    }
    assert {
        path.name for path in (ROOT / "data_preparation").glob("*.ipynb")
    } == expected
    assert {path.name for path in active if "nrms" in path.name} == {
        "4_4_nrms.ipynb", "7_1_nrms_popularity_gated.ipynb", "7_2_nrms_saved_latent.ipynb"
    }
    assert not list((ROOT / "models").rglob("*.py"))
    assert {path.name for path in (ROOT / "utils").glob("*.py")} == {
        "__init__.py",
        "data_utils.py",
        "metrics.py",
    }
    for path in active:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        source = "\n".join(
            cell.source for cell in notebook.cells if cell.cell_type == "code"
        )
        ast.parse(source, filename=str(path))
        missing = undefined_globals(source, str(path))
        assert not missing, (path, missing)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                assert cell.execution_count is None or isinstance(
                    cell.execution_count, int
                ), path
                assert not any(
                    output.output_type == "error" for output in cell.outputs
                ), path
        print("PASS", path.relative_to(ROOT))
    for path in (ROOT / "utils").glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    print(f"Validated {len(active)} active notebooks and utility modules")


if __name__ == "__main__":
    main()
