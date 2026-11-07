"""Cross-model locations and table I/O."""

from pathlib import Path
import json
import hashlib
import os
import pandas as pd

NEWS_COLUMNS = [
    "news_id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
]
BEHAVIOR_COLUMNS = ["orig_impression_id", "user_id", "time", "history", "impressions"]


def project_paths(project_root):
    root = Path(project_root).resolve()
    drive = os.environ.get("BT4222_USE_DRIVE") == "1"
    base = Path(
        os.environ.get(
            "BT4222_BASE",
            "/content/drive/MyDrive/Y4S1/BT4222" if drive else str(root / "data"),
        )
    )
    raw = Path(os.environ.get("MIND_DATA_DIR", str(base if drive else base / "raw")))
    processed = Path(os.environ.get("BT4222_PROCESSED_DIR", str(base / "processed")))
    features = Path(
        os.environ.get(
            "BT4222_FEATURE_DIR", str(base / "MLP" if drive else processed / "features")
        )
    )
    outputs = Path(
        os.environ.get(
            "BT4222_OUTPUT_DIR", str(base / "outputs" if drive else root / "outputs")
        )
    )
    return dict(
        root=root,
        raw=raw,
        processed=processed,
        features=features,
        outputs=outputs,
        glove=Path(
            os.environ.get(
                "BT4222_GLOVE_PATH", str(base / "glove" / "glove.6B.300d.txt")
            )
        ),
    )


def source_directory(raw, source):
    raw = Path(raw)
    names = (
        ("train", "MINDsmall_train")
        if source == "train"
        else ("dev", "valid", "MINDsmall_dev")
    )
    roots = [raw, raw.parent] if raw.name == "raw" else [raw]
    matches = [
        r / n for r in roots for n in names if (r / n / "behaviors.tsv").is_file()
    ]
    if not matches and raw.exists():
        matches = [
            p for n in names for p in raw.rglob(n) if (p / "behaviors.tsv").is_file()
        ]
    if len(matches) != 1:
        raise ValueError(f"Expected one {source} source under {raw}, found {matches}")
    return matches[0]


def save_table(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(str(path) + ".parquet", index=False)
    except ImportError:
        df.to_pickle(str(path) + ".pkl")


def load_table(path):
    path = Path(path)
    if Path(str(path) + ".parquet").is_file():
        return pd.read_parquet(str(path) + ".parquet")
    return pd.read_pickle(str(path) + ".pkl")


def load_preprocessed(paths):
    directory = Path(paths["processed"]) / "preprocessing"
    metadata = json.loads((directory / "metadata.json").read_text())
    if Path(paths["raw"]).exists() and "source_files" in metadata and metadata["source_files"] != source_fingerprint(
        paths
    ):
        raise ValueError(
            "Raw inputs changed. Run preprocessing and regenerate feature artifacts."
        )
    return load_table(directory / "behaviors"), load_table(directory / "news")


def source_fingerprint(paths):
    """Identify source files used by prepared inputs without reading model artifacts."""
    records = {}
    for source in ("train", "dev"):
        directory = source_directory(paths["raw"], source)
        for name in ("news.tsv", "behaviors.tsv", "entity_embedding.vec"):
            path = directory / name
            if path.is_file():
                state = path.stat()
                records[f"{source}/{name}"] = {
                    "size": state.st_size,
                    "sha256": file_digest(path),
                }
    return records


def file_digest(path):
    """Hash file contents independently of their location or modification time."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_identity(value):
    """Return a stable identity for JSON-compatible preparation settings."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def publish_features(paths, preparation):
    """Publish a complete portable feature bundle after all exports succeed."""
    directory = Path(paths["features"])
    files = {p.relative_to(directory).as_posix(): file_digest(p)
             for p in sorted(directory.rglob("*"))
             if p.is_file() and p.suffix in {".json", ".npy", ".pkl", ".parquet"}
             and p.name != "feature_manifest.json"}
    manifest = {"schema_version": 1, "preparation": preparation, "files": files}
    manifest["feature_id"] = content_identity(manifest)
    target = directory / "feature_manifest.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(target)
    return manifest["feature_id"]


def validate_features(paths):
    """Validate exported contents without requiring raw data at model runtime."""
    directory = Path(paths["features"])
    target = directory / "feature_manifest.json"
    if not target.is_file():
        raise ValueError("Feature bundle is incomplete or outdated. Run feature engineering to completion.")
    manifest = json.loads(target.read_text(encoding="utf-8"))
    identity = manifest.pop("feature_id")
    if manifest.get("schema_version") != 1 or identity != content_identity(manifest):
        raise ValueError("Invalid feature manifest. Rebuild the feature bundle.")
    for name, expected in manifest["files"].items():
        path = directory / name
        if not path.is_file() or file_digest(path) != expected:
            raise ValueError(f"Missing or changed feature artifact: {name}. Rebuild features.")
    return identity


def bind_feature_run(paths, feature_id):
    """Keep checkpoints and score caches associated with one prepared bundle."""
    output = Path(paths["outputs"])
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "feature_identity.json"
    if marker.exists():
        if json.loads(marker.read_text())["feature_id"] != feature_id:
            raise ValueError("Output directory belongs to different features. Choose a new BT4222_OUTPUT_DIR.")
    elif (any((output / name).exists() for name in ("mlp", "nrms", "ubcf_cbf"))
          or any((output / "base_models" / name).exists()
                 for name in ("ubcf", "cbf", "nrms"))):
        raise ValueError("Existing output directory has no feature identity. Choose a new BT4222_OUTPUT_DIR.")
    else:
        marker.write_text(json.dumps({"feature_id": feature_id}), encoding="utf-8")
