# BT4222 Group 12

MIND-small news recommendation: data exploration, shared preparation, basic recommenders, hybrid ranking, feature-selected MLP and popularity-gated NRMS.

## Repository structure

```text
bt4222-group12/
├── README.md
├── requirements.txt                       # Python dependencies
├── REFACTOR_PLAN.md                        # Refactor scope and acceptance criteria
├── data_preparation/                       # Shared data pipeline
│   ├── 1_eda.ipynb                         # Interaction, content and temporal analysis
│   ├── 2_preprocessing.ipynb               # Source tables, identities and temporal splits
│   └── 3_feature_engineering.ipynb         # Shared features and model input bundles
├── models/                                 # Model fitting, scoring and evaluation
│   ├── 4_base_models/                      # Standalone baselines
│   │   ├── 4_1_popularity.ipynb            # Frozen training click counts
│   │   ├── 4_2_ubcf.ipynb                  # User-based collaborative filtering
│   │   ├── 4_3_cbf.ipynb                   # Content-based filtering
│   │   ├── 4_4_nrms.ipynb                  # Original NRMS architecture
│   │   └── 4_5_popularity_smoothed.ipynb   # Smoothed CTR and stacking score exports
│   ├── 5_ubcf_cbf/                         # Advanced scorers and hybrid stacking
│   │   ├── 5_1_ubcf.ipynb
│   │   ├── 5_2_cbf.ipynb
│   │   └── 5_4_stacking.ipynb              # Combines 4_5, 5_1 and 5_2 scores
│   ├── 6_mlp/
│   │   ├── 6_1_mlp.ipynb                  # Initial and improved MLP experiments
│   │   └── 6_2_mlp_diagnostics.ipynb      # Saved-model plots and hour-feature ablation
│   └── 7_nrms/
│       └── 7_1_nrms_popularity_gated.ipynb # Gated NRMS, seed runs and comparisons
├── utils/                                  # Shared helpers across model families
│   ├── data_utils.py                      # Paths, table I/O and artifact checks
│   ├── metrics.py                         # Ranking metrics and evaluation helpers
│   ├── metrics.ipynb                      # Metric documentation and examples
│   └── __init__.py
├── tests/                                  # Validation and notebook execution tools
│   ├── verify_notebooks.py                # Notebook structure, syntax and contracts
│   ├── verify_pipeline.py                 # Synthetic pipeline and training smoke checks
│   ├── run_cpu_pipeline.py                # Selected real-data CPU workflows
│   └── run_local_pipeline.py              # Local notebook execution
├── docs/                                   # Technical documentation
│   ├── FEATURES.md                        # Feature reference
│   ├── migration.md                       # Migration notes
│   └── validation.md                      # Validation evidence and limitations
├── reports/                                # Submitted reports and historical artifacts
│   ├── BT4222_FinalReport.md              # Also available as PDF
│   ├── BT4222_InterimReport.md            # Also available as PDF
│   ├── figures/
│   ├── tables/
│   └── results/                           # Historical result JSONs
└── references/                             # Supporting material
    ├── dataset_links.md                   # Dataset download references
    ├── guidelines/                        # Course deliverable guidelines
    ├── examples/                          # Teaching example notebooks
    └── project_notes/                     # Original project and model notes
```

The tree shows source files and groups report/reference assets by folder. Raw data, generated features, caches, checkpoints and new experiment outputs live in configured local or Google Drive locations; they are separate from the historical artifacts in `reports/` and are not committed to Git.

Read the pipeline as follows (notebook numbers label stages, not a strict execution order):

```text
Raw MIND data
    └── 2_preprocessing ──┬── 1_eda (descriptive analysis)
                          └── 3_feature_engineering
                                  ├── 4_* baselines
                                  ├── 5_1 UBCF + 5_2 CBF + 4_5 popularity
                                  │       └── 5_4 stacking
                                  ├── 6_1 MLP ── 6_2 diagnostics
                                  └── 7_1 popularity-gated NRMS
```

Model-specific code stays in its notebook. For stacking, finish the score exports from `4_5`, `5_1` and `5_2` first. MLP diagnostics load the saved improved model from `6_1`; NRMS paired comparisons require matching reference and gated checkpoints. See **Execution** and **Run on Google Colab** below for details.

## Shared Google Drive storage

The [Group 12 shared Drive folder](https://drive.google.com/drive/u/0/folders/1ag8e78lQXeWicnx39IpkwBZ4fFdjytLD) stores raw data, prepared inputs, experiment artifacts and working report/presentation files. The layout below was verified with a read-only recursive rclone listing on **2026-11-09**. It is a compact inventory: CSV files are omitted, and repeated experiment files are grouped using `{...}` and `*`.

```text
BT4222 - Group 12/                          # Shared project data root (BASE in Colab)
├── BT4222 Final Presentation.pptx
├── BT4222 Final Report.docx
├── BT4222 Interim Report.docx
├── data/                                  # Raw MIND-small data and pretrained GloVe
│   ├── train/
│   │   ├── news.tsv
│   │   ├── behaviors.tsv
│   │   ├── entity_embedding.vec
│   │   └── relation_embedding.vec
│   ├── dev/                               # Same four filenames as train/
│   └── glove/
│       └── glove.6B.300d.txt
├── processed/
│   └── preprocessing/
│       ├── news.parquet
│       ├── behaviors.parquet
│       └── metadata.json
├── features/                              # Shared feature bundle consumed by models
│   ├── feature_manifest.json
│   ├── feature_meta.json
│   ├── articles.parquet
│   ├── user_profiles.parquet
│   ├── article_content_emb.npy
│   ├── article_content_emb_{config,ids}.json
│   ├── kg_entity_emb.npy
│   ├── user_category_affinity.npy
│   ├── user_history_{emb,indices}.npy
│   ├── hybrid_news.pkl
│   ├── hybrid_user_state.pkl
│   ├── mlp_inputs.pkl
│   ├── nrms_features.pkl
│   └── {train,val,test}/
│       ├── impressions.parquet
│       ├── groups.parquet
│       └── candidates.parquet
├── outputs/                               # Separate score/cache location
│   ├── cache/
│   └── scores_ubcf_{stack_train,val,test}.parquet
└── runs/                                  # Results separated by experiment run
    ├── run_01/
    │   ├── feature_identity.json
    │   ├── base_models/                    # popularity, popularity_smoothed, ubcf, cbf, nrms
    │   ├── ubcf_cbf/                       # Scorer/stacker scores and cache
    │   ├── eda/                            # figures/ and tables/
    │   ├── nrms/                           # checkpoints/ and figures/ (including result JSONs)
    │   └── mlp/
    │       ├── initial/                    # Weights, metrics, predictions and ablation_runs/
    │       ├── improved/                   # Weights, metrics, predictions and ablation_runs/
    │       ├── attention/
    │       ├── comprehensive_ablation/     # runs/ and figures/
    │       └── staged/                     # stage1_initial through stage4_final
    ├── run_02/
    │   ├── feature_identity.json
    │   ├── pipeline.log
    │   ├── pipeline_rerun.log
    │   ├── base_models/                    # Same five model folders as run_01
    │   ├── ubcf_cbf/                       # UBCF, CBF and hybrid score exports
    │   ├── eda/                            # figures/ and tables/
    │   ├── nrms/                           # Seed checkpoints, figures and result JSONs
    │   ├── mlp/                            # initial/, improved/ and attention/
    │   └── executed/                       # Executed notebook copies and .done markers
    └── run_temp/
        ├── feature_identity.json
        ├── mlp/saved_best_hour_ablation/    # Hour ablation weights, vectors and figures
        └── nrms/latent_gated_seed42/        # Latent vectors, PCA and visualization
```

Use `data/` for raw inputs, `processed/preprocessing/` for cleaned tables, `features/` for model inputs, and `runs/<RUN_NAME>/` for checkpoints, predictions, metrics and figures. For example, MLP weights and metrics are at `runs/run_02/mlp/improved/mlp_best.pt` and `metrics.json`; NRMS checkpoints are at `runs/run_02/nrms/checkpoints/`. Existing Drive artifacts are listed by filename only; this inventory does not establish that every checkpoint matches the current code or feature bundle. The `executed/` folder also retains older notebook names.

In Colab, make this folder accessible under My Drive (add a shortcut if it is shared with you), then set `BASE = Path("/content/drive/MyDrive/BT4222 - Group 12")` in the setup cell. Adjust `BASE` if your shortcut has a different name or location. The setup maps `MIND_DATA_DIR` to `BASE/data`, `BT4222_PROCESSED_DIR` to `BASE/processed`, `BT4222_FEATURE_DIR` to `BASE/features`, and `BT4222_OUTPUT_DIR` to `BASE/runs/RUN_NAME`.

To refresh the listing with an authenticated rclone Drive remote (replace `YOUR_REMOTE` with your configured remote name):

```sh
rclone lsf YOUR_REMOTE: --drive-root-folder-id 1ag8e78lQXeWicnx39IpkwBZ4fFdjytLD --recursive --exclude '*.csv' --exclude '*.CSV' --tpslimit 1
```

## Environment and locations

Install `requirements.txt` in a Python environment with a Jupyter kernel. GPU PyTorch is recommended for real training. LightGBM is optional; the existing stacker uses its sklearn alternative when it is unavailable. Parquet is optional; table I/O falls back to pickle.

The default raw location is `data/raw`. Existing `data/train` and `data/dev` layouts, `valid` as the development source name, and `MINDsmall_train`/`MINDsmall_dev` folders are supported. Set `MIND_DATA_DIR` to select one dataset unambiguously. Raw folders contain `news.tsv`, `behaviors.tsv`, and entity vectors. Set `BT4222_GLOVE_PATH` to the existing `glove.6B.300d.txt` file; the default is `data/glove/glove.6B.300d.txt`.

Configure environment variables before the setup cell:

| Variable | Purpose |
|---|---|
| `MIND_DATA_DIR` | Raw source parent directory |
| `BT4222_BASE` | Local or Drive project data root |
| `BT4222_PROCESSED_DIR` | Preprocessing output directory |
| `BT4222_FEATURE_DIR` | Feature artifact directory |
| `BT4222_OUTPUT_DIR` | Generated experiment results |
| `BT4222_GLOVE_PATH` | GloVe text file |
| `BT4222_USE_DRIVE=1` | Mount Google Drive in Colab |
| `HF_TOKEN` | Optional Hugging Face dataset token |

The notebook Colab setup uses `/content/drive/MyDrive/BT4222 - Group 12` and its `features/` folder as described above. When the shared path helper is used without these setup overrides, its legacy Drive defaults are `/content/drive/MyDrive/Y4S1/BT4222` and the `MLP/` feature folder; configure the variables above to select the shared layout. Open notebooks from the repository or their containing directory so `utils` is accessible. In Colab, make the repository available before running setup. The optional original Hugging Face download is in preprocessing; existing links are retained in [dataset references](references/dataset_links.md).

## Execution

1. Run `2_preprocessing.ipynb` to export source-aware tables and the 5-1-1 partition.
2. Run `3_feature_engineering.ipynb` to prepare all required inputs, including MLP standardization and attention histories. Its metadata documents canonical names and distinct index spaces. Article-based consumers use the same MiniLM content vectors and normalized TransE entity vectors; missing entity vectors are zero. No second CBF text encoder or fallback representation is generated.
3. Run `1_eda.ipynb`. The raw-data sections can run after preprocessing; the descriptive sections calculate statistics directly from the preprocessed tables. EDA is not an upstream dependency of model fitting.
4. Run the required baseline notebooks independently. For hybrid stacking, finish `4_5` (including its final stacking export), `5_1` and `5_2`, then run `5_4`. The stacker consumes canonical 4_5 popularity scores rather than a separately tuned popularity model.
5. Run the MLP or NRMS experiments. Initial/improved MLP experiments use separate directories. Enable NRMS paired comparisons only after the reference and gated seed checkpoints have been generated with the advanced notebook's matching evaluation protocol.

Training, ablation and multi-seed cells can be expensive. Their original defaults are retained. Model notebooks load prepared features rather than running feature engineering through notebook execution.

Generated artifacts belong under configured local/Drive locations and are ignored by Git. Embedding caches identify input text and encoder settings. The final feature manifest verifies exported file contents independently of their location. Regenerate the complete bundle when preparation changes and choose a new experiment output directory. Legacy bundles without a manifest must be rebuilt. Historical reports/results under `reports` are not live training outputs.

## Evaluation

Report AUC, MRR, nDCG@5 and nDCG@10. User cohorts use history at the impression: cold = 0, sparse = 1-4, warm = at least 5. Original detailed model diagnostics may additionally report other non-exclusive groups.

Existing metric semantics are preserved: MLP averages reciprocal ranks over clicked items; hybrid and NRMS use the first clicked rank. Tie and one-class AUC behavior also follow their source evaluators. Cold-article reporting retains each source workflow's click-history window and candidate selection. Hybrid uses build-window click/history knowledge; MLP uses its existing lagged click feature and removes clicked warm candidates in its cold-article ranking. The original NRMS evaluator has no article-cold cohort; no new definition is fabricated. Do not interpret scores from different evaluator or information-window protocols as directly interchangeable.

## Verification

Run `python tests/verify_notebooks.py` for structure, syntax, imports and layout contracts. Run `python tests/verify_pipeline.py` for the temporary synthetic pipeline, including one-epoch training smoke tests. It uses seeded cached text-vector fixtures, not downloaded pretrained MiniLM weights, and disables optional repeated training. No fixture metric is a project benchmark.

See [migration notes](docs/migration.md), [validation](docs/validation.md), and [refactor plan](REFACTOR_PLAN.md).

CPU execution: `python tests/run_cpu_pipeline.py preprocessing eda popularity` runs real data and saves execution counts, text, rich tables and plots directly in the source notebooks, with review copies and cell logs under `outputs/cpu_validation`. Notebooks not executed on real inputs retain empty outputs. Formatting checks allow valid actual execution output and reject error output.

The additional `4_5_popularity_smoothed.ipynb` baseline uses alpha=20, a training CTR prior and the MLP-compatible preceding-day update protocol. It reports standalone smoothed CTR and frozen click counts using the same MLP evaluator. Run `python tests/run_cpu_pipeline.py smoothed_popularity` to produce real-data output in the notebook. The learned MLP fallback remains a separate combination rule.

The complete implemented feature dictionary, formulas, missing-value conventions, sequence inputs and artifact inventory are in the opening Markdown cells of [feature engineering](data_preparation/3_feature_engineering.ipynb).

Feature split tables are stored under `BT4222_FEATURE_DIR/train/`, `val/` and `test/`. Each contains `impressions`, `groups` and `candidates` tables (Parquet or pickle). Shared catalog/profile tables, arrays and model input bundles remain at the feature root.

## Run on Google Colab

Open each notebook from the same pushed GitHub branch. Every pipeline notebook starts with a Colab setup cell. Edit `REPO_URL`, `BRANCH`, `RUN_NAME` and `BASE`, then run the cell; it mounts Drive, clones the selected branch and installs requirements. A monorepo containing `bt4222-group12/` and a standalone project repo are both supported. Private repositories require GitHub authentication; do not put access tokens in notebook source. Use a fresh runtime when changing the checked-out code.

Place raw files at `BASE/data/train/` and `BASE/data/dev/` (each contains `news.tsv`, `behaviors.tsv`, `entity_embedding.vec`), and GloVe at `BASE/data/glove/glove.6B.300d.txt`. Existing preprocessing/features made before the portable manifest revision must be regenerated into empty configured directories; keep old outputs separately.

Run in this order:

1. `2_preprocessing.ipynb` (CPU).
2. `1_eda.ipynb` (CPU; can run independently after preprocessing).
3. `3_feature_engineering.ipynb` (GPU recommended; needs MiniLM download and GloVe). Finish the final **Completed feature bundle** cell.
4. Baselines `4_1` through `4_5` (independent; GPU recommended for NRMS).
5. Finish `4_5` popularity, `5_1` UBCF and `5_2` CBF, then `5_4` stacking. Use the same output root for these notebooks.
6. `6_1_mlp.ipynb`, then `6_2_mlp_diagnostics.ipynb` for saved-model diagnostics and hour-feature ablation (GPU recommended).
7. `7_1_nrms_popularity_gated.ipynb` (GPU recommended). Inspect optional multi-seed/ablation switches and budgets before running expensive sections.

All notebooks in one experiment use the same `RUN_NAME`; the models have separate subfolders. Use `run_02` for a fresh experiment instead of reusing `run_01`. `initial` and `improved` are MLP variants, not run counters. Save/download each executed notebook to retain its displayed outputs; experiment artifacts are written separately to `BASE/runs/RUN_NAME/`.

MLP and NRMS validate only the exported feature bundle and can use it without raw files. EDA, popularity and hybrid workflows also use preprocessed tables. When raw inputs are available, loading preprocessing verifies their content hashes rather than absolute paths or timestamps. A feature bundle is published only after all exports complete; changed/missing files are rejected. Output roots record the feature identity so checkpoints cannot silently cross prepared datasets.

The `4_5` final export writes `stack_train_scores`, `val_scores` and `test_scores` under `base_models/popularity_smoothed/`, plus `stacking_scores.json` with dataset provenance, phase cutoffs, priors and file hashes. Stack-training scores use days 0–3 to score day 4; their prior also uses only days 0–3. Validation/test retain the original 4_5 scoring policy. The former `5_3_popularity.ipynb` is removed. After updating, run 4_5 through its final cell and rerun 5_4; existing compatible UBCF/CBF scores can be reused. No feature regeneration is required for this scoring change.
