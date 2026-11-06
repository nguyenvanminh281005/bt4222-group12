# Group 12 repository refactor plan

Status: source consolidation implemented and fixture-validated. Real-data training and optional experiment reruns remain unverified; see docs/validation.md. Scope: bt4222-group12 only. Confirmed requirements are separated from unresolved evaluation decisions below.

## 1. Scope and preservation

- Preserve formulas, algorithms, architectures, hyperparameters, split dates, sampling policies and data meaning. Standardize variable/column names and coding conventions across notebooks using 3_feature_engineering.ipynb as the authoritative prepared-data contract. The latest naming instruction supersedes the earlier requirement to keep all original names unchanged.
- Standardize filenames, markdown, section order, imports, variable names, code layout and input/output configuration. Downstream notebooks adopt the feature notebook contract directly rather than retaining incompatible names behind post-processing renames or schema adapters.
- Use Vietnamese for discussion with the user and English for code, filenames, comments, notebook markdown and repository documentation.
- Retain all existing dataset download links, download commands, Colab/Google Drive setup and Drive import/export functionality. Organize them into explicit configuration/setup sections.
- Preserve genuine experiment variants and historical submitted reports. Record necessary behavior corrections separately from structural changes.
- Implement missing basic baselines using Group 12 contracts and inputs. Do not import Group 10 code or claim newly written code is an external original implementation.
- Do not introduce abstractions solely to eliminate repetition within one model family.

### Pipeline integrity and implementation discipline

- Preserve the original modeling intent and the declared pipeline: preprocessing produces clean partitions; feature engineering prepares canonical model inputs; model notebooks fit, score and evaluate using those inputs.
- When an inconsistency appears, trace it to its producer and check data identities, schema, ordering and original semantics before changing a consumer. Correct the owning stage and its consumers consistently.
- Do not introduce speculative variables, replacement features, arbitrary default values, alias chains, duplicate mappings or extra processing steps to make a cell run. Reuse the verified canonical data contract. New names must represent an actual required concept, follow the feature notebook conventions and preserve meaning.
- Preserve genuine model-internal structures and existing model-specific behavior; do not force superficially similar concepts into one variable or implementation.
- Keep code direct, readable and organized into functional cells. Do not add abstractions or generated boilerplate without a demonstrated pipeline need.
- Resolve routine structural issues within the agreed scope. If resolving a problem requires changing the algorithm, feature meaning, sampling, information window or original experiment intent, explain the concrete issue and proposed behavior change before implementing that change.
- Check intermediate artifacts and model inputs after each meaningful migration, rather than waiting until the whole repository has been rewritten.

## 2. Target structure

```text
bt4222-group12/
├── README.md
├── requirements.txt
├── .gitignore
├── data_preparation/
│   ├── 1_eda.ipynb
│   ├── 2_preprocessing.ipynb
│   └── 3_feature_engineering.ipynb
├── models/
│   ├── 4_base_models/
│   │   ├── 4_1_popularity.ipynb
│   │   ├── 4_2_ubcf.ipynb
│   │   ├── 4_3_cbf.ipynb
│   │   └── 4_4_nrms.ipynb
│   ├── 5_ubcf_cbf/
│   │   ├── 5_1_ubcf.ipynb
│   │   ├── 5_2_cbf.ipynb
│   │   └── 5_4_stacking.ipynb
│   ├── 6_mlp/
│   │   ├── 6_1_mlp.ipynb
│   │   └── 6_2_mlp_attention.ipynb
│   └── 7_nrms/
│       ├── 7_1_nrms_popularity_gated.ipynb
│       └── 7_2_nrms_ablation.ipynb    # Optional: substantive standalone ablation workflow
├── utils/
│   ├── __init__.py
│   ├── data_utils.py
│   ├── metrics.py
│   └── metrics.ipynb
├── references/
│   ├── guidelines/
│   ├── examples/
│   ├── project_notes/
│   └── dataset_links.md
├── reports/
│   └── figures/
├── docs/
│   ├── migration.md
│   └── validation.md
├── tests/
│   ├── verify_notebooks.py
│   └── verify_pipeline.py
├── outputs/                     # Configurable local or Drive destination
│   ├── eda/
│   ├── base_models/
│   ├── ubcf_cbf/
│   ├── mlp/
│   └── nrms/
└── data/                        # Local or Drive; datasets excluded from Git
    ├── raw/
    └── processed/
```

Smoothed popularity has one canonical producer: 4_5_popularity_smoothed.ipynb. It exports validation/test scores under its fixed-alpha daily policy and an earlier-window stack-training block with a prior fitted only before that block. Stacking reads those files directly. The duplicate tuned/online hybrid popularity notebook is removed.

Keep actual source dataset names and downloaded folder layouts accessible through configuration. Distinguish development-source folders from temporal validation/test partitions without changing observation membership.

## 3. Utility eligibility: cross-family reuse only

A function qualifies for utils only when multiple model families use it with the same processing semantics. A shared function name, use in several notebooks, or reuse between baseline and advanced versions of one family is insufficient.

- One notebook uses the code: keep it in that notebook.
- Several notebooks of one model family use the code: repeat it in those notebooks, preserving the original implementation and minimizing changes.
- Multiple families use the code with different semantics: keep separate implementations in their notebooks.
- Multiple families use exactly the same processing: place that implementation in utils and import it.

Do not create model_utils.py, utils/nrms.py, utils/mlp.py, or family-specific helper modules to deduplicate a single family. Model-specific architecture, training and scoring stay in model notebooks. All repeated family-specific code follows the same agreed naming and layout conventions. Similarities between UBCF, CBF, MLP and NRMS must be verified before extraction.

The compact utility scope is:

- data_utils.py: only proven cross-family I/O, path handling and data operations. Omit helpers that do not meet the eligibility rule.
- metrics.py: only evaluation calculations and reporting operations verified equivalent across model families. Preserve distinct existing evaluator behavior in the relevant notebooks; do not force different formulas through a universal evaluator.
- metrics.ipynb: English definitions, usage and illustrative checks importing metrics.py; no second metrics implementation.

Feature preparation has its own explicit ownership rule below and is not extracted into utility feature modules.

## 4. Data preparation ownership

### 1_eda.ipynb

This is the only project EDA notebook. Supplied teaching notebooks remain references under references/examples.

Order the consolidated analyses as follows:

1. Dataset scope, schemas, observation period, missingness and duplicate diagnostics.
2. Interaction hierarchy: users, impressions, article exposures and clicks.
3. Slate length, clicks per impression, exposure CTR and CTR by slate length.
4. Feedback semantics and repeated user-article pairs with changing labels.
5. History sparsity and cold-user distribution.
6. Article popularity concentration.
7. Consecutive-impression gaps and hourly patterns.
8. Content/category, entity coverage, reading-profile and context distributions.
9. Descriptive report tables, figures and exports.
10. Findings and interpretation.

Merge unique content from 01_eda.ipynb, 01_eda_sparsity_temporal_eda.ipynb and eda_charlene.ipynb. Move descriptive sections from the existing feature notebook into the appropriate EDA sections. Verify the provenance and equivalence of temporal CSV inputs before changing their loaders. EDA can load prepared feature artifacts for descriptive analysis; document that dependency explicitly.

### 2_preprocessing.ipynb

Own cleaning, schema preparation, source-aware identities and temporal train/validation/test partitioning. Include actual split generation, rather than only displaying split labels. Preserve alignment required by each existing model. Keep download/setup and output configuration readable and functional.

### 3_feature_engineering.ipynb

This is the only notebook defining and executing feature preparation, feature fitting/transformation, feature selection and feature exports.

- Preserve equivalent existing feature calculations; do not recompute or duplicate them. Normalize their naming consistently where necessary without changing values or meaning.
- Compare the current MLP feature notebook against those sections.
- Append distinct MLP feature requirements, feature-selection steps and required input exports at the end.
- Preserve arrays, metadata, identity mappings and row alignment. Declare their canonical names, schemas, dtypes, shapes, PAD/UNK conventions and artifact filenames in the feature notebook, and apply those names consistently in consumers.
- Fit data-dependent preparation using training data; use validation for selection decisions and retain test as held out. Record any discovered behavior correction separately.
- Model notebooks load prepared artifacts. Model batching, device transfer and forward passes remain model operations. New feature derivation belongs here.

Retire 3.1_mlp_feature_engineering.ipynb and the feature notebook generator only after all unique content and exports have verified replacements.

### Coding and prepared-data contract

3_feature_engineering.ipynb defines the canonical prepared-data names and conventions. Inspect its current implementations, retain the correct calculations, and settle one coherent contract there before rewriting consumers. The contract includes table/column names, mappings, feature order, embedding row alignment, shapes, dtypes, PAD/UNK values, saved artifacts and their loading conventions. Document it in concise functional markdown.

- Write downstream loading, indexing, model inputs, scoring and evaluation using those canonical names directly. Update producers and consumers together. Do not add alias chains, repeated renaming or post-processing conversion layers merely to accommodate inconsistent notebook naming.
- Rename equivalent concepts consistently after checking that their mappings and meanings are actually equivalent. Similar spelling alone is insufficient. Genuine semantic differences receive distinct, descriptive names and a documented role.
- Adopt one coding style: snake_case for variables/functions, PascalCase for classes, UPPER_SNAKE_CASE for configuration constants, consistent imports/spacing and ordered functional sections. Use the same contract in newly added baselines.
- Keep model-internal structures appropriate to their algorithms. A model-specific token vocabulary or batching tensor may differ from the prepared table representation; name and explain that distinction explicitly rather than disguising it as the same mapping. New feature construction and feature selection remain in the feature notebook.
- Verify naming changes through identities, row alignment and feature values, not only successful execution. Record the old-to-canonical naming map in migration documentation, not in notebook narrative.

## 5. Model organization

### Basic baselines

Keep basic popularity, UBCF, CBF and NRMS in 4_base_models. Reuse appropriate existing basic implementations; add missing ones.

- Popularity: a clearly declared basic global popularity ranking, using training interactions. Any newly implemented click-count baseline must be labeled explicitly. Preserve a distinct existing CTR/smoothing/online variant separately where justified.
- UBCF: basic user-item interactions, similarity and neighbor aggregation under a declared configuration. Advanced time decay, IUF, shrinkage and tuning remain in the advanced workflow.
- CBF: basic article representation, user-history profile and similarity scoring. Required feature artifacts come from the feature notebook.
- NRMS: Group 12's original-architecture workflow under its declared configuration.

Do not silently simplify an existing advanced implementation and present it as unchanged. Newly written baseline configurations must be documented explicitly.

### Hybrid workflows

UBCF, CBF and stacking notebooks retain their respective architecture/algorithm, configuration, scoring, evaluation and exports. Keep model-specific scoring functions in these notebooks. Only verified cross-family equivalent helpers may move to utils.

### MLP

MLP is a separate feature-selected model family, not a project baseline. Organize initial and improved experiments as clearly named configurations or sections in 6_1_mlp.ipynb while preserving their meaningful differences. Keep attention in 6_2_mlp_attention.ipynb. Allow repeated model-family code between these notebooks. Do not create an MLP helper module just to share it.

### NRMS: baseline, popularity-gated model and optional ablation

- 4_base_models/4_4_nrms.ipynb: basic architecture, configuration, training and evaluation.
- 7_nrms/7_1_nrms_popularity_gated.ipynb: popularity-gated NRMS, its configurations, multi-seed experiments and comparisons.
- 7_nrms/7_2_nrms_ablation.ipynb: optional, only when audited ablation content forms a substantive separate workflow. Keep a short ablation section in the relevant model notebook instead of creating a separate file. Do not create an empty placeholder or invent an additional experiment.

Audit and merge needed functionality from train_and_evaluate.py, evaluate_seeds.py, generate_figures_and_summary.py and the existing NRMS notebooks into these workflows or the appropriate EDA/report presentation sections. Remove replaced standalone NRMS runners after verifying coverage. Do not retain a parallel script-based NRMS workflow. The baseline and popularity-gated notebooks are the two primary NRMS notebooks; a substantive standalone ablation notebook is the only permitted third notebook, following the latest user instruction. Repeating family-specific components between these notebooks is permitted.

## 6. Scoring and evaluation

Scoring computes candidate recommendation scores using a model. Metrics consume scores and labels, group candidates by impression and calculate evaluation results. The shared metrics implementation does not train a model or generate recommendation scores.

Report AUC, MRR, nDCG@5 and nDCG@10 overall and for:

| Cohort | Definition |
|---|---|
| cold_user | 0 historical articles at the impression time |
| sparse_user | 1-4 historical articles |
| warm_user | At least 5 historical articles |
| cold_article | Existing click-history-based definition and evaluation population of the source workflow |

Preserve current cold-article behavior, including its click-history source, time window and candidate selection. Do not replace it with a new exposure-based or fixed-training-only definition. Existing implementations differ; the requirement is to retain them rather than invent a common policy.

Verified source behavior:

- Hybrid: item_cold uses the phase build log, which includes known build-window histories and clicked impression articles. Its new-item cohort selects impressions with at least one clicked cold item and evaluates their full slates.
- MLP: is_cold_article uses the existing pre-impression-day click statistics. For impressions containing a clicked cold item, the cold-article evaluator removes clicked warm items, retains all non-clicked candidates, and evaluates the resulting ranking with clicked cold items as positives. Preserve the actual configured update window.
- NRMS: the original notebook evaluator does not provide an article-cold cohort. Reuse an existing equivalent NRMS cohort if found in its scripts; otherwise document the missing cohort rather than inventing a new definition. Any extension needs an explicitly identified source behavior before implementation.

Preserve current metric formulas and edge-case behavior, including ties, multiple positives, zero-positive handling, one-class AUC and averaging. Do not assume those policies are wrong. The MLP MRR averages reciprocal ranks across clicked items; hybrid/NRMS MRR uses the first relevant rank. These are distinct existing definitions and must not be silently unified. Extract only equivalent operations into utils. Clearly identify evaluator provenance when presenting comparisons; matching metric names alone does not establish equivalence.

Regression checks establish that refactoring preserved existing behavior. Legitimate existing zero, NaN or AUC fallback conventions remain unchanged. Distinguish these from fabricated missing-experiment results such as hard-coded seed scores.

## 7. Notebook presentation and artifacts

Each notebook follows: introduction and functionality; inputs/outputs; imports; configuration/setup; data loading; ordered functional sections; checks; export; findings/limitations. Model notebooks also separate architecture, training/model selection, validation and held-out evaluation.

Use concise English markdown explaining functionality, algorithms, inputs, outputs and results. Present notebooks as a finished project. Put migration history, old-versus-new comparisons and refactor commentary in docs/migration.md or this plan. Comments explain code purpose and reasoning.

Use lowercase underscore filenames and stage/subindex numbering such as 5_1_ubcf.ipynb. Split oversized cells along logical task boundaries. Remove redundant setup within a notebook, obsolete comments and stale references. Intentional repetition across notebooks of the same family is allowed.

Retain existing downloads and Google Drive operations. Organize run outputs by model/variant/configuration/seed to prevent overwriting. Resolve accidental nested output directories. Preserve report-linked assets until links are checked. Keep generated data, caches and checkpoints out of source control.

Remove hard-coded seed-score fallback reporting from executable experiments. Results must identify their producing run/artifact. Missing seed, gated or ablation outputs must be explicitly unavailable. Preserve historical submitted reports independently of new execution.

## 8. Implementation order and acceptance

1. Inventory existing cells, formulas, configurations, dependencies, data paths, artifacts and report links. Compare the current uncommitted refactor with Git HEAD. Give every old section a retained, merged or removed disposition.
2. Consolidate preprocessing and feature ownership. Establish the canonical feature-data naming contract and rewrite downstream consumers to follow it directly. Preserve data identities, splits, model input alignment, Drive integrations and original calculation behavior. Extract only cross-family equivalent utility code.
3. Build the single EDA, preserving every unique analysis. Its raw-data sections run independently; feature-distribution sections load completed feature artifacts. Document the two execution modes without a feature-to-EDA dependency cycle.
4. Inventory and preserve existing evaluator behavior; extract only cross-family equivalent operations before integrating consumers. Add missing basic baselines and organize numbered model notebooks. Preserve advanced variants and allow within-family repetition. Keep NRMS to its baseline and popularity-gated notebooks, plus the optional substantive ablation workflow.
5. Verify every model retains its source evaluator semantics and uses extracted equivalent helpers correctly, isolate run artifacts, correct misleading result fallbacks and update reports/references without breaking links.
6. Remove superseded files after coverage checks. Update README execution order, dependencies, ignore rules and migration notes.
7. Validate notebook structure, Python syntax, imports, configured paths, split membership, candidate identities and key calculations with meaningful fixtures. Run fresh-kernel workflows with real data when available. State static/fixture checks separately from actual training/evaluation.

## 9. Artifact boundaries and execution contracts

### Feature construction versus model-generated scores

The feature notebook owns deterministic data features, fitted data transformations, tokenization/vocabulary preparation, pretrained embedding preparation and MLP feature selection. Its distinct MLP requirements remain at the end. Model-specific sections for UBCF/CBF/NRMS may be added when existing data-feature preparation must be relocated; do not manufacture new features.

Learned neural embedding layers, trainable encoders, user-neighbor fitting and forward-pass representations remain model operations. They are not moved into feature engineering merely because their outputs are called embeddings or features.

Stacking inputs derived from UBCF/CBF/popularity scores are model-generated outputs. Their production belongs to those model notebooks; their keyed join and stacker fitting belong to the stacking notebook. This explicit boundary prevents a circular pipeline in which feature engineering would require models that themselves require its features. Data-derived stacker inputs such as history/context statistics are prepared in the feature notebook. Preserve the existing stacker-training imputation and sampling behavior as training operations and document them.

### Execution order

1. Configure existing download/local/Drive locations and obtain the declared raw inputs.
2. Run preprocessing to export clean tables, identities, partition membership and the training reference window.
3. Run feature engineering to export the selected feature artifacts and metadata.
4. Run EDA fully using raw/processed tables and feature artifacts; raw-only sections may run earlier for inspection. EDA never produces an artifact required by feature engineering or model training. Filenames denote stage labels, not a guarantee that every notebook must run in numerical order.
5. Run basic baseline and advanced model workflows as required. The hybrid scorers export stack_train, val and test phase outputs before stacking runs.
6. Run attention/gated comparisons and optional ablation after their declared comparison artifacts exist.

### Artifact checks

- Reuse existing saved artifacts where their contracts are adequate. Document or extend existing metadata rather than inventing unnecessary files.
- Record input identity, configuration, split/build window, feature ordering and producing run where needed to validate compatibility and caches. Never silently reuse an old result because a filename happens to match.
- Preserve a stable candidate identity, including candidate position where repeated article IDs occur within an impression. Inspect actual duplicates before choosing keys. Do not silently drop candidate rows or use inner joins that lose evaluation candidates.
- Check mapping identity, reserved indices and embedding-row alignment when standardizing names. Distinct valid index spaces remain explicitly distinct; spelling unification alone must not reindex arrays.
- Preserve hybrid stack_train: scorers fit on its earlier build window and score the later training block. Validation/test scores use their original training build window. Do not replace these with in-sample scores or full-training baseline outputs.
- Keep training negative-sampling policies model-specific and unchanged. Overall/user-cohort evaluation uses unsampled full slates. Validate identical candidate coverage for each comparison.
- Identify static frozen and online/lagged information protocols explicitly. Existing lagged model features may remain unchanged; report comparisons separately when information windows differ.
- Name model and evaluation flags according to their actual window and event definition; preserve the source workflow semantics instead of substituting a new fixed-training flag.
- Retain baseline/current run outputs separately. Do not overwrite submitted figures, JSON or reports during validation.

## 10. Verification gates and unresolved decisions

### Confirmed requirements

| ID | Requirement | Acceptance evidence |
|---|---|---|
| R1 | One EDA, preprocessing and feature notebook | Final tree and cell migration coverage |
| R2 | Feature notebook owns prepared-data naming and feature preparation | Consumer names/schemas match exported artifacts directly |
| R3 | Same-family repetition permitted; utilities only cross-family equivalent processing | Each extracted helper lists its family consumers and equivalent behavior |
| R4 | Four baselines; MLP separate | Four executable baseline workflows and separate selected-feature MLP |
| R5 | NRMS baseline plus popularity-gated, optional ablation | Two primary notebooks, optional third only for genuine standalone ablation; replaced runners removed |
| R6 | Functional English presentation; Vietnamese discussion | Notebook/readme review |
| R7 | Download and Drive integrations preserved | Link/configuration migration coverage and feasible I/O checks |
| R8 | Agreed user cohorts; preserved source cold-article and metric behavior | Source comparison, regression fixtures and explicit evaluator provenance |
| R9 | Names/style unified without mathematical changes | Producer/consumer alignment, value checks and configuration comparison |

### Evaluation preservation decision

The user has instructed preservation of existing cold-article evaluation and metric edge cases. No replacement definition is pending approval. Document verified differences between workflows and avoid pooling their results as if their evaluators were equivalent. A missing cohort remains an explicit gap until an existing applicable implementation is identified; this does not authorize inventing a definition.

### Required evidence by phase

- Before deleting a source notebook: every unique section has a verified destination and saved artifacts/report references have been checked.
- After renaming: parse/validate notebooks, check undefined names/imports, validate artifact schemas and compare identities/arrays, not just cell counts.
- After model consolidation: compare architecture, configuration, loss, sampling and predictions on available representative inputs. Check trainable-state compatibility before reusing checkpoints after class/attribute renames.
- After metrics integration: use old-versus-new regression fixtures covering ties, multiple positives, degenerate labels, short slates, empty groups, history boundaries and the actual source cold-item selection. Tests verify preservation, not a new policy.
- Report metric-specific eligible impression counts and undefined/excluded counts alongside results.
- Run fresh-kernel workflows with actual data and dependencies when available. Do not present fixture/static validation as completed real-data training.
- Keep checks proportionate: verify changed behavior and compatibility rather than adding tests for cosmetic changes.

## 11. Completion checklist

- One project EDA, one preprocessing and one feature-engineering notebook.
- Equivalent existing feature calculations preserved; distinct MLP requirements appended at the end.
- Prepared-data naming, schemas and coding conventions originate in the feature notebook and are used directly by all downstream notebooks. No naming-only adapter or alias layer.
- Popularity, UBCF, CBF and NRMS baselines present; MLP kept separate.
- Two primary NRMS notebooks: baseline and popularity-gated; an optional standalone ablation is the only permitted third notebook. No parallel NRMS runner scripts.
- Utilities contain only equivalent processing used by multiple model families.
- No model_utils.py or family-specific helper modules introduced solely for deduplication.
- Model-specific code remains in its notebooks; within-family repetition is permitted.
- Model notebook filenames follow stage/subindex numbering.
- English notebook documentation explains functionality and results without refactor commentary.
- Downloads and local/Google Drive import/export workflows retained.
- AUC, MRR, nDCG@5 and nDCG@10 retain existing formulas/edge cases; agreed user cohorts are applied and source cold-article definitions and selection are preserved.
- Existing cold-article populations and metric edge cases are documented and regression-checked; no new definition is silently substituted.
- Stack-training chronology, candidate coverage, causal information windows and cache compatibility are verified.
- Variable names and writing style follow the feature notebook contract throughout the project; formulas, split membership, sampling and data meaning are preserved.
- No untracked content loss, output collisions, stale numerical fallbacks or broken report references.
- Documentation describes the implemented tree and verification limits accurately.

## Unified pipeline acceptance update

The latest instruction replaces per-family feature producers with one feature flow. Equivalent content/entity/profile features are generated once and consumed directly across models. The user authorizes correcting inconsistent feature semantics using original Group 12 code and the interim report; preserving a conflicting implementation is no longer the default for those features. Model-specific neural layers remain model operations. EDA calculates its tables and distributions rather than consuming static report numbers.

## Standalone smoothed popularity reference

4_5_popularity_smoothed.ipynb uses the current MLP alpha=20/training-prior/daily-window contract and its evaluator. The frozen training click-count reference remains 4_1_popularity.ipynb. Report-style introductions now explain assumptions, inputs, equations, validation, metrics and limits; old narrative test claims were removed from active MLP explanations.
