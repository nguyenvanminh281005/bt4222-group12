# Validation

## Executed checks

- Python parsing and undefined-name inspection across all active notebook code.
- Temporary synthetic MIND-like dataset: 30 articles, 420 impressions over seven days, cold/sparse/warm histories, multiple clicked candidates and linked entity vectors.
- Full fixture execution of preprocessing, feature engineering, consolidated EDA, three newly added basic baselines, all four hybrid workflows, initial/improved MLP, MLP attention, baseline NRMS and popularity-gated NRMS.
- Neural training smoke tests use one epoch. Optional ablation/permutation/multiple-seed retraining is disabled in the fixture runner. Committed notebook training defaults are preserved.
- The fixture supplies deterministic cached MLP text vectors and a small GloVe file; CBF reads those same cached canonical content vectors. These checks do not validate downloaded pretrained MiniLM models.
- Original versus refactored MLP impression metrics: 100 multi-positive/tied-score slates, matching within numerical tolerance.
- Original versus refactored baseline NRMS impression metrics: 100 multi-positive/tied-score slates, matching within numerical tolerance.
- Shared hybrid evaluator equals the original evaluator, including its cold-article cohort selection and reproducible score ties, on a controlled fixture.
- Prepared artifacts reject changed raw-source metadata. The optional global top-20 fallback also has an inference shape/coverage fixture.
- AST comparisons of AttentionPooling, NewsEncoder, UserEncoder, NRMS, GatedNRMS and TrainDataset preserve source architecture/dataset logic after agreed identifier normalization.

## Reproduction

Install requirements, then run:

```sh
python tests/verify_notebooks.py
python tests/verify_pipeline.py
```

The pipeline runner writes its data and experiment artifacts to a temporary directory. Its synthetic scores are not benchmarks. It is designed for CPU validation; actual training requires the real inputs and appropriate resources.

## Not established by these checks

Real MIND-small training, GPU behavior, real pretrained MiniLM downloads, mounted Drive operations, optional LightGBM ranking, full multi-seed retraining, self-attention ablation retraining and the optional paired-checkpoint comparison have not been executed on the real dataset. Existing report numbers are historical, not newly verified measurements.

Fixture warnings may include SVD variance on a small repeated corpus and plot layout limitations. A constant-feature fixture exposed an existing MLP cold-input assertion's sensitivity to near-zero scaling variance; the original assertion and formula were retained, and the representative fixture uses varied entity vectors.

## Fixture environment

numpy 2.5.4, pandas 3.1.1, scipy 1.18.1, scikit-learn 1.9.1, matplotlib 3.11.2, seaborn 0.13.2, torch 2.15.0+cpu, nbformat 5.11.1. Python 3.12; CPU training.

## Unified feature revision

The latest user instruction authorizes corrections to conflicting feature implementations. A single sentence-level content representation and normalized mean TransE representation are now produced for consumers. CBF reads those same arrays rather than generating a second embedding. Missing entity vectors and empty history profiles are zero; category match is binary, with zero for an empty history. Category/subcategory embedding widths are 32/32. Existing MLP lagged smoothed CTR is retained as the canonical historical popularity feature. Neural title sequences are separate token inputs, not competing semantic article embeddings.

EDA computes descriptive features from preprocessed tables, uses the earliest impression snapshot consistently for user-level history statistics, retains zero inactivity gaps for the full numeric summary, and computes numeric/categorical/text/entity tables, category-match/history pies and split comparisons. Descriptive full-period CTR is kept separate from lagged model inputs. These revisions supersede the earlier preservation claims for the changed features; historical report numbers are not claimed to have been reproduced without real data.

The unified revision passes the complete synthetic pipeline with one-epoch model smoke tests. Additional assertions cover the absence of a separate CBF embedding producer, zero empty-history vectors, binary category matching, and computed report table/figure artifacts. Real-data statistics, original report percentages and full training runs remain unverified.

## Real-data CPU execution

Preprocessing, EDA and the frozen click-count popularity baseline completed on the supplied MIND-small TSV files. Split counts: train 126,695; validation 30,270; test 73,152. Execution times were approximately 3.5, 53.0 and 28.9 seconds. Executed review notebooks and cell logs are under outputs/cpu_validation; genuine outputs are saved into the executed source notebooks.

MLP category/subcategory dimensions are restored to 32/32 in initial, improved and attention variants. The saved configuration records both dimensions. Formatting changes preserve executable syntax and do not replace historical reported measurements.

The supplied embedding.npy and embedding_all.npy have shapes (34,304, 300) and (60,993, 300). Their source/vocabulary compatibility is unverified and they were not used. MiniLM generation and real-data MLP/NRMS training have not been run. Observed recency statistics differ from the historical report sample; no historical values were silently replaced.

## Notebook presentation and saved outputs

Blank lines were removed throughout executable cells before applying consistent block spacing across all 16 active notebooks. Formatting equivalence was checked using Black's AST equivalence validation. The CPU runner now captures stdout, displayed tables, last-expression values and PNG figures, and saves them directly with the source cells. Preprocessing, EDA and popularity are rerun for genuine saved outputs; other model notebooks do not receive synthetic or historical output.

## Smoothed historical CTR baseline

The new 4_5 baseline ran on the full real validation/test data with its MLP-compatible evaluator in approximately 127 seconds. Test overall nDCG@5 is 0.28338885 and cold-user nDCG@5 is 0.28281427. Full results and scores are under outputs/base_models/popularity_smoothed; genuine notebook outputs are retained. Raw-score versus standardized-feature floating-point ties may differ, as documented. Formatting and notebook schema checks cover all 16 active notebooks.

## Portable feature bundles and Colab setup

Pipeline notebooks include an optional Colab setup cell. Source signatures use content hashes; the final feature manifest is published after all exports. MLP/NRMS validation works after relocation without raw files. Missing/changed exported contents and output roots bound to another feature identity are rejected. MLP reusable-run specifications include the feature identity; attention settings also record both categorical widths. Existing bundles without a manifest must be regenerated. These changes do not change model formulas or split membership.
