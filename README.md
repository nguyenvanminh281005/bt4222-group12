# BT4222 Mining Web Data for Business Insights - Group 12

This repository contains the source code, notebooks, and deliverables for the BT4222 group project.

---

## 📌 Project Pipeline & Workflow Stages

Our project follows a standard end-to-end data science lifecycle divided into 5 clear stages:

| Stage # | Stage Name | Description | Target Directory | Example Files |
|---|---|---|---|---|
| **1** | **Exploratory Data Analysis (EDA)** | Understand data distribution, correlation, outliers, and initial business insights. | `data_preparation/` | `1_eda.ipynb`, `1_eda_text.ipynb` |
| **2** | **Data Preprocessing** | Missing value imputation, outlier handling, text cleaning, normalization, deduplication. | `data_preparation/` | `2_preprocessing.ipynb` |
| **3** | **Feature Engineering** | Extraction of domain features, TF-IDF / embeddings, encoding, feature selection, scaling. | `data_preparation/` | `3_feature_engineering.ipynb` |
| **4** | **Base Models** | Establish simple benchmarks (e.g., Logistic Regression, Naive Bayes, Decision Tree). | `models/` | `4_basemodels.ipynb` |
| **5** | **Advanced / Other Models** | Tree ensembles (Random Forest, LightGBM, XGBoost), Deep Learning/NLP models, ensembling & tuning. | `models/` | `5_xgboost.ipynb`, `5_bert.ipynb`, `5_ensemble.ipynb` |

---

## 📁 Repository Structure & File Naming Conventions

### 1. Directory Structure

```text
bt4222-group12/
├── README.md
├── data_preparation/
│   ├── 1_eda.ipynb
│   ├── 2_preprocessing.ipynb
│   └── 3_feature_engineering.ipynb
├── models/
│   ├── 4_basemodels.ipynb
│   ├── 5_xgboost.ipynb
│   └── 5_neural_net.ipynb
├── references/              # Project guidelines, rubrics, and references
│   └── ...
└── data/                    # (Local only - DO NOT commit large datasets)
    ├── raw/
    └── processed/
```

### 2. File Naming Rules

To keep everyone's work organized and avoid merge collisions:
- **Prefix files with the pipeline stage number**: `<stage_number>_<description>.ipynb`
  - Examples: `1_eda.ipynb`, `2_preprocessing_text.ipynb`, `3_features_embedding.ipynb`, `4_basemodels.ipynb`, `5_lightgbm.ipynb`
- **Use lowercase and underscores (`_`)**: Avoid spaces, special characters, or names like `test.ipynb`, `final.ipynb`, `final_v2.ipynb`.
- If working on alternative experiments, append your model or feature identifier (e.g., `5_bert_classifier.ipynb`).

---

## 🌿 Git Branching & Contribution Guidelines

> **Rule #1:** Never push untested or direct code changes straight to the `main` branch. Always work on a separate branch and open a Pull Request (PR).

### 1. Branch Naming Convention

Format your branch name as:
```text
<stage>/<short-description>
```
or (if preferred for tracking ownership):
```text
<your-name>/<stage>-<short-description>
```

**Allowed stage prefixes:**
- `eda/` - for exploratory analysis
- `prep/` - for data cleaning and preprocessing
- `feat/` - for feature engineering
- `model/` - for model development (baseline or advanced)
- `docs/` - for documentation and report preparation
- `fix/` - for bug fixes in shared utility code

**Examples:**
- `eda/target-distribution`
- `prep/handle-missing-data`
- `feat/bert-embeddings`
- `model/xgboost-hyperopt`
- `minh/model-stacking`

---

### 2. Step-by-Step Contribution Workflow

#### Step 1: Update your local `main`
Always pull the latest changes before starting new work:
```bash
git checkout main
git pull origin main
```

#### Step 2: Create and switch to your feature branch
```bash
git checkout -b <stage>/<feature-description>
# Example:
git checkout -b feat/tfidf-features
```

#### Step 3: Implement your code & test
- Ensure notebooks run top-to-bottom without errors (`Kernel` -> `Restart and Run All`).
- Set a fixed random seed (e.g., `seed = 42`) for reproducibility.
- Clear excessive outputs (e.g. huge printouts or gigabytes of embedded interactive plots) before saving if they inflate notebook file size.

#### Step 4: Commit your changes with meaningful messages
```bash
git status
git add <specific-files>
git commit -m "<stage>: brief summary of what was done"
# Examples:
# git commit -m "prep: clean text descriptions and remove stopwords"
# git commit -m "model: add LightGBM baseline with 5-fold cross validation"
```

#### Step 5: Push your branch to GitHub
```bash
git push -u origin <stage>/<feature-description>
```

#### Step 6: Create a Pull Request (PR)
1. Go to GitHub repository page.
2. Open a Pull Request targeting `main`.
3. In the PR description:
   - Summarize key changes and decisions made.
   - Attach performance metrics (e.g., Accuracy, F1-Score, AUC) if applicable.
   - Mention team members for review before merging.

---

## ⚠️ Important Team Rules

1. **Large Data Files:**
   - **DO NOT commit raw or large data files** (`.csv`, `.xlsx`, `.parquet`, `.zip`, `.pt`, `.pkl` > 10MB) to Git.
   - Keep datasets in a local `data/` directory (ignored by git) or store shared datasets on team Google Drive / cloud storage.
2. **Reproducibility:**
   - Always state package dependencies or versions if installing external libraries.
   - Document input paths and output files at the top of each notebook.
3. **Model Evaluation Consistency:**
   - Agree on shared validation splits (e.g., train/validation/test split ratios or stratified K-fold) and primary evaluation metrics so all models can be benchmarked fairly.