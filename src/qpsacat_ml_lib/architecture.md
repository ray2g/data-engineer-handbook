# Architecture — qpsacat_ml_lib Training Pipeline

This document describes the internal architecture of the `qpsacat_ml_lib` training pipeline: how the stages are connected, what each component produces, and how the artefacts flow from raw data to a registered MLflow model.

---

## High-Level Data Flow

```
┌──────────────────────────────────────────────────────────────────┐
│  config.yaml                                                     │
│  platform_properties.yaml + .env  (Dataproc / local)            │
└──────────────────────┬───────────────────────────────────────────┘
                       │  parsed at startup
                       ▼
              ┌─────────────────┐
              │  PipelineConfig  │  (validated Pydantic model)
              └────────┬────────┘
                       │
          ┌────────────▼────────────┐
          │      DataLoader          │  BigQuery SQL / local file
          │  ─────────────────────  │  → DataFrame (train + test split)
          └────────────┬────────────┘
                       │  raw DataFrame
          ┌────────────▼────────────┐
          │   FeatureEngineer        │  transform_features steps
          │  ─────────────────────  │  → enriched DataFrame
          └────────────┬────────────┘
                       │  engineered DataFrame
          ┌────────────▼────────────┐
          │   FeatureSelector        │  variance → correlation → Boruta
          │  ─────────────────────  │  → reduced feature set
          └────────────┬────────────┘
                       │  selected feature names + transformed DataFrame
          ┌────────────▼────────────┐
          │    Optimiser             │  Optuna TPE search
          │  ─────────────────────  │  → best hyperparameters
          └────────────┬────────────┘
                       │  best params
          ┌────────────▼────────────┐
          │      Trainer             │  fit model (EBM / LGBM)
          │  ─────────────────────  │  → fitted model
          └────────────┬────────────┘
                       │  fitted model
     ┌─────────────────┼─────────────────┐
     │                 │                 │
┌────▼────┐     ┌──────▼──────┐   ┌─────▼──────┐
│Evaluator│     │Explainability│   │  MLflowLogger│
│ metrics │     │ EBM HTML /  │   │  artifacts  │
│ + ext.  │     │ SHAP plots  │   │  + registry │
│ models  │     └─────────────┘   └────────────┘
└─────────┘
```

---

## Component Descriptions

### PipelineConfig

Reads and validates `config.yaml` using Pydantic.  
Makes the entire configuration available as a typed object to all downstream stages.  
Raises clear validation errors at startup if required fields are missing or have wrong types.

### DataLoader

- Executes the BigQuery / Spark SQL query with `{start_training_date}` and `{end_training_date}` placeholders substituted from the config.
- Applies `columns.drop_cols` immediately after loading.
- Performs a **chronological train/test split** on `columns.date` using `data_loading.test_size`.
- Optionally applies `log1p` to the target column when `data_loading.log_target: true`.

### FeatureEngineer

Applies the transformation `steps` defined in `transform_features` **in order**.  
Each step type is implemented as a separate transformer class:

| Step type | Description |
|-----------|-------------|
| `ratio` | `numerator / denominator` with configurable `fill_value` for zero denominators |
| `date_diff` | Integer difference between two date columns in a chosen `unit` (`days`, `months`) |
| `clip` | Caps values between `lower` and `upper` bounds |
| `log1p` | Applies `log(1 + x)` to a numeric column |
| `one_hot` | One-hot encodes a categorical column; optionally drops the first level |

The `FeatureEngineer` is fitted on the **training split only** and stores its fit state so it can be replayed identically on inference data.

### FeatureSelector

Three sequential sub-stages (each individually toggled in `feature_selection.steps`):

1. **Variance filter**: drops columns whose variance on the training set is below `variance_threshold`.
2. **Correlation Elimination**: iteratively removes columns that are correlated above `threshold` with another column; when two columns are correlated the one with the lower absolute Pearson correlation with the target is dropped.
3. **Boruta**: wraps a Random Forest in the Boruta algorithm; features that cannot be distinguished from random permutations (shadow features) are rejected at the `alpha` significance level.

The `FeatureSelector` records which features were kept so the same mask can be applied at inference time.

### Optimiser

- Uses **Optuna** (TPE sampler) to search the hyperparameter space defined in `optimization.models.<default_model>`.
- Evaluates each trial with `cv_splits`-fold cross-validation on the training split.
- Minimises or maximises `optuna_params.metric` (default: `rmse`).
- When `optimization.enable: false` the pre-selected params in `pre_selected_params` are used directly and this stage is skipped.

### Trainer

Instantiates the model class matching `default_model` with the best hyperparameters from the Optimiser (or `pre_selected_params`) and fits it on the full training split.

### Evaluator

Computes each metric in `evaluation.metrics` on the held-out test split.

**External model evaluation**: if `external_models_evaluation.enable: true`, each model listed under `models` is loaded from MLflow, its predictions are computed on the same test split, and the same metrics are computed — enabling side-by-side comparison in MLflow.

### Explainability

| Sub-component | Output |
|---------------|--------|
| `EBMExplainer` | Per-feature HTML pages + a global summary dashboard logged as MLflow artifacts |
| `SHAPExplainer` | Summary beeswarm plot + waterfall plot for individual predictions logged as MLflow artifacts |

Both components operate on the test split only.  
SHAP uses a background sample of `bg_sample_size` rows and explains up to `test_sample_size` test rows.

### MLflowLogger

- Creates / reuses the MLflow experiment identified by `output.mlflow.experiment_name`.
- Logs all config parameters, metrics, and artifacts under a single run.
- Saves the full scikit-learn `Pipeline` object (preprocessor → feature engineer → feature selector → model) as an MLflow model artefact at `artifact_path`.
- Optionally registers the model in the MLflow Model Registry (`register_model: true`).

---

## Persisted Pipeline Object

The artefact saved to MLflow is a **scikit-learn `Pipeline`** with the following named steps:

```
Pipeline(steps=[
    ("preprocessor",        Preprocessor),        # type casting, encoding, drop_cols
    ("feature_engineering", FeatureEngineer),      # transform_features steps
    ("feature_selector",    FeatureSelector),      # fitted selection masks
    ("model",               <fitted model>),       # EBM or LGBM
])
```

Loading this pipeline and calling `.predict()` on raw inference data (same schema as training data) automatically applies every transformation in order.  
See the [Inference section in README.md](./README.md#using-the-saved-pipeline-transformer-for-inference) for usage examples.

---

## Execution Modes

| Mode | How to run | Notes |
|------|-----------|-------|
| **Local** | `python -m qpsacat_ml_lib.train --config config.yaml` | Uses local credentials; `data_loading.local_path` for offline data |
| **Dataproc Batch** | `python submit_dataproc_job.py --config ... --platform ...` | Packaged in Docker; credentials from `.env`; resources from `platform_properties.yaml` |

---

## Configuration Dependencies

```
config.yaml
├── columns            ← used by: all stages
├── data_loading       ← used by: DataLoader
├── transform_features ← used by: FeatureEngineer
├── feature_selection  ← used by: FeatureSelector
├── optimization       ← used by: Optimiser, Trainer
├── training           ← used by: Trainer, Evaluator, Explainability, MLflowLogger
├── evaluation         ← used by: Evaluator
├── feature_explainability ← used by: EBMExplainer, SHAPExplainer
└── output.mlflow      ← used by: MLflowLogger

platform_properties.yaml
├── gcp                ← used by: DataLoader (BQ), MLflowLogger
├── dataproc           ← used by: job submission script
├── mlflow             ← used by: MLflowLogger (tracking URI)
└── logging            ← used by: all stages

.env
├── GCP_PROJECT_ID, GCP_REGION     ← used by: DataLoader, job submission
├── MLFLOW_TRACKING_*              ← used by: MLflowLogger
└── GOOGLE_APPLICATION_CREDENTIALS ← used by: GCP clients (local only)
```
