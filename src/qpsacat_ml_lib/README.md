# qpsacat_ml_lib

**qpsacat_ml_lib** is a modular Python library for end-to-end supervised ML training pipelines.  
It covers every step from raw feature engineering to model registration in MLflow, and runs both locally and on Google Cloud Dataproc Batch.

---

## Table of Contents

1. [Overview](#overview)
2. [Pipeline Stages](#pipeline-stages)
3. [Quick Start (Local)](#quick-start-local)
4. [Running on Dataproc](#running-on-dataproc)
   - [Building a New Docker Image](#building-a-new-docker-image)
   - [Files to Configure](#files-to-configure)
   - [Submitting the Job](#submitting-the-job)
5. [Configuration Reference](#configuration-reference)
6. [Using the Saved Pipeline Transformer for Inference](#using-the-saved-pipeline-transformer-for-inference)
   - [Preprocessing Inference Data](#preprocessing-inference-data)
   - [Creating New Features on Inference Data](#creating-new-features-on-inference-data)
   - [Predicting with the Trained Model](#predicting-with-the-trained-model)
7. [Module Documentation](#module-documentation)
8. [Supported Models](#supported-models)

---

## Overview

```
Input Data (BigQuery / CSV)
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                    qpsacat_ml_lib Pipeline                   │
│                                                             │
│  0. Columns        ─ column registry (target, date, id)    │
│  1. Data Loading   ─ SQL query / local file + train/test   │
│  2. Feature Engin. ─ derived features & transformations    │
│  3. Feature Select.─ variance filter → correlation → Boruta│
│  4. Optimization   ─ Optuna hyperparameter search          │
│  5. Training       ─ model fit on selected features        │
│  6. Evaluation     ─ metrics + external model comparison   │
│  7. Explainability ─ EBM HTML dashboard + SHAP plots       │
│  8. Output         ─ MLflow logging & model registry       │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
MLflow Experiment + Registered Model
```

All pipeline behaviour is controlled by a single **`config.yaml`** file.  
See [`config_example.yaml`](./config_example.yaml) for the full reference with inline documentation.

---

## Pipeline Stages

| # | Stage | Key Config Section | `enable` Flag |
|---|-------|--------------------|---------------|
| 0 | **Columns** | `columns` | always active |
| 1 | **Data Loading** | `data_loading` | always active |
| 2 | **Feature Engineering** | `transform_features` | `transform_features.enable` |
| 3 | **Feature Selection** | `feature_selection` | `feature_selection.enable` |
| 4 | **Optimization** | `optimization` | `optimization.enable` |
| 5 | **Training** | `training` | `training.enable` |
| 6 | **Evaluation** | `evaluation` | `evaluation.enable` |
| 7 | **Explainability** | `feature_explainability` | per sub-section |
| 8 | **Output** | `output.mlflow` | `output.mlflow.enable` |

Setting `training.enable: false` skips stages 5, 6, 7, and 8 entirely — useful when you only want to run feature selection.

### Feature Selection (Stage 3)

When `feature_selection.enable: true`, the automated pipeline runs:

1. **Variance filter** — drops features whose variance falls below `variance_threshold`.
2. **Correlation Elimination** (`steps.correlation_elimination: true`) — iteratively removes features that are highly correlated with another feature (Pearson ≥ `threshold`), keeping the one with the higher correlation with the target.
3. **Boruta** (`steps.boruta: true`) — uses a shadow-feature importance test to confirm or reject each remaining feature.

When `feature_selection.enable: false`, the pipeline uses the explicit `predefined_features` list instead.

### Optimization (Stage 4)

When `optimization.enable: true`, Optuna searches the hyperparameter space defined in `optimization.models.<default_model>` for `n_trials` trials using `cv_splits`-fold cross-validation.  
The best params are used automatically in the Training stage.

When `optimization.enable: false`, the params in `optimization.pre_selected_params` are used directly.

### Evaluation (Stage 6)

Computes the metrics listed in `evaluation.metrics` on the held-out test set.

**Regression metrics**: `mae`, `mape`, `mse`, `medae`, `r2`, `rmse`  
**Classification metrics**: `gini`, `f1`, `pr_auc`, `precision`, `recall`, `accuracy`, `log_loss`

**External Model Evaluation** (`evaluation.external_models_evaluation`): loads previously registered MLflow models and computes the same metrics so you can compare the new run against prior versions side-by-side.

### Explainability (Stage 7)

| Sub-stage | Config key | Output |
|-----------|------------|--------|
| EBM HTML dashboard | `feature_explainability.ebm_html` | Interactive HTML file logged to MLflow |
| SHAP summary/waterfall | `feature_explainability.shap` | PNG plots logged to MLflow |

---

## Quick Start (Local)

### Prerequisites

```bash
pip install qpsacat-ml-lib          # or: pip install -e .
```

### Run the pipeline

```bash
python -m qpsacat_ml_lib.train --config path/to/config.yaml
```

Or from a Python script:

```python
from qpsacat_ml_lib import TrainingPipeline

pipeline = TrainingPipeline.from_config("path/to/config.yaml")
pipeline.run()
```

---

## Running on Dataproc

The pipeline is packaged as a Docker image and submitted as a **Dataproc Batch** job.

### Building a New Docker Image

Rebuild the image whenever you change library code or add Python dependencies.

```bash
# 1. Configure your registry (once)
export DOCKER_REGISTRY="europe-docker.pkg.dev/<PROJECT_ID>/<REPO>"
export IMAGE_TAG="qpsacat-ml:latest"

# 2. Authenticate with Artifact Registry
gcloud auth configure-docker europe-docker.pkg.dev

# 3. Build the image
docker build \
  --platform linux/amd64 \
  -t "${DOCKER_REGISTRY}/${IMAGE_TAG}" \
  -f Dockerfile \
  .

# 4. Push to Artifact Registry
docker push "${DOCKER_REGISTRY}/${IMAGE_TAG}"
```

After pushing, update `platform_properties.yaml` → `dataproc.container_image` with the new URI.

> **Tip**: Tag production images with a version (e.g. `qpsacat-ml:v1.2.0`) rather than `latest` so runs are reproducible.

### Files to Configure

Before submitting a Dataproc Batch job you must configure **two** files:

#### `platform_properties.yaml`

Controls GCP project settings, Dataproc resources, and the Docker image:

| Key | Description |
|-----|-------------|
| `gcp.project_id` | GCP project ID |
| `gcp.region` | Dataproc region (e.g. `europe-west1`) |
| `gcp.staging_bucket` | GCS bucket for Dataproc staging artifacts |
| `dataproc.container_image` | Full Docker image URI to use for the batch job |
| `dataproc.executor.instances` | Number of Spark executor nodes |
| `dataproc.executor.machine_type` | Machine type for executors |
| `mlflow.tracking_uri` | MLflow server URL |

See [`platform_properties.yaml`](./platform_properties.yaml) for the full template.

#### `.env`

Stores secrets and credentials that must not be committed to source control.  
Copy `.env.example` to `.env` and fill in the values:

```bash
cp .env.example .env
# edit .env with real values
```

| Variable | Description |
|----------|-------------|
| `GCP_PROJECT_ID` | GCP project |
| `GCP_REGION` | Dataproc region |
| `MLFLOW_TRACKING_URI` | MLflow server URL |
| `MLFLOW_TRACKING_USERNAME` | MLflow basic-auth username |
| `MLFLOW_TRACKING_PASSWORD` | MLflow basic-auth password |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to GCP service account JSON (local only) |

### Submitting the Job

```bash
# From the experiment directory
python submit_dataproc_job.py \
  --config config.yaml \
  --platform platform_properties.yaml \
  --env .env
```

---

## Configuration Reference

The full annotated example is in [`config_example.yaml`](./config_example.yaml).

### Minimal config skeleton

```yaml
task: regression
default_model: ebm_regressor
start_training_date: "20250101"
end_training_date: "20260101"

columns:
  target: my_target
  date: reference_date
  id: row_id
  drop_cols: []
  categorical: []

data_loading:
  query: "SELECT * FROM `project.dataset.table`"
  test_size: 0.2
  random_state: 42
  log_target: false

transform_features:
  enable: false

feature_selection:
  enable: false

optimization:
  enable: false
  pre_selected_params: {}

training:
  enable: true

evaluation:
  enable: true
  metrics: [mae, rmse, r2]

feature_explainability:
  ebm_html:
    enable: false
  shap:
    enable: false

output:
  mlflow:
    enable: true
    experiment_name: "my_experiment"
    run_name: "my_run_v1"
    artifact_path: "model"
    register_model: false
```

---

## Using the Saved Pipeline Transformer for Inference

After a training run the **full pipeline object** (including all fitted transformers) is logged to MLflow under the `artifact_path` key.  
Load it to apply the *exact same* preprocessing, feature engineering, and feature selection to new data before prediction.

### Loading the Pipeline from MLflow

```python
import mlflow.pyfunc

# Load by run ID
pipeline = mlflow.pyfunc.load_model("runs:/<RUN_ID>/pipeline/")

# Or by registered model name + version
pipeline = mlflow.pyfunc.load_model("models:/qp_sac_tmc_mdl/Production")
```

### Preprocessing Inference Data

The pipeline transformer reproduces every preprocessing step (type casting, encoding, column dropping, log-transform) that was applied during training:

```python
import pandas as pd

# Load raw inference data (same schema as training data)
df_inference = pd.read_parquet("inference_data.parquet")

# Apply all preprocessing transformations
df_preprocessed = pipeline.named_steps["preprocessor"].transform(df_inference)
```

### Creating New Features on Inference Data

All feature engineering steps defined in `transform_features` are stored inside the pipeline object and are replayed automatically:

```python
# Applies every transform_features step (ratios, date diffs, encodings, etc.)
df_features = pipeline.named_steps["feature_engineering"].transform(df_preprocessed)
```

### Applying Feature Selection on Inference Data

Only the features selected during training are passed to the model.  
The feature selector stored in the pipeline keeps track of which columns survived:

```python
# Applies variance filter + correlation elimination + Boruta masks
df_selected = pipeline.named_steps["feature_selector"].transform(df_features)
```

### Predicting with the Trained Model

```python
# End-to-end: preprocess → engineer → select → predict
predictions = pipeline.predict(df_inference)

# If log_target was true during training, predictions are already in the
# original scale (expm1 is applied automatically by the pipeline).
print(predictions)
```

> **Note**: Pass the raw inference DataFrame (same columns as the original training data) directly to `pipeline.predict()`.  
> The pipeline handles all intermediate transformations internally.

### Accessing Individual Pipeline Stages

```python
# List all named steps
print(pipeline.named_steps.keys())

# Access the fitted model directly
model = pipeline.named_steps["model"]
selected_features = pipeline.named_steps["feature_selector"].get_feature_names_out()
```

---

## Module Documentation

| Module | Description |
|--------|-------------|
| [`feature_engineering/`](./feature_engineering/README.md) | Transforms and derived feature definitions |
| [`feature_selection/`](./feature_selection/README.md) | Variance filter, correlation elimination, and Boruta |
| [`registry/`](./registry/README.md) | MLflow experiment tracking and model registration helpers |
| [`preprocessing/`](./preprocessing/) | Data loading, type casting, and train/test split |

---

## Supported Models

| Key | Algorithm | Task |
|-----|-----------|------|
| `ebm_regressor` | Explainable Boosting Machine (EBM) | Regression |
| `ebm_classifier` | Explainable Boosting Machine (EBM) | Classification |
| `lgbm_regressor` | LightGBM | Regression |
| `lgbm_classifier` | LightGBM | Classification |

EBM models also enable the HTML explainability dashboard (`feature_explainability.ebm_html`).
