# Train Experiment Template

This directory is the canonical starting point for a new training experiment using `qpsacat_ml_lib`.

Copy the entire directory to a new location (e.g. `experiments/my_experiment_v2/`), then edit the three configuration files described below.

---

## Directory Contents

```
train_experiment_template/
├── config.yaml               # Pipeline behaviour (features, model, metrics, …)
├── platform_properties.yaml  # GCP / Dataproc / Docker settings
├── .env.example              # Environment variable template (copy to .env)
└── README.md                 # This file
```

---

## Quick Start

### 1. Copy the template

```bash
cp -r experiments/train_experiment_template experiments/my_experiment_v2
cd experiments/my_experiment_v2
```

### 2. Configure `config.yaml`

At a minimum, update:

| Field | Description |
|-------|-------------|
| `start_training_date` | First date of the training window (`YYYYMMDD`) |
| `end_training_date` | Last date of the training window (`YYYYMMDD`) |
| `columns.target` | Name of the column to predict |
| `columns.date` | Date column used for the chronological train/test split |
| `columns.id` | Unique row identifier (excluded from features) |
| `data_loading.query` | BigQuery / Spark SQL query that returns the training data |
| `output.mlflow.experiment_name` | MLflow experiment name |
| `output.mlflow.run_name` | Descriptive run name (include date range and model) |

Toggle each pipeline stage with its `enable` flag:

```yaml
transform_features.enable:              true  # run feature engineering
feature_selection.enable:              true  # run automated feature selection
feature_selection.steps.correlation_elimination: true
feature_selection.steps.boruta:        true
optimization.enable:                   true  # run Optuna hyperparameter search
training.enable:                       true  # fit the model
evaluation.enable:                     true  # compute test-set metrics
evaluation.external_models_evaluation.enable: true  # compare to prior models
feature_explainability.ebm_html.enable: true  # generate EBM HTML dashboard
feature_explainability.shap.enable:    true  # generate SHAP plots
output.mlflow.enable:                  true  # log everything to MLflow
```

See [`config.yaml`](./config.yaml) for the full annotated configuration.

### 3. Configure `platform_properties.yaml`

Edit this file to point to your GCP project, Dataproc region, staging bucket, and Docker image:

```yaml
gcp:
  project_id: "your-gcp-project-id"
  region: "europe-west1"
  staging_bucket: "gs://your-bucket/dataproc-staging"

dataproc:
  container_image: "europe-docker.pkg.dev/<PROJECT>/<REPO>/qpsacat-ml:latest"
  executor:
    instances: 4
    machine_type: "n1-standard-8"

mlflow:
  tracking_uri: "https://your-mlflow-server/mlflow"
```

### 4. Create and fill `.env`

```bash
cp .env.example .env
# Open .env and fill in secrets (MLflow credentials, GCP project, etc.)
```

> **Never commit `.env` to source control.**  It is listed in `.gitignore`.

### 5. Run locally

```bash
python -m qpsacat_ml_lib.train --config config.yaml
```

### 6. Run on Dataproc

First, build and push a Docker image if library code has changed (see [Building a New Docker Image](#building-a-new-docker-image)), then submit:

```bash
python submit_dataproc_job.py \
  --config config.yaml \
  --platform platform_properties.yaml \
  --env .env
```

---

## Building a New Docker Image

Rebuild whenever you modify the `qpsacat_ml_lib` source code or update Python dependencies.

```bash
# Set variables
export DOCKER_REGISTRY="europe-docker.pkg.dev/<PROJECT_ID>/<REPO>"
export IMAGE_TAG="qpsacat-ml:latest"   # or use a specific version tag

# Authenticate
gcloud auth configure-docker europe-docker.pkg.dev

# Build (must target linux/amd64 for Dataproc)
docker build \
  --platform linux/amd64 \
  -t "${DOCKER_REGISTRY}/${IMAGE_TAG}" \
  -f Dockerfile \
  .

# Push to Artifact Registry
docker push "${DOCKER_REGISTRY}/${IMAGE_TAG}"
```

After pushing, update `platform_properties.yaml` → `dataproc.container_image` with the new image URI.

> **Best practice**: tag production images with a version number (e.g. `qpsacat-ml:v1.2.0`) rather than `latest` so historical runs remain reproducible.

---

## Comparing to Previous Models

Add entries under `evaluation.external_models_evaluation.models` to load and score prior MLflow models on the same test split:

```yaml
evaluation:
  external_models_evaluation:
    enable: true
    models:
      - name: ebm_M1
        path: "runs:/39391de82ae045df945eb73ad6def1ca/pipeline/"
        apply_expm1: null   # null = inherit log_target from pipeline
      - name: ebm_M2
        path: "runs:/your_m2_run_id/pipeline/"
        apply_expm1: null
```

Metrics are logged as `<metric>_ebm_M1`, `<metric>_ebm_M2`, etc., making them directly comparable in the MLflow UI.

---

## Using the Saved Pipeline for Inference

Once the run completes, the full pipeline object is available in MLflow:

```python
import mlflow.pyfunc
import pandas as pd

# Load by run ID
pipeline = mlflow.pyfunc.load_model("runs:/<RUN_ID>/pipeline/")

# Load raw inference data (same schema as training data)
df_inference = pd.read_parquet("inference_data.parquet")

# Run the full pipeline: preprocessing → feature engineering → feature selection → predict
predictions = pipeline.predict(df_inference)
```

Access individual steps for debugging:

```python
# Apply only preprocessing + feature engineering
fe = pipeline.named_steps["feature_engineering"]
df_engineered = fe.transform(df_raw)

# Check which features survived selection
fs = pipeline.named_steps["feature_selector"]
selected_features = fs.get_feature_names_out()
```

See the [main library README](../../../../src/qpsacat_ml_lib/README.md#using-the-saved-pipeline-transformer-for-inference) for a full inference walkthrough.
