# Registry Module

This module implements the **MLflow logging and model registration** functionality (stage 8) of the `qpsacat_ml_lib` training pipeline.

It is controlled by the `output.mlflow` section of `config.yaml`.

---

## Role in the Pipeline

```
Trainer → Evaluator → Explainability → [MLflowLogger] → MLflow Model Registry
```

The `MLflowLogger`:
1. Creates or reuses the named MLflow experiment.
2. Starts a new MLflow run.
3. Logs all configuration parameters, evaluation metrics, and artefacts (plots, HTML dashboard, pipeline object).
4. Optionally registers the pipeline object in the **MLflow Model Registry**.

---

## Configuration

```yaml
output:
  mlflow:
    enable: true                  # false = skip all MLflow logging
    experiment_version: "v1.0.0"  # informational tag logged as a parameter
    experiment_name: "my_experiment"       # MLflow experiment name
    run_name: "my_model_20250101_20260101" # MLflow run name
    experiment_description: "Short description of this run."
    artifact_path: "pipeline"     # sub-path under the run where the model is saved
    register_model: true          # true = register in Model Registry after logging
  log_file: "logs/training.log"   # local log file path ("" to disable)
```

---

## What Gets Logged

### Parameters

All `config.yaml` values are flattened and logged as MLflow parameters, including:

- `task`, `default_model`, `start_training_date`, `end_training_date`
- All `data_loading`, `feature_selection`, `optimization`, and `evaluation` sub-keys
- `experiment_version`

### Metrics

Every metric in `evaluation.metrics` is logged for the trained model and, if enabled, for each external model under a `<metric>_<model_name>` key.

### Artefacts

| Artefact | Path in MLflow | Description |
|----------|----------------|-------------|
| Full pipeline | `<artifact_path>/` | scikit-learn `Pipeline` (preprocessor → FE → FS → model) |
| Evaluation metrics CSV | `metrics/` | All test-set metrics in tabular form |
| EBM HTML dashboard | `explainability/ebm/` | Interactive global explainability HTML |
| EBM per-feature HTML | `explainability/ebm/features/` | One HTML page per top feature |
| SHAP summary plot | `explainability/shap/` | Beeswarm plot PNG |
| SHAP waterfall plot | `explainability/shap/` | Waterfall plot PNG |
| Feature importance CSV | `feature_importance/` | Ranked feature importances |
| Selected features list | `feature_selection/` | Features that survived the selection stage |

---

## Loading a Registered Model

### By run ID

```python
import mlflow.pyfunc

pipeline = mlflow.pyfunc.load_model("runs:/<RUN_ID>/pipeline/")
predictions = pipeline.predict(df_inference)
```

### By registered model name and stage

```python
pipeline = mlflow.pyfunc.load_model("models:/my_model_name/Production")
predictions = pipeline.predict(df_inference)
```

### By registered model name and version

```python
pipeline = mlflow.pyfunc.load_model("models:/my_model_name/3")
predictions = pipeline.predict(df_inference)
```

---

## Model Registry Lifecycle

When `register_model: true` the trained pipeline is registered under the name `<experiment_name>` (or a custom name if configured).  
Use the MLflow UI or the MLflow Python client to promote versions through `Staging → Production → Archived`.

```python
from mlflow import MlflowClient

client = MlflowClient()
client.transition_model_version_stage(
    name="my_model_name",
    version=3,
    stage="Production",
)
```

---

## Comparing Runs

All runs within the same experiment share the same parameter and metric keys.  
Use the MLflow UI **Compare** feature to view metric trends across runs, or query programmatically:

```python
import mlflow

runs = mlflow.search_runs(
    experiment_names=["my_experiment"],
    order_by=["metrics.rmse ASC"],
)
print(runs[["run_id", "metrics.rmse", "params.default_model"]])
```
