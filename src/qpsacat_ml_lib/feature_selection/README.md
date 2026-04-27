# Feature Selection Module

This module implements the **Feature Selection** stage (stage 3) of the `qpsacat_ml_lib` training pipeline.

It is controlled by the `feature_selection` section of `config.yaml`.

---

## Role in the Pipeline

```
FeatureEngineer → [FeatureSelector] → Optimiser → Trainer
```

The `FeatureSelector` is **fitted on the training split only** and stores the list of surviving features.  
At inference time the same feature mask is applied automatically when the saved pipeline is loaded.

---

## Configuration

```yaml
feature_selection:
  enable: true            # false = use predefined_features list instead

  # Explicit feature list used when enable: false.
  # Leave empty to pass all engineered features through.
  predefined_features: []

  # Variance filter threshold applied before the selection steps.
  variance_threshold: 0.0

  steps:
    correlation_elimination: true
    boruta: true

  # ── 3a. Correlation Elimination ───────────────────────────
  correlation_elimination:
    threshold: 0.95        # max Pearson correlation allowed between any two features

  # ── 3b. Boruta ────────────────────────────────────────────
  boruta:
    max_iter: 100          # total Boruta iterations
    alpha: 0.05            # significance level for shadow-feature test
    model: null            # null = inherit default_model from top of config
    n_estimators: 100      # Random Forest trees used inside Boruta
    random_state: 42
```

---

## Stage Details

### Variance Filter

Applied unconditionally before the two optional steps.  
Features with training-set variance below `variance_threshold` are dropped.  
Default is `0.0` (drops only constant features).

### Correlation Elimination

1. Computes the Pearson correlation matrix of all remaining features on the training set.
2. Identifies pairs of features with |correlation| ≥ `threshold`.
3. For each such pair, drops the feature with the **lower** absolute correlation with the target.
4. Repeats until no correlated pairs remain.

This step is deterministic given the same training data and threshold.

### Boruta

Boruta wraps a Random Forest and adds *shadow features* (randomly permuted copies of every real feature).  
Features that cannot beat the best shadow feature at significance level `alpha` are rejected.

- `max_iter`: higher values give more stable results but take longer.
- `alpha`: lower values are more conservative (keep fewer features).
- `model: null` uses the model specified by `default_model` at the top of `config.yaml`; set to a specific model key to override.

---

## Manual Feature List

When `enable: false`, the pipeline skips the automated selection and uses the explicit `predefined_features` list:

```yaml
feature_selection:
  enable: false
  predefined_features:
    - feature_a
    - feature_b
    - invoice_ratio
```

If `predefined_features` is empty, all features from the previous stage are used.

---

## Using the FeatureSelector at Inference Time

The fitted feature selector is stored inside the saved MLflow pipeline:

```python
import mlflow.pyfunc

pipeline = mlflow.pyfunc.load_model("runs:/<RUN_ID>/pipeline/")

# Apply only the feature selection mask to new data
df_selected = pipeline.named_steps["feature_selector"].transform(df_engineered)

# Retrieve the list of features that survived selection
selected_features = pipeline.named_steps["feature_selector"].get_feature_names_out()
print(selected_features)
```
