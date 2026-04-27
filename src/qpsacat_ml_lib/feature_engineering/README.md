# Feature Engineering Module

This module implements the **Feature Engineering** stage (stage 2) of the `qpsacat_ml_lib` training pipeline.

It is controlled by the `transform_features` section of `config.yaml`.

---

## Role in the Pipeline

```
DataLoader → [FeatureEngineer] → FeatureSelector → Optimiser → Trainer
```

The `FeatureEngineer` is **fitted on the training split only**, stores its state, and is serialised inside the pipeline object saved to MLflow.  
When the saved pipeline is loaded for inference, the exact same transformations are replayed automatically.

---

## Configuration

```yaml
transform_features:
  enable: true   # false = skip this stage entirely

  steps:
    # ── Ratio ─────────────────────────────────────────────────
    - name: invoice_ratio
      type: ratio
      numerator: invoice_amount
      denominator: contract_value
      fill_value: 0.0          # used when denominator == 0

    # ── Date difference ───────────────────────────────────────
    - name: days_to_expiry
      type: date_diff
      start_col: reference_date
      end_col: expiry_date
      unit: days               # "days" or "months"

    # ── Clip / cap ────────────────────────────────────────────
    - name: col_capped
      type: clip
      source: raw_col
      lower: 0.0
      upper: 99.0

    # ── Log transform ─────────────────────────────────────────
    - name: col_log
      type: log1p
      source: raw_col

    # ── One-hot encoding ──────────────────────────────────────
    - name: category_encoded
      type: one_hot
      source: category_col
      drop_first: true
```

---

## Supported Step Types

| `type` | Output | Notes |
|--------|--------|-------|
| `ratio` | `numerator / denominator` | `fill_value` applied where denominator is zero |
| `date_diff` | integer difference | supports `unit: days` and `unit: months` |
| `clip` | bounded numeric column | replaces values outside `[lower, upper]` |
| `log1p` | `log(1 + x)` | input must be ≥ 0 |
| `one_hot` | binary dummy columns | `drop_first: true` avoids perfect multicollinearity |

---

## Using the FeatureEngineer at Inference Time

After loading the saved MLflow pipeline, the feature engineering transformations are already embedded:

```python
import mlflow.pyfunc
import pandas as pd

# Load the full pipeline from MLflow
pipeline = mlflow.pyfunc.load_model("runs:/<RUN_ID>/pipeline/")

# Option 1 – apply feature engineering only
df_engineered = pipeline.named_steps["feature_engineering"].transform(df_raw)

# Option 2 – full end-to-end prediction (preprocessing + FE + selection + model)
predictions = pipeline.predict(df_raw)
```

Pass a DataFrame with the **same column schema** as the original training data.
