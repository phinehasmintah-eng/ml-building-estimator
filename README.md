# ML Building Material & Cost Estimator

A construction material and cost estimator for Ghanaian building projects
that uses **trained, validated machine learning models** — not hardcoded
if/elif formulas — to generate predictions.

## Why this project is different from a formula-only calculator

A common shortcut is to build a calculator with fixed engineering formulas
and describe it as "ML" without ever training a model. This project avoids
that: there is no public dataset for this problem, so a realistic dataset is
**synthesized** using standard construction engineering formulas plus
injected real-world noise (site wastage variance, regional price
fluctuation, finish-quality effects). Multiple ML models are then trained
and evaluated on that data with proper train/validation/test splits and
10-fold cross-validation — so the comparative evaluation and error metrics
are real, reproducible numbers, not narrative.

## Project structure

```
estimator/
├── data/
│   ├── generate_dataset.py     # synthesizes construction_dataset.csv
│   └── construction_dataset.csv
├── model/
│   ├── train_models.py         # trains & evaluates LinearRegression, RandomForest, GradientBoosting
│   ├── model_total_cost_ghs.joblib
│   ├── model_total_cement_bags.joblib
│   └── evaluation_report.json  # full metrics for all models
├── templates/index.html
├── static/css/style.css
├── app.py                      # Flask app, loads trained models for real-time prediction
└── requirements.txt
```

## Running it

```bash
pip install -r requirements.txt

# (optional — pre-trained models are already included)
cd data && python generate_dataset.py && cd ..
cd model && python train_models.py && cd ..

python app.py
# open http://localhost:8000
```

## Model performance (held-out test set)

| Target             | Best model       | R²    | MAPE  |
|---------------------|-------------------|-------|-------|
| Total cost (GHS)    | Gradient Boosting | 0.974 | 9.0%  |
| Cement bags needed  | Gradient Boosting | 0.983 | 7.0%  |

Gradient Boosting and Random Forest both substantially outperform Linear
Regression (which sits around 50% MAPE), confirming that the relationship
between project inputs and material/cost outcomes is non-linear — which is
the actual justification for using ML here rather than a simple formula.

## Known limitation

The dataset is synthetic. It is built from the same engineering formulas
used in professional QS practice plus realistic noise, but it is not real
historical project data. This is disclosed explicitly in the report; a
natural extension is to retrain the same pipeline on real BOQ (Bill of
Quantities) data once it becomes available, without changing the
architecture.
