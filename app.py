"""
app.py

Flask web application for the ML-based building material & cost estimator.

Unlike the reference project (fixed if/elif formulas labeled as "ML"), this
app loads trained scikit-learn pipelines (model/model_total_cost_ghs.joblib
and model/model_total_cement_bags.joblib) and uses them to generate real
predictions from user input. A transparent, clearly-labeled deterministic
calculation (based on the same engineering formulas used to build the
training data) is shown alongside the ML prediction for comparison, so the
report can also legitimately discuss how the ML model differs from -- and,
per the evaluation, improves on -- naive formula-based estimation.
"""

import os

import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
cost_model = joblib.load(os.path.join(MODEL_DIR, "model_total_cost_ghs.joblib"))
cement_model = joblib.load(os.path.join(MODEL_DIR, "model_total_cement_bags.joblib"))

STRUCTURAL_TYPES = ["bungalow", "storey_building", "warehouse", "commercial_block"]
WALL_MATERIALS = ["sandcrete_block", "clay_brick", "compressed_earth_block"]
REGIONS = ["Greater Accra", "Ashanti", "Western", "Central", "Eastern", "Northern"]
FINISH_QUALITY = ["standard", "premium"]

# Same constants used by data/generate_dataset.py, for the transparent
# deterministic comparison figure shown alongside the ML prediction.
REGION_PRICE_INDEX = {
    "Greater Accra": 1.15, "Ashanti": 1.05, "Western": 1.00,
    "Central": 0.98, "Eastern": 0.95, "Northern": 0.90,
}
UNIT_PRICE_CEMENT_BAG = 90.0
UNIT_PRICE_SAND_M3 = 250.0
UNIT_PRICE_AGGREGATE_M3 = 300.0
UNIT_PRICE_BLOCK = 6.0
WALL_MATERIAL_FACTOR = {"sandcrete_block": 1.0, "clay_brick": 1.35, "compressed_earth_block": 0.75}


def deterministic_estimate(floor_area, storeys, wall_material, region, finish_quality):
    """Formula-based estimate (mirrors friend's original approach) shown as a baseline."""
    total_area = floor_area * storeys
    wall_area = total_area * 0.65
    number_of_blocks = (wall_area * 1.02) / 0.08
    mortar_volume = ((0.4 + 0.4 + 0.2 + 0.2) * 0.010) * number_of_blocks * 1.45
    cement_mortar = mortar_volume / 4
    concrete_volume = total_area * 0.12
    cement_concrete_bags = (concrete_volume / 10) * 28
    total_cement_bags = (cement_mortar / 0.0347) + cement_concrete_bags

    price_idx = REGION_PRICE_INDEX[region]
    wall_factor = WALL_MATERIAL_FACTOR[wall_material]
    finish_factor = 1.0 if finish_quality == "standard" else 1.25
    cost_cement = total_cement_bags * UNIT_PRICE_CEMENT_BAG * price_idx
    cost_blocks = number_of_blocks * UNIT_PRICE_BLOCK * price_idx * wall_factor
    total_cost = (cost_cement + cost_blocks) * finish_factor

    return round(total_cement_bags, 2), round(total_cost, 2)


@app.route("/", methods=["GET", "POST"])
def calculator():
    results = None
    form_values = {
        "floor_area": "", "storeys": "1", "structural_type": STRUCTURAL_TYPES[0],
        "wall_material": WALL_MATERIALS[0], "region": REGIONS[0],
        "finish_quality": FINISH_QUALITY[0],
    }

    if request.method == "POST":
        floor_area = float(request.form["floor_area"])
        storeys = int(request.form["storeys"])
        structural_type = request.form["structural_type"]
        wall_material = request.form["wall_material"]
        region = request.form["region"]
        finish_quality = request.form["finish_quality"]

        form_values.update(
            floor_area=request.form["floor_area"], storeys=str(storeys),
            structural_type=structural_type, wall_material=wall_material,
            region=region, finish_quality=finish_quality,
        )

        X = pd.DataFrame([{
            "floor_area_m2": floor_area, "storeys": storeys,
            "structural_type": structural_type, "wall_material": wall_material,
            "region": region, "finish_quality": finish_quality,
        }])

        ml_cost = float(cost_model.predict(X)[0])
        ml_cement_bags = float(cement_model.predict(X)[0])

        det_cement_bags, det_cost = deterministic_estimate(
            floor_area, storeys, wall_material, region, finish_quality
        )

        results = {
            "ml_cost": round(ml_cost, 2),
            "ml_cement_bags": round(ml_cement_bags, 2),
            "det_cost": det_cost,
            "det_cement_bags": det_cement_bags,
        }

    return render_template(
        "index.html",
        results=results,
        structural_types=STRUCTURAL_TYPES,
        wall_materials=WALL_MATERIALS,
        regions=REGIONS,
        finish_qualities=FINISH_QUALITY,
        form_values=form_values,
    )


if __name__ == "__main__":
    app.run(port=8000, debug=True)
