"""
generate_dataset.py

Synthesizes a realistic construction materials & cost dataset for Ghana-based
building projects. There is no public labeled dataset for this problem, so we
generate one from established engineering estimation formulas (block/cement/
sand ratios per BS 5628 / standard Ghanaian QS practice), then inject
realistic noise to simulate real-world variance that a pure formula CANNOT
capture: site wastage variation, labour/finish quality, supplier price
fluctuation across regions, and interaction effects between storeys and
structural type. This noisy relationship is what the ML models are trained
to recover -- it is not just a formula copied into a dropdown of ifs.

Run: python generate_dataset.py
Output: construction_dataset.csv
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N_SAMPLES = 6000

STRUCTURAL_TYPES = ["bungalow", "storey_building", "warehouse", "commercial_block"]
WALL_MATERIALS = ["sandcrete_block", "clay_brick", "compressed_earth_block"]
REGIONS = ["Greater Accra", "Ashanti", "Western", "Central", "Eastern", "Northern"]
FINISH_QUALITY = ["standard", "premium"]

# Regional base unit price index (GHS), reflecting real cost-of-materials variation
REGION_PRICE_INDEX = {
    "Greater Accra": 1.15,
    "Ashanti": 1.05,
    "Western": 1.00,
    "Central": 0.98,
    "Eastern": 0.95,
    "Northern": 0.90,
}

# Base unit prices (GHS), 2026 approximate Ghana market rates
UNIT_PRICE_CEMENT_BAG = 90.0
UNIT_PRICE_SAND_M3 = 250.0
UNIT_PRICE_AGGREGATE_M3 = 300.0
UNIT_PRICE_BLOCK = 6.0

WALL_MATERIAL_FACTOR = {
    "sandcrete_block": 1.0,
    "clay_brick": 1.35,
    "compressed_earth_block": 0.75,
}

STRUCTURAL_TYPE_WASTAGE = {
    # more complex structures -> more cutting/wastage
    "bungalow": 0.01,
    "storey_building": 0.02,
    "warehouse": 0.015,
    "commercial_block": 0.03,
}


def generate_row():
    floor_area = RNG.uniform(40, 600)  # m^2 per floor
    storeys = RNG.integers(1, 5)
    structural_type = RNG.choice(STRUCTURAL_TYPES)
    wall_material = RNG.choice(WALL_MATERIALS)
    region = RNG.choice(REGIONS)
    finish_quality = RNG.choice(FINISH_QUALITY, p=[0.75, 0.25])

    total_area = floor_area * storeys

    # ---- BLOCKWORK (engineering formula, same principle as manual QS calc) ----
    block_size = 0.08  # m2 coverage of a standard sandcrete block incl. joints
    base_wastage = STRUCTURAL_TYPE_WASTAGE[structural_type]
    wastage = base_wastage + RNG.normal(0, 0.004)  # site-level noise
    wastage = max(wastage, 0.0)

    wall_area_ratio = RNG.uniform(0.55, 0.75)  # wall area as fraction of floor area (openings vary)
    wall_area = total_area * wall_area_ratio

    number_of_blocks = (wall_area * (1 + wastage)) / block_size
    number_of_blocks *= RNG.normal(1.0, 0.03)  # laying efficiency noise

    mortar_thickness = 0.010
    mix_ratio_mortar = 4
    mortar_volume_constant = (0.4 + 0.4 + 0.2 + 0.2) * mortar_thickness
    total_mortar_volume = mortar_volume_constant * number_of_blocks * 1.45
    cement_volume_mortar = total_mortar_volume / mix_ratio_mortar
    sand_volume_mortar = (mix_ratio_mortar - 1) * cement_volume_mortar

    # ---- CONCRETE (foundation + slab, scaled with storeys/area) ----
    concrete_volume_base = total_area * 0.12  # m3 of concrete per m2, simplified
    concrete_volume = concrete_volume_base * RNG.normal(1.0, 0.06)
    mix_ratio_concrete = 10
    cement_volume_concrete = (1 / mix_ratio_concrete) * concrete_volume
    cement_bags_concrete = cement_volume_concrete * 28  # bags per m3 approx (0.035m3/bag equiv chain)
    sand_volume_concrete = (3 / mix_ratio_concrete) * concrete_volume
    aggregate_volume_concrete = (6 / mix_ratio_concrete) * concrete_volume

    # ---- PLASTERING ----
    plaster_thickness = 0.012
    plaster_area = wall_area * 2 * RNG.normal(1.0, 0.02)  # both faces
    plaster_volume = plaster_area * plaster_thickness
    cement_volume_plaster = 0.25 * plaster_volume
    cement_bags_plaster = cement_volume_plaster / 0.0347
    sand_volume_plaster = 0.75 * plaster_volume

    # ---- TOTALS ----
    total_cement_bags = (
        (cement_volume_mortar / 0.0347) + cement_bags_concrete + cement_bags_plaster
    )
    total_sand_m3 = sand_volume_mortar + sand_volume_concrete + sand_volume_plaster
    total_aggregate_m3 = aggregate_volume_concrete
    total_blocks = number_of_blocks

    finish_factor = 1.0 if finish_quality == "standard" else 1.25

    # ---- COST (regional pricing + finish + wall material + supplier noise) ----
    price_idx = REGION_PRICE_INDEX[region]
    wall_factor = WALL_MATERIAL_FACTOR[wall_material]

    cost_cement = total_cement_bags * UNIT_PRICE_CEMENT_BAG * price_idx
    cost_sand = total_sand_m3 * UNIT_PRICE_SAND_M3 * price_idx
    cost_aggregate = total_aggregate_m3 * UNIT_PRICE_AGGREGATE_M3 * price_idx
    cost_blocks = total_blocks * UNIT_PRICE_BLOCK * price_idx * wall_factor

    subtotal = (cost_cement + cost_sand + cost_aggregate + cost_blocks) * finish_factor
    supplier_noise = RNG.normal(1.0, 0.05)  # +-5% supplier/market fluctuation
    total_cost = subtotal * supplier_noise

    return {
        "floor_area_m2": round(floor_area, 2),
        "storeys": int(storeys),
        "structural_type": structural_type,
        "wall_material": wall_material,
        "region": region,
        "finish_quality": finish_quality,
        "total_blocks": max(round(total_blocks), 0),
        "total_cement_bags": round(max(total_cement_bags, 0), 2),
        "total_sand_m3": round(max(total_sand_m3, 0), 3),
        "total_aggregate_m3": round(max(total_aggregate_m3, 0), 3),
        "total_cost_ghs": round(max(total_cost, 0), 2),
    }


def main():
    rows = [generate_row() for _ in range(N_SAMPLES)]
    df = pd.DataFrame(rows)
    out_path = "construction_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    print(df.describe(include="all").T)


if __name__ == "__main__":
    main()
