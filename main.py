"""
main.py — Analysis runner for the CO2 Transport Resilience Cost Model
=====================================================================
Used to orchestrate model, inputs, and resilience strategies.

    python main.py

"""

from __future__ import annotations

from pathlib import Path

from inputs import (
    DISTANCE_RANGE,
    EMISSIONS_RANGE,
    build_distance_grid,
    build_emissions_grid,
    build_resilience_grid,
    build_shared_grids,
    load_system_setup,
)


_INPUTS_DIR = Path(__file__).parent / "inputs"

# Onshore: truck, train, barge, storage (no ship)
onshore_setup = load_system_setup(
    strategy="do_nothing",
    transport_ownership="self_owned",  # "third_party" or "self_owned"
    pipeline_location='onshore',
    pipeline_csv=_INPUTS_DIR / "pipeline.csv",
    truck_csv=_INPUTS_DIR / "truck.csv",
    train_csv=_INPUTS_DIR / "train.csv",
    barge_csv=_INPUTS_DIR / "barge.csv",
    ship_csv=_INPUTS_DIR / "ship.csv",
    ccts_system_csv=_INPUTS_DIR / "ccts_system.csv",
)
# Offshore: do_nothing, ship, storage only
offshore_setup = load_system_setup(
    strategy="do_nothing",
    transport_ownership="self_owned",  # "third_party" or "self_owned"
    pipeline_location='offshore',
    pipeline_csv=_INPUTS_DIR / "pipeline.csv",
    truck_csv=_INPUTS_DIR / "truck.csv",
    train_csv=_INPUTS_DIR / "train.csv",
    barge_csv=_INPUTS_DIR / "barge.csv",
    ship_csv=_INPUTS_DIR / "ship.csv",
    ccts_system_csv=_INPUTS_DIR / "ccts_system.csv",
)
ets_range=(100.0, 400.0)
n_resolution = 800

onshore_shared = build_shared_grids(onshore_setup, n_points=n_resolution,
                                        ets_price_range=ets_range)
onshore_resilience = {
    "do_nothing": onshore_shared["do_nothing"],
    **{
        strategy: build_resilience_grid(onshore_setup, strategy=strategy,
                                        n_points=n_resolution,
                                        ets_price_range=ets_range)
        for strategy in ['truck', 'train', 'barge', 'storage']
    }
}
offshore_shared = build_shared_grids(offshore_setup, n_points=n_resolution,
                                         ets_price_range=ets_range)
offshore_resilience = {
    "do_nothing": offshore_shared["do_nothing"],
    **{
        strategy: build_resilience_grid(offshore_setup, strategy=strategy,
                                        n_points=n_resolution,
                                        ets_price_range=ets_range)
        for strategy in ['ship', 'storage']
    }
}
