"""
inputs.py — Parameter input handling for the resilience model
=============================================================
Supports two input modes:

  1. Programmatic  : instantiate dataclasses directly in Python.
  2. CSV files     : one CSV per parameter group, loaded via load_*_from_csv().

CSV format
----------
Each CSV has exactly two columns: ``parameter`` and ``value``.
Lines starting with ``#`` are treated as comments and ignored.
Example (pipeline.csv):

    # CO2 pipeline parameters
    parameter,value
    capacity_tonne_per_hr,50.0
    utilisation_fraction,0.80
    distance_km,300.0
    failure_probability,0.05
    failure_duration_days,14.0
    opex_per_tonne_km,0.01

String-valued fields (name, mode) are read as strings; all others as float.
"""

from __future__ import annotations

import csv
import dataclasses
from pathlib import Path
from typing import Any, Type, TypeVar

import numpy as np
import pandas as pd

from model import (
    BackupTransportParams,
    CCTSSystemParams,
    PipelineParams,
    SystemSetup,
)

T = TypeVar("T")

# Absolute path to the inputs/ directory, independent of working directory.
_INPUTS_DIR = Path(__file__).parent / "inputs"

# ---------------------------------------------------------------------------
# Grid-builder defaults
# ---------------------------------------------------------------------------

FAILURE_DUR_RANGE = (1.0, 50.0)   # [days]    failure duration  (row axis)
ETS_PRICE_RANGE   = (20.0, 400.0) # [€/t CO2] EU ETS price      (column axis)
DISTANCE_RANGE    = (50.0, 800.0)  # [km]      pipeline distance (column axis)
EMISSIONS_RANGE   = (11.0, 580.0)  # [t/hr]    emitter size       (column axis)
N_GRID_POINTS     = 1000            # resolution along each axis
DEPRECIATION_YEARS_DEFAULT = 20.0 # [years] baseline CapEx amortisation

# Fields that must remain strings (not coerced to float)
_STRING_FIELDS = {"name", "mode", "strategy", "pipeline_location"}


# ---------------------------------------------------------------------------
# Generic CSV loader
# ---------------------------------------------------------------------------

def _load_dataclass_from_csv(csv_path: Path | str, dataclass_type: Type[T]) -> T:
    """
    Load a dataclass from a two-column CSV (parameter, value).

    Only fields present in the CSV are overridden; all other fields
    keep their dataclass default values.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    # Collect default field values; skip init=False fields (set via __post_init__)
    init_fields = [f for f in dataclasses.fields(dataclass_type) if f.init]
    defaults: dict[str, Any] = {
        f.name: f.default
        if f.default is not dataclasses.MISSING
        else f.default_factory()  # type: ignore[misc]
        for f in init_fields
    }

    valid_fields = {f.name for f in init_fields}
    overrides: dict[str, Any] = {}

    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        for row in reader:
            # Skip blank lines and comments
            if not row or row[0].strip().startswith("#"):
                continue
            if row[0].strip().lower() == "parameter":
                continue  # header row

            if len(row) < 2:
                continue

            key = row[0].strip()
            raw_value = row[1].strip()

            if key not in valid_fields:
                raise ValueError(
                    f"Unknown parameter '{key}' for {dataclass_type.__name__}. "
                    f"Valid fields: {sorted(valid_fields)}"
                )

            if key in _STRING_FIELDS:
                overrides[key] = raw_value
            else:
                try:
                    overrides[key] = float(raw_value)
                except ValueError:
                    raise ValueError(
                        f"Cannot convert '{raw_value}' to float for field '{key}'."
                    )

    return dataclass_type(**{**defaults, **overrides})  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Typed loaders (one per dataclass)
# ---------------------------------------------------------------------------

def _load_gwi_csv(path: Path | str) -> tuple[float, float]:
    """Read gwi_fix and gwi_var from a simple two-column CSV."""
    values: dict[str, float] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row or row[0].strip().startswith("#"):
                continue
            if row[0].strip().lower() == "parameter":
                continue
            key = row[0].strip()
            if key in ("gwi_fix", "gwi_var"):
                values[key] = float(row[1].strip())
    return values["gwi_fix"], values["gwi_var"]


def load_pipeline_from_csv(path: Path | str) -> PipelineParams:
    """Load PipelineParams from a CSV file."""
    return _load_dataclass_from_csv(path, PipelineParams)


def load_backup_transport_from_csv(path: Path | str) -> BackupTransportParams:
    """Load BackupTransportParams from a CSV file."""
    return _load_dataclass_from_csv(path, BackupTransportParams)


def load_ccts_system_from_csv(path: Path | str) -> CCTSSystemParams:
    """Load CCTSSystemParams from a CSV file."""
    return _load_dataclass_from_csv(path, CCTSSystemParams)


def load_transport_cost_data(
    path: Path | str | None = None,
) -> dict[str, dict[str, float]]:
    """
    Load the transport cost reference table from a CSV file.

    CSV format: one header row (mode, a1, a2, a3, a4, eq), then one data row per
    transport mode.  Lines starting with ``#`` are ignored.

    Returns
    -------
    dict mapping mode name → {a1, a2, a3, a4, eq}.
    """
    resolved = Path(path) if path is not None else _INPUTS_DIR / "transport_cost_data.csv"
    if not resolved.exists():
        raise FileNotFoundError(f"Transport cost data CSV not found: {resolved}")

    result: dict[str, dict[str, float]] = {}
    with open(resolved, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row or row[0].strip().startswith("#"):
                continue
            if row[0].strip().lower() == "mode":
                continue  # header row
            if len(row) < 5:
                continue
            mode = row[0].strip()
            result[mode] = {
                "a1": float(row[1].strip()),
                "a2": float(row[2].strip()),
                "a3": float(row[3].strip()),
                "a4": float(row[4].strip()),
                "eq": int(row[5].strip()),
            }
    return result


# ---------------------------------------------------------------------------
# Full SystemSetup loader
# ---------------------------------------------------------------------------

def load_system_setup(
    strategy: str,
    transport_ownership: str = "self_owned",
    pipeline_location: str = "onshore",
    depreciation_years: float = 20.0,
    pipeline_csv: Path | str | None = None,
    truck_csv: Path | str | None = None,
    train_csv: Path | str | None = None,
    barge_csv: Path | str | None = None,
    ship_csv: Path | str | None = None,
    ccts_system_csv: Path | str | None = None,
    transport_cost_csv: Path | str | None = None,
    buffer_storage_csv: Path | str | None = None,
    liquefaction_csv: Path | str | None = None,
) -> SystemSetup:
    """
    Build a SystemSetup from CSV files and/or dataclass defaults.

    With no csv path given for a parameter group, dataclass defaults will be used.

    Parameters
    ----------
    strategy:
        One of the Strategy literals defined in model.py.
    transport_ownership:
        One of "self_owned" or "third_party". Affects CapEx/Opex.
    pipeline_location:
        Is the chain on- or offshore?
    depreciation_years:
        CapEx amortisation period [years].
    pipeline_csv, truck_csv, train_csv, barge_csv, ship_csv, ccts_system_csv:
        Optional paths to CSV files for each parameter group.
    transport_cost_csv:
        Path to the transport cost reference CSV. Defaults to
        ``inputs/transport_cost_data.csv``.
    buffer_storage_csv, liquefaction_csv:
        Optional paths for buffer-storage and liquefaction GWI parameters.
    """
    pipeline = (
        load_pipeline_from_csv(pipeline_csv) if pipeline_csv else PipelineParams()
    )
    truck = (
        load_backup_transport_from_csv(truck_csv)
        if truck_csv
        else BackupTransportParams.truck_default()
    )
    train = (
        load_backup_transport_from_csv(train_csv)
        if train_csv
        else BackupTransportParams.train_default()
    )
    barge = (
        load_backup_transport_from_csv(barge_csv)
        if barge_csv
        else BackupTransportParams.barge_default()
    )
    ship = (
        load_backup_transport_from_csv(ship_csv)
        if ship_csv
        else BackupTransportParams.ship_default()
    )
    ccts_system = (
        load_ccts_system_from_csv(ccts_system_csv)
        if ccts_system_csv
        else CCTSSystemParams()
    )
    ccts_system = dataclasses.replace(ccts_system, pipeline_location=pipeline_location)

    gwi_fix_buf, gwi_var_buf = (
        _load_gwi_csv(buffer_storage_csv) if buffer_storage_csv else (5.42, 0.0)
    )
    gwi_fix_liqu, gwi_var_liqu = (
        _load_gwi_csv(liquefaction_csv) if liquefaction_csv else (0.184, 44.6)
    )

    return SystemSetup(
        strategy=strategy,
        transport_ownership=transport_ownership,
        pipeline=pipeline,
        truck=truck,
        train=train,
        barge=barge,
        ship=ship,
        ccts_system=ccts_system,
        depreciation_years=depreciation_years,
        transport_cost_data=load_transport_cost_data(transport_cost_csv),
        gwi_fix_buffer_storage=gwi_fix_buf,
        gwi_var_buffer_storage=gwi_var_buf,
        gwi_fix_liquef=gwi_fix_liqu,
        gwi_var_liquef=gwi_var_liqu,
    )


# ---------------------------------------------------------------------------
# Cost-grid builder
# ---------------------------------------------------------------------------

def _make_axes(
    failure_dur_range: tuple[float, float],
    ets_price_range: tuple[float, float],
    n_points: int,
) -> tuple[np.ndarray, np.ndarray, pd.Index, pd.Index]:
    """Return (durations, prices, dur_idx, price_cols) for the 2-D grid."""
    durations  = np.linspace(*failure_dur_range, n_points)
    prices     = np.linspace(*ets_price_range,   n_points)
    dur_idx    = pd.Index(np.round(durations, 4), name="failure_duration_days")
    price_cols = pd.Index(np.round(prices, 4),    name="ets_price")
    return durations, prices, dur_idx, price_cols


def _make_axes_distance(
    failure_dur_range: tuple[float, float],
    distance_range: tuple[float, float],
    n_points: int,
) -> tuple[np.ndarray, np.ndarray, pd.Index, pd.Index]:
    """Return (durations, distances, dur_idx, dist_cols) for the distance 2-D grid."""
    durations  = np.linspace(*failure_dur_range, n_points)
    distances  = np.linspace(*distance_range,    n_points)
    dur_idx    = pd.Index(np.round(durations, 4), name="failure_duration_days")
    dist_cols  = pd.Index(np.round(distances, 4), name="distance_km")
    return durations, distances, dur_idx, dist_cols


def _make_axes_emissions(
    failure_dur_range: tuple[float, float],
    emissions_range: tuple[float, float],
    n_points: int,
) -> tuple[np.ndarray, np.ndarray, pd.Index, pd.Index]:
    """Return (durations, emissions, dur_idx, emis_cols) for the emissions 2-D grid."""
    durations  = np.linspace(*failure_dur_range, n_points)
    emissions  = np.linspace(*emissions_range,   n_points)
    dur_idx    = pd.Index(np.round(durations, 4), name="failure_duration_days")
    emis_cols  = pd.Index(np.round(emissions, 4), name="emissions_tonne_per_hr")
    return durations, emissions, dur_idx, emis_cols


def build_shared_grids(
    setup: "SystemSetup",
    failure_dur_range: tuple[float, float] = FAILURE_DUR_RANGE,
    ets_price_range: tuple[float, float] = ETS_PRICE_RANGE,
    n_points: int = N_GRID_POINTS,
    cost_mode: str = "annual",
) -> dict[str, pd.DataFrame]:
    """
    Build the two cost grids that are independent of resilience strategy.

    Returns a dict with keys: ``ets_price``, ``do_nothing``.
    Each DataFrame has index = failure_duration_days, columns = ets_price.
    ``cost_mode="per_tonne"`` → values in [€/t]; ``cost_mode="annual"`` → [€/yr].
    """
    durations, prices, dur_idx, price_cols = _make_axes(
        failure_dur_range, ets_price_range, n_points
    )

    if cost_mode == "annual":
        co2_tpa = setup.pipe_trsp_tonne_per_hr * 8760
        ets_price_df = pd.DataFrame(
            np.tile(prices * co2_tpa, (n_points, 1)), index=dur_idx, columns=price_cols
        )
    else:
        ets_price_df = pd.DataFrame(
            np.tile(prices, (n_points, 1)), index=dur_idx, columns=price_cols
        )

    do_nothing_df = build_resilience_grid(
        setup, strategy="do_nothing",
        failure_dur_range=failure_dur_range,
        ets_price_range=ets_price_range,
        n_points=n_points,
        cost_mode=cost_mode,
    )

    return {
        "ets_price": ets_price_df,
        "do_nothing": do_nothing_df,
    }


def build_resilience_grid(
    setup: "SystemSetup",
    strategy: str,
    failure_dur_range: tuple[float, float] = FAILURE_DUR_RANGE,
    ets_price_range: tuple[float, float] = ETS_PRICE_RANGE,
    n_points: int = N_GRID_POINTS,
    cost_mode: str = "annual",
) -> pd.DataFrame:
    """
    Build a single resilience cost grid for *strategy* over the 2-D parameter space.

    Returns a DataFrame with index = failure_duration_days, columns = ets_price.
    ``cost_mode="per_tonne"`` → values in [€/t]; ``cost_mode="annual"`` → [€/yr].
    """
    from model import calc_resilience_cost

    durations, prices, dur_idx, price_cols = _make_axes(
        failure_dur_range, ets_price_range, n_points
    )

    cost_key = "cost_per_tonne" if cost_mode == "per_tonne" else "total_cost_annual"
    base = dataclasses.replace(setup, strategy=strategy)
    arr  = np.zeros((n_points, n_points))
    for i, dur in enumerate(durations):
        new_pipeline = dataclasses.replace(
            base.pipeline, failure_duration_days=float(dur),
        )
        for j, price in enumerate(prices):
            new_ccts = dataclasses.replace(
                base.ccts_system, ets_price=float(price),
            )
            cell_setup = dataclasses.replace(
                base, pipeline=new_pipeline, ccts_system=new_ccts,
            )
            arr[i, j] = calc_resilience_cost(cell_setup)[cost_key]
    return pd.DataFrame(arr, index=dur_idx, columns=price_cols)


def build_distance_grid(
    setup: "SystemSetup",
    strategy: str,
    failure_dur_range: tuple[float, float] = FAILURE_DUR_RANGE,
    distance_range: tuple[float, float] = DISTANCE_RANGE,
    n_points: int = N_GRID_POINTS,
    cost_mode: str = "annual",
) -> pd.DataFrame:
    """
    Build a single resilience cost grid for *strategy* over
    failure_duration_days × distance_km.

    ETS price is held fixed at the nominal value in *setup*.
    Returns a DataFrame with index = failure_duration_days, columns = distance_km.
    """
    from model import calc_resilience_cost

    durations, distances, dur_idx, dist_cols = _make_axes_distance(
        failure_dur_range, distance_range, n_points
    )

    cost_key = "cost_per_tonne" if cost_mode == "per_tonne" else "total_cost_annual"
    base = dataclasses.replace(setup, strategy=strategy)
    arr  = np.zeros((n_points, n_points))
    for i, dur in enumerate(durations):
        new_pipeline = dataclasses.replace(
            base.pipeline, failure_duration_days=float(dur),
        )
        for j, dist in enumerate(distances):
            new_ccts = dataclasses.replace(
                base.ccts_system, distance_km=float(dist),
            )
            cell_setup = dataclasses.replace(
                base, pipeline=new_pipeline, ccts_system=new_ccts,
            )
            arr[i, j] = calc_resilience_cost(cell_setup)[cost_key]
    return pd.DataFrame(arr, index=dur_idx, columns=dist_cols)


def build_emissions_grid(
    setup: "SystemSetup",
    strategy: str,
    failure_dur_range: tuple[float, float] = FAILURE_DUR_RANGE,
    emissions_range: tuple[float, float] = EMISSIONS_RANGE,
    n_points: int = N_GRID_POINTS,
    cost_mode: str = "annual",
) -> pd.DataFrame:
    """
    Build a single resilience cost grid for *strategy* over
    failure_duration_days × emissions_tonne_per_hr.

    ETS price and distance are held fixed at the nominal values in *setup*.
    Returns a DataFrame with index = failure_duration_days, columns = emissions_tonne_per_hr.
    """
    from model import calc_resilience_cost

    durations, emissions, dur_idx, emis_cols = _make_axes_emissions(
        failure_dur_range, emissions_range, n_points
    )

    cost_key = "cost_per_tonne" if cost_mode == "per_tonne" else "total_cost_annual"
    base = dataclasses.replace(setup, strategy=strategy)
    arr  = np.zeros((n_points, n_points))
    for i, dur in enumerate(durations):
        new_pipeline = dataclasses.replace(
            base.pipeline, failure_duration_days=float(dur),
        )
        for j, emis in enumerate(emissions):
            new_ccts = dataclasses.replace(
                base.ccts_system, emissions_tonne_per_hr=float(emis),
            )
            cell_setup = dataclasses.replace(
                base, pipeline=new_pipeline, ccts_system=new_ccts,
            )
            arr[i, j] = calc_resilience_cost(cell_setup)[cost_key]
    return pd.DataFrame(arr, index=dur_idx, columns=emis_cols)


def build_shared_distance_grids(
    setup: "SystemSetup",
    failure_dur_range: tuple[float, float] = FAILURE_DUR_RANGE,
    distance_range: tuple[float, float] = DISTANCE_RANGE,
    n_points: int = N_GRID_POINTS,
    cost_mode: str = "annual",
) -> dict[str, pd.DataFrame]:
    """
    Build the do-nothing reference grid for the distance-sweep heatmap.

    Returns a dict with key ``do_nothing``; DataFrame has
    index = failure_duration_days, columns = distance_km.
    The do-nothing cost includes pipeline cost (which scales with distance)
    plus the ETS penalty for the unhandled CO2 during failure.
    """
    do_nothing_grid = build_distance_grid(
        setup, strategy="do_nothing",
        failure_dur_range=failure_dur_range,
        distance_range=distance_range,
        n_points=n_points,
        cost_mode=cost_mode,
    )

    return {"do_nothing": do_nothing_grid}


def build_cost_grids(
    setup: "SystemSetup",
    strategy: str = "truck",
    failure_dur_range: tuple[float, float] = FAILURE_DUR_RANGE,
    ets_price_range: tuple[float, float] = ETS_PRICE_RANGE,
    n_points: int = N_GRID_POINTS,
) -> dict[str, pd.DataFrame]:
    """
    Build four cost-grid DataFrames over a 2-D parameter space.

    Thin wrapper around :func:`build_shared_grids` and :func:`build_resilience_grid`
    for backward compatibility. Returns dict with keys:
    ``ets_price``, ``normal_operation``, ``do_nothing``, ``resilience``.
    """
    shared = build_shared_grids(setup, failure_dur_range, ets_price_range, n_points)
    do_nothing_df = build_resilience_grid(
        setup, "do_nothing", failure_dur_range, ets_price_range, n_points
    )
    resilience_df = build_resilience_grid(
        setup, strategy, failure_dur_range, ets_price_range, n_points
    )
    return {**shared, "do_nothing": do_nothing_df, "resilience": resilience_df}
