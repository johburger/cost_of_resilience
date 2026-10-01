"""
sobol_analysis.py — Global Sensitivity Analysis via Sobol' Indices.

Quantifies which model parameters most significantly influence the
cost_per_tonne of a chosen resilience strategy.

Usage
-----
    python sobol_analysis.py

For custom analyses, import this module and call ``main(...)``.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import dataclasses
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from SALib.analyze.sobol import analyze as sobol_analyze
from SALib.sample.sobol import sample as sobol_sample

from inputs import load_transport_cost_data
from model import (
    BackupTransportParams,
    CCTSSystemParams,
    PipelineParams,
    SystemSetup,
    calc_resilience_cost,
)

_INPUTS_DIR = Path(__file__).parent / "inputs"
_SAMPLES_DIR = Path(__file__).parent / "samples"
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Problem definitions
# ---------------------------------------------------------------------------

_NAMES_FULL = [
    "distance_km",
    "emissions_tonne_per_hr",
    "ets_price",
    "interest_rate",
    "capex_capture",
    "opex_capture",
    "capex_compression",
    "opex_compression",
    "opex_geol_storage",
    "depreciation_years",
    "failure_duration_days",
    "backup_opex_factor",
    "backup_capex_factor",
    "buffer_capex_factor",
    "buffer_opex_factor",
    "cond_capex_factor",
    "cond_opex_factor",
]

_BOUNDS_FULL = [
    [50.0,         800.0],        # distance_km [km]
    [11.0,         580.0],        # emissions_tonne_per_hr [t/hr]
    [50.0,         400.0],        # ets_price [€/t CO₂]
    [0.02,          0.12],        # interest_rate [-] ±50 % of 0.07
    [4_203_124.0, 12_609_371.0],  # capex_capture [€/(t CO₂/hr)] ±50 % of 8 406 247
    [12.42,         37.26],       # opex_capture [€/t CO₂] ±50 % of 24.84
    [79_213.0, 237_639.0],        # capex_compression [€/(t CO₂/hr)] ±50 % of 158 426
    [7.7,           23.1],        # opex_compression [€/t CO₂] ±50 % of 15.4
    [20.36,         61.07],       # opex_geol_storage [€/t CO₂] ±50 % of 40.71
    [10.0,          40.0],        # depreciation_years [yr]
    [1.0,           50.0],        # failure_duration_days [days]
    [0.5,            1.5],        # backup_opex_factor [-]  ±50 %
    [0.5,            1.5],        # backup_capex_factor [-] ±50 %
    [0.5,            1.5],        # buffer_capex_factor [-] ±50 %
    [0.5,            1.5],        # buffer_opex_factor [-]  ±50 %
    [0.5,            1.5],        # cond_capex_factor [-]   ±50 %
    [0.5,            1.5],        # cond_opex_factor [-]    ±50 %
]

# 17-parameter problem: used for "cost_per_tonne" and "resilience_premium"
PROBLEM_FULL: dict = {"num_vars": 17, "names": _NAMES_FULL, "bounds": _BOUNDS_FULL}

# 11-parameter problem for "breakeven_ets": failure_duration and ets_price excluded.
_BREAKEVEN_EXCLUDE = {"failure_duration_days", "ets_price", "distance_km"}
DISTANCE_KM_BREAKEVEN = 200.0  # [km] fixed value when excluded
FAILURE_DURATION_BREAKEVEN = 26  # [days] fixed value when excluded
ETS_PRICE_BREAKEVEN = 200.0        # [€/t]  reference value (overridden analytically)
_LIQU_CAPEX_INTEGRATION_FACTOR = 0.35  # CapEx −65 % for integrated liquefaction

LOCATION_STRATEGIES: dict[str, list[str]] = {
    "onshore": ["truck", "train", "barge", "storage"],
    "offshore": ["ship", "storage"],
}

PROBLEM_BREAKEVEN: dict = {
    "num_vars": PROBLEM_FULL["num_vars"] - len(_BREAKEVEN_EXCLUDE),
    "names":  [n for n in _NAMES_FULL if n not in _BREAKEVEN_EXCLUDE],
    "bounds": [b for n, b in zip(_NAMES_FULL, _BOUNDS_FULL)
               if n not in _BREAKEVEN_EXCLUDE],
}


def _get_problem(output_metric: str) -> dict:
    return PROBLEM_BREAKEVEN if output_metric == "breakeven_ets" else PROBLEM_FULL


# Human-readable labels for the parameter names
PARAM_LABELS: dict[str, str] = {
    "distance_km":            "Distance [km]",
    "emissions_tonne_per_hr": "Emissions [t/h]",
    "ets_price":              "Emission price [€/t]",
    "interest_rate":          "Interest rate",
    "capex_capture":          "Capex capture",
    "opex_capture":           "Opex capture",
    "capex_compression":      "Capex compression",
    "opex_compression":       "Opex compression",
    "opex_geol_storage":      "Opex geol. storage",
    "depreciation_years":     "Depreciation years",
    "failure_duration_days":  "Failure duration [d]",
    "backup_opex_factor":     "Opex backup transport",
    "backup_capex_factor":    "Capex backup transport",
    "buffer_capex_factor":    "capex buffer storage",
    "buffer_opex_factor":     "opex buffer storage",
    "cond_capex_factor":      "Capex liquefaction",
    "cond_opex_factor":       "Opex liquefaction",
}

TABLE_ROW_LABELS: dict[str, str] = {
    "distance_km":            r"Pipeline distance $d$",
    "emissions_tonne_per_hr": r"Plant \ce{CO2} emission rate $\dot{m}^s_{em}$",
    "ets_price":              r"Emissions price $p_{em}$",
    "interest_rate":          r"Interest rate $i$",
    "capex_capture":          "Capture CAPEX",
    "opex_capture":           "Capture OPEX",
    "capex_compression":      "Compression CAPEX",
    "opex_compression":       "Compression OPEX",
    "opex_geol_storage":      "Geological storage fee",
    "depreciation_years":     r"Depreciation time $n$",
    "failure_duration_days":  r"Annual disruption $\Delta t_f$",
    "backup_opex_factor":     "Backup transport OPEX factor",
    "backup_capex_factor":    "Backup transport CAPEX factor",
    "buffer_capex_factor":    "Buffer storage CAPEX factor",
    "buffer_opex_factor":     "Buffer storage OPEX factor",
    "cond_capex_factor":      "Liquefaction CAPEX factor",
    "cond_opex_factor":       "Liquefaction OPEX factor",
}

TABLE_COL_LABELS: dict[str, str] = {
    "truck":   "Truck",
    "train":   "Train",
    "barge":   "Barge",
    "storage": "Storage",
    "ship":    "Ship",
}

# Output metric options
# ---------------------
#   "cost_per_tonne"     total all-in specific cost of the strategy [€/t]
#   "resilience_premium" strategy cost_per_tonne minus do-nothing cost_per_tonne [€/t]
#   "breakeven_ets"      ETS price [€/t] at which strategy == do-nothing;
#                        sampled ets_price plays no role (S_T ≈ 0 by construction)

# ---------------------------------------------------------------------------
# Per-process worker state
# ---------------------------------------------------------------------------

_transport_cost_data: dict = {}


def _init_worker(tcd: dict) -> None:
    global _transport_cost_data
    _transport_cost_data = tcd


def _evaluate_row(args: tuple) -> float:
    """Map one Sobol sample row to the requested output_metric scalar."""
    row, strategy, output_metric, integration, location = args
    liqu_capex_factor = _LIQU_CAPEX_INTEGRATION_FACTOR if integration else 1.0
    transport_ownership = "third_party" if integration else "self_owned"
    pipeline_location = location

    if output_metric == "breakeven_ets":
        (
            emissions_tonne_per_hr, interest_rate,
            capex_capture, opex_capture, capex_compression, opex_compression,
            opex_geol_storage, depreciation_years,
            backup_opex_factor, backup_capex_factor,
            buffer_capex_factor, buffer_opex_factor,
            cond_capex_factor, cond_opex_factor,
        ) = row
        distance_km = DISTANCE_KM_BREAKEVEN
        ets_price = ETS_PRICE_BREAKEVEN
        failure_duration_days = FAILURE_DURATION_BREAKEVEN
    else:
        (
            distance_km, emissions_tonne_per_hr, ets_price, interest_rate,
            capex_capture, opex_capture, capex_compression, opex_compression,
            opex_geol_storage, depreciation_years, failure_duration_days,
            backup_opex_factor, backup_capex_factor,
            buffer_capex_factor, buffer_opex_factor,
            cond_capex_factor, cond_opex_factor,
        ) = row

    ccts = CCTSSystemParams(
        distance_km=float(distance_km),
        emissions_tonne_per_hr=float(emissions_tonne_per_hr),
        ets_price=float(ets_price),
        interest_rate=float(interest_rate),
        capex_capture=float(capex_capture),
        opex_capture=float(opex_capture),
        capex_compression=float(capex_compression),
        opex_compression=float(opex_compression),
        opex_geol_storage=float(opex_geol_storage),
        pipeline_location=pipeline_location
    )
    pipeline = PipelineParams(failure_duration_days=int(failure_duration_days))
    truck = dataclasses.replace(
        BackupTransportParams.truck_default(),
        opex_factor=float(backup_opex_factor),
        capex_factor=float(backup_capex_factor),
    )
    train = dataclasses.replace(
        BackupTransportParams.train_default(),
        opex_factor=float(backup_opex_factor),
        capex_factor=float(backup_capex_factor),
    )
    barge = dataclasses.replace(
        BackupTransportParams.barge_default(),
        opex_factor=float(backup_opex_factor),
        capex_factor=float(backup_capex_factor),
    )
    ship = dataclasses.replace(
        BackupTransportParams.ship_default(),
        opex_factor=float(backup_opex_factor),
        capex_factor=float(backup_capex_factor),
    )

    setup = SystemSetup(
        strategy=strategy,
        transport_ownership=transport_ownership,
        pipeline=pipeline,
        ccts_system=ccts,
        truck=truck,
        train=train,
        barge=barge,
        ship=ship,
        depreciation_years=float(depreciation_years),
        transport_cost_data=_transport_cost_data,
        liqu_capex_factor=liqu_capex_factor,
        buffer_capex_factor=float(buffer_capex_factor),
        buffer_opex_factor=float(buffer_opex_factor),
        cond_capex_factor=float(cond_capex_factor),
        cond_opex_factor=float(cond_opex_factor),
    )

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            if output_metric == "cost_per_tonne":
                return float(calc_resilience_cost(setup)["cost_per_tonne"])

            elif output_metric == "resilience_premium":
                cost_s = calc_resilience_cost(setup)["total_cost_annual"]
                cost_dn = calc_resilience_cost(
                    dataclasses.replace(setup, strategy="do_nothing")
                )["total_cost_annual"]
                return float(cost_s - cost_dn)

            elif output_metric == "breakeven_ets":
                # Sampled ets_price is replaced; output is independent of it.
                # Costs are linear in ETS → solve crossing from two evaluations.
                _ets_lo, _ets_hi = 0.0, 500.0
                ccts_lo = dataclasses.replace(ccts, ets_price=_ets_lo)
                ccts_hi = dataclasses.replace(ccts, ets_price=_ets_hi)
                cost_s_lo  = calc_resilience_cost(
                    dataclasses.replace(setup, ccts_system=ccts_lo)
                )["total_cost_annual"]
                cost_s_hi  = calc_resilience_cost(
                    dataclasses.replace(setup, ccts_system=ccts_hi)
                )["total_cost_annual"]
                cost_dn_lo = calc_resilience_cost(
                    dataclasses.replace(setup, ccts_system=ccts_lo,
                                        strategy="do_nothing")
                )["total_cost_annual"]
                cost_dn_hi = calc_resilience_cost(
                    dataclasses.replace(setup, ccts_system=ccts_hi,
                                        strategy="do_nothing")
                )["total_cost_annual"]
                slope_s  = (cost_s_hi  - cost_s_lo)  / (_ets_hi - _ets_lo)
                slope_dn = (cost_dn_hi - cost_dn_lo) / (_ets_hi - _ets_lo)
                denom = slope_s - slope_dn
                if abs(denom) < 1e-10:
                    return np.nan  # costs are parallel — no crossing
                return float((cost_dn_lo - cost_s_lo) / denom)

            else:
                raise ValueError(f"Unknown output_metric: {output_metric!r}")

    except Exception as e:
        print(e)
        return np.nan


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def run_sobol_analysis(
    strategy: str = "truck",
    location: str = "onshore",
    output_metric: str = "cost_per_tonne",
    n_samples: int = 1024,
    seed: int = 42,
    n_workers: int | None = None,
    integration: bool = False,
    save_samples: bool = False,
) -> tuple[pd.DataFrame, dict, np.ndarray, pd.DataFrame]:
    """
    Run Sobol' global sensitivity analysis for one resilience strategy.

    Parameters
    ----------
    strategy      : resilience strategy key (e.g. "truck", "train", "barge", "storage")
    output_metric : "cost_per_tonne" | "resilience_premium" | "breakeven_ets"
    n_samples             : base sample size; total evaluations = N × (2k + 2)
    seed          : RNG seed for reproducibility
    n_workers     : parallel processes (None → all available CPUs)
    integration   : if True, sets liqu_capex_factor=0.35 and transport_ownership="third_party"

    Returns
    -------
    df         : DataFrame with S1, S1_conf, ST, ST_conf indexed by parameter name
    Si         : raw SALib analysis dict
    Y          : 1-D array of all model output values (length N × (k + 2));
                 use for uncertainty quantification, e.g. np.percentile(Y, [5, 50, 95])
    samples_df : DataFrame with integer index, one column per parameter, and a final
                 column named after output_metric containing the corresponding Y value
    """
    transport_cost_data = load_transport_cost_data()
    problem = _get_problem(output_metric)

    k = problem["num_vars"]
    n_evals = n_samples * (k + 2)  # calc_second_order=False → N × (k + 2)
    print(f"\n[{strategy}|{output_metric}] Sampling N={n_samples} → {n_evals:,} evaluations …")

    param_values = sobol_sample(problem, n_samples, calc_second_order=False, seed=seed)
    tasks = [(row, strategy, output_metric, integration, location) for row in param_values]

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(transport_cost_data,),
    ) as pool:
        Y = np.array(list(pool.map(_evaluate_row, tasks, chunksize=256)))

    n_nan = int(np.isnan(Y).sum())
    if n_nan > 0:
        print(f"  Warning: {n_nan} NaN evaluations replaced with median.")
        Y = np.where(np.isnan(Y), float(np.nanmedian(Y)), Y)

    Si = sobol_analyze(problem, Y, calc_second_order=False, seed=seed,
                       print_to_console=False)

    df = pd.DataFrame(
        {
            "S1":      Si["S1"],
            "S1_conf": Si["S1_conf"],
            "ST":      Si["ST"],
            "ST_conf": Si["ST_conf"],
        },
        index=problem["names"],
    )

    # Convergence check
    with_nonzero_ST = df["ST"].abs() > 1e-6
    wide = with_nonzero_ST & (df["ST_conf"] > 0.1 * df["ST"].abs())
    if wide.any():
        print(f"  Convergence warning: {wide.sum()} indices have CI > 10% of |ST|.")
        print("  Consider increasing N.")
    else:
        print("  Convergence OK: all CIs ≤ 10% of |ST|.")

    samples_df = pd.DataFrame(param_values, columns=problem["names"])
    samples_df[output_metric] = Y

    if save_samples:
        label = "int" if integration else "so"
        out = _SAMPLES_DIR / output_metric / f"samples_{strategy}_{label}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        samples_df.to_csv(out, index=False)
        print(f"  Saved samples → {out}")

    return df, Si, Y, samples_df


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_sobol_indices(
    df: pd.DataFrame,
    strategy: str,
    output_path: Path | None = None,
) -> plt.Figure:
    """Horizontal bar chart of ST (total-effect) and S1 (first-order) indices."""
    sorted_df = df.sort_values("ST", ascending=True)
    labels = [PARAM_LABELS.get(n, n) for n in sorted_df.index]
    y_pos = np.arange(len(sorted_df))

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.barh(
        y_pos, sorted_df["ST"],
        xerr=sorted_df["ST_conf"],
        color="steelblue", alpha=0.85,
        capsize=4, error_kw={"linewidth": 1.2},
        label=r"$S_T$ (total-effect)",
    )
    ax.barh(
        y_pos, sorted_df["S1"].clip(lower=0),
        color="darkorange", alpha=0.65,
        label=r"$S_1$ (first-order)",
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Sobol' index")
    ax.set_title(
        f"Global Sensitivity Analysis — strategy: {strategy}\n"
        r"$S_T$ (blue) and $S_1$ (orange); error bars = 95 % CI"
    )
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {output_path}")

    return fig


def plot_all_strategies_comparison(
    results: dict[str, pd.DataFrame],
    output_path: Path | None = None,
) -> plt.Figure:
    """
    Side-by-side ST heatmap across all analysed strategies.
    Rows = parameters, columns = strategies; colour = ST value.
    """
    all_names = list(next(iter(results.values())).index)
    st_df = pd.DataFrame(
        {strategy: df["ST"] for strategy, df in results.items()},
        index=all_names,
    )
    # Sort rows by mean ST across strategies
    st_df = st_df.loc[st_df.mean(axis=1).sort_values(ascending=False).index]
    labels = [PARAM_LABELS.get(n, n) for n in st_df.index]

    fig, ax = plt.subplots(figsize=(max(6, len(results) * 1.5 + 3), 7))
    im = ax.imshow(st_df.values, aspect="auto", cmap="YlOrRd", vmin=0)

    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(list(results.keys()), fontsize=10)
    ax.set_yticks(range(len(st_df)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title(r"Total-effect Sobol' indices $S_T$ across strategies")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(r"$S_T$")

    # Annotate cells
    for i in range(len(st_df)):
        for j in range(len(results)):
            val = st_df.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if val < 0.5 else "white")

    plt.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {output_path}")
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(
    location: str = "onshore",
    output_metric: str = "cost_per_tonne",
    n_samples: int = 1024,
    seed: int = 42,
    n_workers: int | None = None,
    save_figures: bool = True,
    save_samples: bool = False,
    show: bool = True,
    integration: bool = False,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """
    Run Sobol' analysis and plot results.

    Parameters
    ----------
    location      : "onshore" or "offshore" — selects the strategy set via
                    LOCATION_STRATEGIES
    output_metric : "cost_per_tonne" | "resilience_premium" | "breakeven_ets"
    n_samples     : base sample size (total evals = N × (k + 2))
    seed          : RNG seed
    n_workers     : parallel processes (None → all CPUs)
    save_figures  : write PNGs to figures/
    save_samples  : write sample CSVs to figures/<output_metric>/
    show          : call plt.show() at the end and print dataframe
    integration   : if True, use integrated liquefaction (liqu_capex_factor = 0.35)

    Returns
    -------
    results        : dict mapping strategy name → Sobol' index DataFrame (S1, ST, conf intervals)
    sample_results : dict mapping strategy name → labeled samples DataFrame (params + output_metric)
    """
    strategies = LOCATION_STRATEGIES[location]

    results: dict[str, pd.DataFrame] = {}
    sample_results = {}

    for strategy in strategies:
        df, _, _Y, samples_df = run_sobol_analysis(
            strategy=strategy, location=location, output_metric=output_metric,
            n_samples=n_samples, seed=seed, n_workers=n_workers,
            integration=integration, save_samples=save_samples,
        )
        results[strategy] = df
        sample_results[strategy] = samples_df

        if show:
            print(f"\n  Sobol' indices — {strategy} | {output_metric}:")
            print(df.to_string(float_format=lambda x: f"{x:+.4f}"))

        fig = plot_sobol_indices(
            df, strategy=strategy,
            output_path=FIGURES_DIR / output_metric / f"sobol_{strategy}.png"
            if save_figures else None,
        )
        if show:
            fig.show()
        plt.close(fig)

    if len(results) > 1:
        fig = plot_all_strategies_comparison(
            results,
            output_path=FIGURES_DIR / output_metric / "sobol_comparison.png"
            if save_figures else None,
        )
        if show:
            fig.show()
        plt.close(fig)

    if show:
        plt.show(block=True)

    return results, sample_results


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------

def load_sobol_samples(
    strategies: list[str],
    integration: bool,
    output_metric: str = "breakeven_ets",
    figures_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Load previously saved samples DataFrames for the given strategies.

    Files are expected at ``figures_dir/<output_metric>/samples_<strategy>_<label>.csv``
    where ``label`` is ``"int"`` for integrated and ``"so"`` for self-owned.

    Raises FileNotFoundError if any file is missing — run sobol_analysis.py
    with save_samples=True first.
    """
    if figures_dir is None:
        figures_dir = _SAMPLES_DIR
    label = "int" if integration else "so"
    result: dict[str, pd.DataFrame] = {}
    for strategy in strategies:
        path = figures_dir / output_metric / f"samples_{strategy}_{label}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Samples not found: {path}\n"
                "Run sobol_analysis.py with save_samples=True first."
            )
        result[strategy] = pd.read_csv(path)
        print(f"  Loaded {len(result[strategy])} samples: {path.name}")
    return result


def compute_breakeven_bands(
    strategy_samples: dict[str, pd.DataFrame],
    x_vals: np.ndarray,
    fixed_distance_km: float,
    tolerance_km: float = 50.0,
    percentiles: tuple[float, float] = (10.0, 90.0),
    min_samples_warn: int = 30,
    failure_duration_days: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute lower/upper percentile bands for the minimum breakeven ETS price.

    For each x-axis bin (``emissions_tonne_per_hr``), the band spans the
    ``percentiles[0]``-th to ``percentiles[1]``-th percentile of the
    element-wise minimum breakeven ETS across all supplied strategies.

    All DataFrames must have been produced by ``run_sobol_analysis`` with
    ``output_metric="breakeven_ets"`` and **the same seed**, so that row *i*
    in every DataFrame corresponds to an identical parameter draw.  The
    minimum across strategies is then well-defined sample-by-sample.

    The pipeline distance is treated as a nuisance variable: only samples
    where ``distance_km`` falls within ``fixed_distance_km ± tolerance_km``
    are used.  The resulting band therefore quantifies uncertainty from all
    remaining parameters (cost params, interest rate, depreciation, backup
    factors) at approximately the distance used in the deterministic line.

    Parameters
    ----------
    strategy_samples  : ``{strategy: samples_df}`` from ``run_sobol_analysis``
    x_vals            : x-axis grid (emissions_tonne_per_hr) for the plot;
                        returned arrays are aligned to this grid.
    fixed_distance_km : pipeline distance used in the deterministic plot line
    tolerance_km      : ±window around ``fixed_distance_km``
    percentiles       : ``(lower_pct, upper_pct)`` for the shaded band
    min_samples_warn  : warn when a bin has fewer samples than this threshold
    failure_duration_days : when provided, validated against
                        ``FAILURE_DURATION_BREAKEVEN``; raises ``ValueError``
                        if they differ so that band and deterministic line are
                        guaranteed to reflect the same failure scenario.

    Returns
    -------
    lower, upper : ``np.ndarray`` of shape ``(len(x_vals),)``; NaN where data
                   are insufficient to compute the requested percentiles.

    Notes
    -----
    # TODO: consider kernel-density smoothing as an alternative to hard binning
    # when sample counts per bin are small.
    """
    if not strategy_samples:
        raise ValueError("strategy_samples must not be empty.")
    if failure_duration_days is not None and failure_duration_days != FAILURE_DURATION_BREAKEVEN:
        raise ValueError(
            f"failure_duration_days={failure_duration_days} does not match "
            f"FAILURE_DURATION_BREAKEVEN={FAILURE_DURATION_BREAKEVEN}. "
            "Re-run sobol_analysis.py after updating the constant."
        )

    ref_df = next(iter(strategy_samples.values()))
    param_cols = [c for c in ref_df.columns if c != "breakeven_ets"]
    combined = ref_df[param_cols].copy()

    y_stack = np.stack(
        [df["breakeven_ets"].values for df in strategy_samples.values()], axis=0
    )
    combined["y_min"] = np.nanmin(y_stack, axis=0)

    x_arr = np.asarray(x_vals, dtype=float)
    edges = np.empty(len(x_arr) + 1)
    edges[0]    = x_arr[0]  - (x_arr[1]  - x_arr[0])  / 2.0
    edges[-1]   = x_arr[-1] + (x_arr[-1] - x_arr[-2]) / 2.0
    edges[1:-1] = (x_arr[:-1] + x_arr[1:]) / 2.0

    lower = np.full(len(x_vals), np.nan)
    upper = np.full(len(x_vals), np.nan)
    sparse_bins: list[int] = []

    for i, (lo_e, hi_e) in enumerate(zip(edges[:-1], edges[1:])):
        bin_mask = (
            (combined["emissions_tonne_per_hr"] >= lo_e) &
            (combined["emissions_tonne_per_hr"] <  hi_e)
        )
        vals = combined.loc[bin_mask, "y_min"].dropna().values
        n = len(vals)
        if n < min_samples_warn:
            sparse_bins.append(i)
        if n >= 2:
            lower[i] = np.percentile(vals, percentiles[0])
            upper[i] = np.percentile(vals, percentiles[1])

    if sparse_bins:
        print(
            f"  Warning: {len(sparse_bins)} x-bins have fewer than "
            f"{min_samples_warn} samples. Consider increasing N."
        )

    return lower, upper


if __name__ == "__main__":
    start = pd.Timestamp.now()

    # ── Sobol' comparison heatmap for resilience_premium ─────────────────
    _rp_combined: dict[str, pd.DataFrame] = {}
    for _loc in ["onshore", "offshore"]:
        _rp_results, _ = main(
            location=_loc,
            output_metric="resilience_premium",
            n_samples=2**13,
            seed=42,
            n_workers=None,
            save_figures=False,
            save_samples=False,
            show=False,
            integration=False,
        )
        for _s, _df in _rp_results.items():
            if _s not in _rp_combined:
                _rp_combined[_s] = _df

    if _rp_combined:
        # _fig = plot_all_strategies_comparison(
        #     _rp_combined,
        #     output_path=_FIGURES_DIR / "resilience_premium" / "sobol_comparison_S1.png",
        # )
        # plt.close(_fig)

        st_table = pd.DataFrame(
            {TABLE_COL_LABELS.get(s, s): df["ST"] for s, df in _rp_combined.items()},
        )
        st_table.index = [TABLE_ROW_LABELS.get(n, n) for n in st_table.index]
        st_table = st_table.loc[
            st_table.mean(axis=1).sort_values(ascending=False).index
        ]
        st_table.index.name = "parameter"
        _csv_path = FIGURES_DIR / "resilience_premium" / "sobol_comparison.csv"
        _csv_path.parent.mkdir(parents=True, exist_ok=True)
        st_table.to_csv(_csv_path, float_format="%.4f")
        print(f"  Saved Sobol table → {_csv_path}")

    end = pd.Timestamp.now()
    print(f"\nTotal runtime: {end - start}")
