"""
CO2 Pipeline Transport Resilience Cost Model
=============================================
Models two complementary cost perspectives:

  1. Normal-operation chain cost  (compute_normal_operation_cost)
     The annualised cost of running the full capture → compress → transport →
     store chain when everything is working as intended — no failures.
     Used as a break-even benchmark against the EU ETS price.

  2. Resilience cost  (compute_resilience_cost)
     The additional annualised cost incurred by a chosen resilience strategy
     to mitigate the consequences of primary pipeline failures.

Resilience strategies
---------------------
  do_nothing      : Accept failure; unhandled CO2 is vented and penalised at ETS price
  truck           : Road-tanker backup transport
  train           : Train-tanker backup transport
  barge           : Inland/coastal barge backup transport (Sievert et al. CapEx)
  ship            : Ocean ship backup transport (Sievert et al. CapEx)
  storage         : Buffer tank only; residual vented and penalised

Cost components (all independent, all in €/yr)
-----------------------------------------------
  capex_annualised      CapEx annualised via Capital Recovery Factor (CRF)
  opex_backup           Backup transport opex during expected failure events
  opex_buffer_storage          Storage facility standing opex
  ets_penalty_cost      ETS cost for CO2 vented due to unhandled pipeline failures

"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np

from cost_calc import (
    capex_barge,
    capex_ship,
    capex_train,
    capex_truck,
    liqu_lp_capex,
    liqu_lp_opex,
    liqu_mp_capex,
    liqu_mp_opex,
    opex_barge,
    opex_ship,
    opex_train,
    opex_truck,
    storage_capex,
    storage_opex,
)

# ── Utility ───────────────────────────────────────────────────────────────────


def annualise(capex: float, interest_rate: float, depreciation_years: float) -> float:
    """Annualise a capital cost via the Capital Recovery Factor (CRF)."""
    i, n = interest_rate, depreciation_years
    return capex * i * (1 + i) ** n / ((1 + i) ** n - 1)


def gwi_liqu_and_buffer(setup, capacity_tpa, flow, buffer_vol) -> float:
    """calculates GWI of liquefaction and buffer storage for given setup."""
    gwi_fix = (setup.gwi_fix_liquef * capacity_tpa +
               setup.gwi_fix_buffer_storage * buffer_vol)
    gwi_var = (setup.gwi_var_liquef * flow +
               setup.gwi_var_buffer_storage * buffer_vol)
    return gwi_fix + gwi_var


# ── Type alias ────────────────────────────────────────────────────────────────
Strategy = Literal[
    "do_nothing",
    "truck",
    "train",
    "barge",
    "ship",
    "storage",
]

ALL_STRATEGIES: list[Strategy] = [
    "do_nothing",
    "truck",
    "train",
    "barge",
    "ship",
    "storage",
]

STRATEGY_LABELS: dict[str, str] = {
    "normal_chain": "Normal Chain\n(no failure)",
    "do_nothing": "CO$_2$ Release",
    "truck": "Truck",
    "train": "Train",
    "barge": "Barge",
    "ship": "Ship",
    "storage": "Buffer storage",
}

# CapEx and OpEx functions from Sievert et al. (2025) keyed by strategy base name.
_SIEVERT_CAPEX: dict[str, Callable[[float, float], float]] = {
    "truck": capex_truck,
    "train": capex_train,
    "barge": capex_barge,
    "ship": capex_ship,
}
_SIEVERT_OPEX: dict[str, Callable[[float, float, float], float]] = {
    "truck": opex_truck,
    "train": opex_train,
    "barge": opex_barge,
    "ship": opex_ship,
}


@dataclass
class CostComponents:
    """Cost components for all cost calculations to reduce complexity."""
    capex_capture_annual: float = 0
    opex_capture_annual: float = 0
    capex_compr_annual: float = 0
    opex_compr_annual: float = 0
    cost_pipeline_annual: float = 0
    capex_backup_trsp_annual: float = 0
    opex_backup_trsp_annual: float = 0
    capex_liquef_annual: float = 0
    opex_liquef_annual: float = 0
    capex_buffer_stge_annual: float = 0
    opex_buffer_stge_annual: float = 0
    cost_geol_stge_annual: float = 0
    ets_penalty_annual: float = 0

    def __iter__(self):
        return (getattr(self, f.name) for f in dataclasses.fields(self))

    @property
    def total_annual_cost(self) -> float:
        """Returns the sum of all cost components."""
        return sum([
            self.capex_capture_annual, self.opex_capture_annual,
            self.capex_compr_annual, self.opex_compr_annual,
            self.cost_pipeline_annual, self.capex_backup_trsp_annual,
            self.opex_backup_trsp_annual, self.capex_liquef_annual,
            self.opex_liquef_annual, self.capex_buffer_stge_annual,
            self.opex_buffer_stge_annual, self.cost_geol_stge_annual,
            self.ets_penalty_annual
        ])

    @property
    def total_capex(self) -> float:
        """Dynamically sums only the CAPEX-related fields."""
        return sum([
            self.capex_capture_annual,
            self.capex_compr_annual,
            self.capex_backup_trsp_annual,
            self.capex_buffer_stge_annual
        ])


# ── Parameter dataclasses ──────────────────────────────────────────────────────

@dataclass
class PipelineParams:
    """Primary CO2 pipeline parameters."""
    failure_probability: float = 0.05  # annual probability [-], currently unused
    failure_duration_days: int = 26
    gwi_fix_pipeline: float = 0  # [kg / tCO2] fixed GWI of pipeline
    gwi_var_pipeline: float = 0.0051  # [kg / (tCO2 km)] variable GWI per t-km

    def calc_gwi(self, capacity, flow, distance_km) -> float:
        """Calculate the GWI of pipeline transport for given distance and throughput."""
        return (self.gwi_fix_pipeline * capacity
                + self.gwi_var_pipeline * flow) * distance_km


@dataclass
class BackupTransportParams:
    """Physical parameters for a single backup transport mode."""

    name: str = "Truck"
    capacity_tonne_per_hr: float = 1e5  # [t/hr]
    lead_time_days: float = 0.0  # [days]
    opex_factor: float = 1.0  # multiplicative scaling factor applied to OpEx [-]
    capex_factor: float = 1.0  # multiplicative scaling factor applied to CapEx [-]
    gwi_fix_transport: float = 0.02  # [kg / tCO2] fixed GWI of backup transport
    gwi_var_transport: float = 0.05  # [kg / (tCO2 km)] variable GWI per t-km

    def calc_gwi(self, capacity, flow, distance_km) -> float:
        """Calculate the GWI of backup transport for given distance and throughput."""
        return (self.gwi_fix_transport * capacity
                + self.gwi_var_transport * flow) * distance_km

    @classmethod
    def truck_default(cls) -> "BackupTransportParams":
        return cls(name="Truck", capacity_tonne_per_hr=1e5, lead_time_days=0.0)

    @classmethod
    def train_default(cls) -> "BackupTransportParams":
        return cls(name="Train", capacity_tonne_per_hr=1e5, lead_time_days=0.0)

    @classmethod
    def barge_default(cls) -> "BackupTransportParams":
        return cls(name="Barge", capacity_tonne_per_hr=1e5, lead_time_days=0.0)

    @classmethod
    def ship_default(cls) -> "BackupTransportParams":
        return cls(name="Ship", capacity_tonne_per_hr=1e5, lead_time_days=0.0)


@dataclass
class CCTSSystemParams:
    """
    System-level parameters for the CO2 Capture, Transport and Storage (CCTS) chain.
    """
    distance_km: float = 300.0  # pipeline route length [km]
    emissions_tonne_per_hr: float = 50.0  # CO2 emissions at the capture plant [t/hr]
    ets_price: float = 85.0  # EU ETS carbon price [€/t CO2]
    interest_rate: float = 0.05  # for CapEx annualisation (CRF)
    # Capture
    capex_capture: float = 7098996  # [€ / (t CO2/hr)] capex for capture
    opex_capture: float = 12.57  # [€/t CO2]  operating cost
    capture_rate: float = 1.0  # [-] capture rate (0-1)
    gwi_fix_capture: float = 0  # [kg / tCO2] fixed GWI from capture
    gwi_var_capture: float = 29.65  # [kg / tCO2] variable GWI per tonne captured
    # Compression
    capex_compression: float = 158426  # [€ / (t CO2/hr)] capex for compression
    opex_compression: float = 15.4  # [€/t CO2]  operating cost, excl. energy
    gwi_fix_compression: float = 0  # [kg / tCO2] fixed GWI from compression
    gwi_var_compression: float = 11.27  # [kg / tCO2] variable GWI per tonne captured
    # Storage (assumed third-party provider: no capex, fee per tonne only)
    opex_geol_storage: float = 40.0  # [€/t CO2]  storage fee
    gwi_fix_geol_storage: float = 0  # [kg / tCO2] fixed GWI from geol. storage
    gwi_var_geol_storage: float = 3.52  # [kg / tCO2] variable GWI per tonne captured
    # Pipeline routing
    pipeline_location: str = "onshore"  # "onshore" or "offshore"

    @property
    def captured_tonne_per_hr(self) -> float:
        return self.emissions_tonne_per_hr * self.capture_rate


@dataclass
class SystemSetup:
    """Complete system configuration passed to the cost model. """
    strategy: Strategy | str = "do_nothing"
    transport_ownership: str = "self_owned"  # "third_party" or "self_owned"
    ccts_system: CCTSSystemParams = field(default_factory=CCTSSystemParams)
    pipeline: PipelineParams = field(default_factory=PipelineParams)
    truck: BackupTransportParams = field(
        default_factory=BackupTransportParams.truck_default
    )
    train: BackupTransportParams = field(
        default_factory=BackupTransportParams.train_default
    )
    barge: BackupTransportParams = field(
        default_factory=BackupTransportParams.barge_default
    )
    ship: BackupTransportParams = field(
        default_factory=BackupTransportParams.ship_default
    )
    depreciation_years: float = 20.0  # CapEx amortisation period [years]
    transport_cost_data: dict[str, dict[str, float]] = field(default_factory=dict)
    liqu_capex_factor: float = 1.0  # ratio to reduce liqu capex due to integration
    buffer_capex_factor: float = 1.0  # multiplicative scaling factor for storage_capex
    buffer_opex_factor: float = 1.0   # multiplicative scaling factor for storage_opex
    cond_capex_factor: float = 1.0    # multiplicative scaling factor for liqu_*_capex
    cond_opex_factor: float = 1.0     # multiplicative scaling factor for liqu_*_opex
    gwi_fix_buffer_storage: float = 5.42   # [kg/t], from buffer_storage.csv
    gwi_var_buffer_storage: float = 0.0    # [kg/t], from buffer_storage.csv
    gwi_fix_liquef: float = 0.184          # [kg/t], from liquefaction.csv
    gwi_var_liquef: float = 44.6           # [kg/t], from liquefaction.csv

    # ── Derived quantities that cross dataclass boundaries ─────────────────

    @property
    def pipe_trsp_tonne_per_hr(self) -> float:
        """CO2 transported per hour [t/hr]. Pipeline is sized to capture."""
        return self.ccts_system.captured_tonne_per_hr

    @property
    def uncaptured_tonne_per_hr(self) -> float:
        """CO2 that is lost on site due to capture rate and losses in compression."""
        return self.ccts_system.emissions_tonne_per_hr - self.pipe_trsp_tonne_per_hr

    @property
    def co2_volume_per_failure(self) -> float:
        """CO2 that cannot be transported during one full failure event [t]."""
        return self.pipe_trsp_tonne_per_hr * self.pipeline.failure_duration_days * 24.0

    def annualise_capex(self, capex: float) -> float:
        """Annualise capex using this setup's interest rate and depreciation period."""
        return annualise(capex, self.ccts_system.interest_rate, self.depreciation_years)

    def calc_unitary_transport_cost(self, mode: str = "truck") -> float:
        """Calculate the unitary transport cost [€/t/km] for a transport mode.
        Parameters
        ----------
        mode:
            Key into self.transport_cost_data selecting the transport cost curve.
        """
        if not self.transport_cost_data:
            raise ValueError("transport_cost_data is empty. Use: load_system_setup().")

        params = self.transport_cost_data[mode]
        ratio_d = self.ccts_system.distance_km
        ratio_m = self.ccts_system.captured_tonne_per_hr * 8760  # convert to t/a

        if params["eq"] == 1:  # Eq. 39 & 41 from Oeuvray et al. 2024
            term1 = params["a1"]
            term2 = params["a2"] * (ratio_d ** params["a3"]) * (ratio_m ** params["a4"])
        elif params["eq"] == 2:  # Eq. 40 from Oeuvray et al. 2024
            term1 = params["a1"] + params["a2"] / ratio_d
            term2 = params["a3"] / ratio_m
        else:
            raise ValueError(f"Unsupported equation ID: {params['eq']}")

        return (term1 + term2) * 1.207  # inflation adjustment to 2025


# ── Core cost function ────────────────────────────────────────────────────────

def calc_normal_operation_cost(setup: SystemSetup) -> dict:
    """
    Compute the annualised cost of the full CCTS chain during **normal operation**.

    This covers every part of the chain:
        Capture → Compression → Pipeline transport → Geological storage

    The result is independent of the chosen resilience strategy. Capex is annualised
     via the Capital Recovery Factor (CRF). OpEx assumes 8760 operating hours.

    Parameters
    ----------
    setup:
        SystemSetup class with details and costs of the system. BackupTransportParams
        and related resilience parameters are not used.

    Returns
    -------
    dict with components of the regular operation chain. Annual costs in €/yr,
    cost per tonne in €/t.
    """
    ccts = setup.ccts_system

    # ── CapEx components ───────────────────────────────────────────────────
    # Capture Capex is sized on plant emission volume
    capex_capture_annual = setup.annualise_capex(
        ccts.capex_capture * ccts.emissions_tonne_per_hr
    )
    # Compression CapEx is sized on captured volume.
    capex_compr_annual = setup.annualise_capex(
        ccts.capex_compression * ccts.captured_tonne_per_hr
    )
    # Transport capex based on a pipeline sized to the captured amount
    pipeline_key = f"pipeline_dense_{ccts.pipeline_location}"
    cost_pipeline_annual = setup.calc_unitary_transport_cost(pipeline_key)
    cost_pipeline_annual *= setup.pipe_trsp_tonne_per_hr * ccts.distance_km * 8760

    # ── OpEx components ────────────────────────────────────────────────────
    # OpEx scales with actual annual throughput per supply chain part.
    opex_capture_annual = ccts.opex_capture * ccts.captured_tonne_per_hr * 8760
    opex_compr_annual = ccts.opex_compression * setup.pipe_trsp_tonne_per_hr * 8760
    cost_geol_stge_annual = ccts.opex_geol_storage * setup.pipe_trsp_tonne_per_hr * 8760

    # ── Totals ─────────────────────────────────────────────────────────────
    cost_components = CostComponents(
        capex_capture_annual=capex_capture_annual,
        opex_capture_annual=opex_capture_annual,
        capex_compr_annual=capex_compr_annual,
        opex_compr_annual=opex_compr_annual,
        cost_pipeline_annual=cost_pipeline_annual,
        cost_geol_stge_annual=cost_geol_stge_annual,
    )
    cost_normal_operation_per_tonne = (
        cost_components.total_annual_cost / (setup.pipe_trsp_tonne_per_hr * 8760)
        if setup.pipe_trsp_tonne_per_hr > 0
        else 0.0
    )

    return dict(
        cost_components=cost_components,
        cost_per_tonne=cost_normal_operation_per_tonne,
        total_cost_annual=cost_components.total_annual_cost,
    )


def calc_resilience_cost(setup: SystemSetup) -> dict:
    """
    Compute the annualised cost of the chosen resilience strategy.

    All cost components are independent and returned separately in a dict
    so callers can inspect or sum any subset.
    """
    pipeline = setup.pipeline
    ccts = setup.ccts_system
    strategy = setup.strategy

    vol_per_failure = setup.co2_volume_per_failure  # [t] per failure event
    normal_chain = calc_normal_operation_cost(setup)
    normal_chain_costs = normal_chain["cost_components"]

    co2_tpa = setup.pipe_trsp_tonne_per_hr * 8760  # [tCO₂/yr] pipeline capacity
    dist = ccts.distance_km

    # ── Helper: cost for a pure backup-transport strategy ─────────────────
    def backup_transport_cost(backup_transport: BackupTransportParams) -> dict:
        """Cost for a backup transport mode without storage."""
        mode_key = backup_transport.name.lower()

        # Volume that cannot be covered during the backup activation lead-time
        lead_time_days = min(
            backup_transport.lead_time_days, pipeline.failure_duration_days
        )
        lead_time_volume = setup.pipe_trsp_tonne_per_hr * lead_time_days * 24.0
        active_days = max(
            pipeline.failure_duration_days - backup_transport.lead_time_days,
            0.0,
        )
        # Volume covered once backup is active
        max_vol_coverable = vol_per_failure - lead_time_volume
        covered_volume = min(
            backup_transport.capacity_tonne_per_hr * active_days * 24.0,
            max_vol_coverable,
        )
        # Volume that is not handled due to lead time and capacity limits. Can be zero.
        unhandled_vol_per_failure = vol_per_failure - covered_volume
        if unhandled_vol_per_failure != 0:
            print('Warning: Unhandled volume per failure is !=0. Check backup cost!')

        # liquefaction is always owned by the emitter since it is on site.
        if backup_transport.name == "Ship":
            capex_liquef_annual = setup.annualise_capex(
                liqu_lp_capex(co2_tpa) * setup.liqu_capex_factor * setup.cond_capex_factor)
            opex_liquef_annual = liqu_lp_opex(co2_tpa, covered_volume) * setup.cond_opex_factor
        else:
            capex_liquef_annual = setup.annualise_capex(
                liqu_mp_capex(co2_tpa) * setup.liqu_capex_factor * setup.cond_capex_factor)
            opex_liquef_annual = liqu_mp_opex(co2_tpa, covered_volume) * setup.cond_opex_factor

        if setup.transport_ownership == "self_owned":
            buffer_days = 1
            capex_backup_transport_annual = setup.annualise_capex(
                _SIEVERT_CAPEX[mode_key](co2_tpa, dist) * backup_transport.capex_factor
            )
            # Opex computed at actual covered_volume — exact for all modes.
            opex_backup_transport_annual = (
                    _SIEVERT_OPEX[mode_key](co2_tpa, covered_volume, dist)
                    * backup_transport.opex_factor
            )
        elif setup.transport_ownership == "third_party":
            buffer_days = 1
            capex_backup_transport_annual = 0.0
            opex_backup_transport_annual = (
                    setup.calc_unitary_transport_cost(mode_key)
                    * covered_volume * dist
                    * backup_transport.opex_factor
            )
        else:
            raise ValueError(f"Unknown transport ownership: "
                             f"{setup.transport_ownership!r}")

        capex_buffer_stge_annual = setup.annualise_capex(
            storage_capex(co2_tpa, buffer_days) * setup.buffer_capex_factor)
        opex_buffer_stge_annual = storage_opex(co2_tpa, buffer_days) * setup.buffer_opex_factor
        # calculate the opex of the infrastructure from normal operation. Depending on
        # the strategy and the covered volume, the cost varies.
        unhandled_vol_fraction = unhandled_vol_per_failure / (
                setup.pipe_trsp_tonne_per_hr * 8760
        )
        # assert that unhandled_vol_fraction is between 0 and 1, otherwise there's a bug
        # in the logic above. Use an assertion with a clear error message.
        assert 0.0 <= unhandled_vol_fraction <= 1.0, (
            f"Unhandled fraction {unhandled_vol_fraction:.2f} is out of bounds. "
            f"Check the backup transport logic for strategy {strategy!r}."
        )

        # Reduce opex according to CO2 which is not captured / stored. Capture etc.
        # continues during backup transport but compresssion & pipeline does not.
        uncaptured = 1 - unhandled_vol_fraction
        uncompressed = 1 - pipeline.failure_duration_days / 365
        opex_capture_annual = normal_chain_costs.opex_capture_annual * uncaptured
        opex_compr_annual = normal_chain_costs.opex_compr_annual * uncompressed
        cost_pipeline_annual = normal_chain_costs.cost_pipeline_annual * uncompressed
        cost_geol_stge_annual = normal_chain_costs.cost_geol_stge_annual * uncaptured

        ets_penalty = unhandled_vol_per_failure * ccts.ets_price

        cost_components = CostComponents(
            capex_capture_annual=normal_chain_costs.capex_capture_annual,
            opex_capture_annual=opex_capture_annual,
            capex_compr_annual=normal_chain_costs.capex_compr_annual,
            opex_compr_annual=opex_compr_annual,
            cost_pipeline_annual=cost_pipeline_annual,
            capex_backup_trsp_annual=capex_backup_transport_annual,
            opex_backup_trsp_annual=opex_backup_transport_annual,
            capex_buffer_stge_annual=capex_buffer_stge_annual,
            opex_buffer_stge_annual=opex_buffer_stge_annual,
            capex_liquef_annual=capex_liquef_annual,
            opex_liquef_annual=opex_liquef_annual,
            cost_geol_stge_annual=cost_geol_stge_annual,
            ets_penalty_annual=ets_penalty,
        )
        cost_per_tonne = (
            cost_components.total_annual_cost
            / (setup.pipe_trsp_tonne_per_hr * 8760 - unhandled_vol_per_failure)
            if setup.pipe_trsp_tonne_per_hr > 0
            else 0.0
        )
        return dict(
            strategy=strategy,
            cost_components=cost_components,
            unhandled_vol_per_failure=unhandled_vol_per_failure,
            cost_per_tonne=cost_per_tonne,
            total_cost_annual=cost_components.total_annual_cost,
        )

    # ── Helper: buffer storage only ───────────────────────────────────────
    def storage_cost() -> dict:
        """Cost for buffer storage only; residual volume vented and penalised.
        Assumption: The storage is emptied through the pipeline by slightly increasing
         the pressure inside the pipeline."""

        store_cap = co2_tpa * setup.pipeline.failure_duration_days / 365

        capex_buffer_stge_annual = setup.annualise_capex(
            storage_capex(co2_tpa, setup.pipeline.failure_duration_days) * setup.buffer_capex_factor)
        opex_buffer_stge_annual = (storage_opex(co2_tpa, setup.pipeline.failure_duration_days)
                                   * setup.buffer_opex_factor)

        covered_by_storage = min(store_cap, vol_per_failure)
        unhandled_vol_per_failure = vol_per_failure - covered_by_storage
        unhandled_vol_fraction = unhandled_vol_per_failure / (
                setup.pipe_trsp_tonne_per_hr * 8760
        )
        if np.round(unhandled_vol_per_failure, 5) != 0:
            print('Warning: Unhandled volume per failure is !=0. Check storage_cost!')
        # liquefaction is always owned by the emitter since it is on site.
        capex_liquef_annual = setup.annualise_capex(
            liqu_mp_capex(co2_tpa) * setup.liqu_capex_factor * setup.cond_capex_factor)
        opex_liquef_annual = liqu_mp_opex(co2_tpa, covered_by_storage) * setup.cond_opex_factor

        uncaptured = 1 - unhandled_vol_fraction
        uncompressed = 1 - pipeline.failure_duration_days / 365
        opex_capture_annual = normal_chain_costs.opex_capture_annual * uncaptured
        opex_compr_annual = normal_chain_costs.opex_compr_annual * uncompressed
        cost_pipeline_annual = normal_chain_costs.cost_pipeline_annual
        cost_geol_stge_annual = normal_chain_costs.cost_geol_stge_annual * uncaptured

        ets_penalty = unhandled_vol_per_failure * ccts.ets_price

        cost_components = CostComponents(
            capex_capture_annual=normal_chain_costs.capex_capture_annual,
            opex_capture_annual=opex_capture_annual,
            capex_compr_annual=normal_chain_costs.capex_compr_annual,
            opex_compr_annual=opex_compr_annual,
            cost_pipeline_annual=cost_pipeline_annual,
            capex_buffer_stge_annual=capex_buffer_stge_annual,
            opex_buffer_stge_annual=opex_buffer_stge_annual,
            capex_liquef_annual=capex_liquef_annual,
            opex_liquef_annual=opex_liquef_annual,
            cost_geol_stge_annual=cost_geol_stge_annual,
            ets_penalty_annual=ets_penalty,
        )
        cost_per_tonne = (
            cost_components.total_annual_cost
            / (setup.pipe_trsp_tonne_per_hr * 8760 - unhandled_vol_per_failure)
            if setup.pipe_trsp_tonne_per_hr > 0
            else 0.0
        )
        return dict(
            strategy=strategy,
            cost_components=cost_components,
            unhandled_vol_per_failure=unhandled_vol_per_failure,
            cost_per_tonne=cost_per_tonne,
            total_cost_annual=cost_components.total_annual_cost,
        )

    if strategy == "do_nothing":
        unhandled_vol_fraction = vol_per_failure / (setup.pipe_trsp_tonne_per_hr * 8760)
        # Reduce opex according to CO2 which is not captured / stored. Capture etc.
        # continues during backup transport but normal transport does not.
        opex_capture_annual = normal_chain_costs.opex_capture_annual * (
                1 - unhandled_vol_fraction
        )
        opex_compr_annual = normal_chain_costs.opex_compr_annual * (
                1 - unhandled_vol_fraction
        )
        cost_pipeline_annual = normal_chain_costs.cost_pipeline_annual * (
                1 - pipeline.failure_duration_days / 365
        )
        cost_geol_stge_annual = normal_chain_costs.cost_geol_stge_annual * (
                1 - unhandled_vol_fraction
        )
        cost_components = CostComponents(
            capex_capture_annual=normal_chain_costs.capex_capture_annual,
            opex_capture_annual=opex_capture_annual,
            capex_compr_annual=normal_chain_costs.capex_compr_annual,
            opex_compr_annual=opex_compr_annual,
            cost_pipeline_annual=cost_pipeline_annual,
            cost_geol_stge_annual=cost_geol_stge_annual,
            ets_penalty_annual=vol_per_failure * ccts.ets_price,
        )
        cost_per_tonne = cost_components.total_annual_cost / (
                setup.pipe_trsp_tonne_per_hr * 8760 - vol_per_failure
        )
        return dict(
            strategy=strategy,
            cost_components=cost_components,
            unhandled_vol_per_failure=vol_per_failure,
            cost_per_tonne=cost_per_tonne,
            total_cost_annual=cost_components.total_annual_cost,
        )

    elif strategy == "truck":
        return backup_transport_cost(setup.truck)
    elif strategy == "train":
        return backup_transport_cost(setup.train)
    elif strategy == "barge":
        return backup_transport_cost(setup.barge)
    elif strategy == "ship":
        return backup_transport_cost(setup.ship)
    elif strategy == "storage":
        return storage_cost()
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")


# ── Core GWI function ────────────────────────────────────────────────────────
def calc_gwi_for_strategy(setup: SystemSetup) -> tuple[float, float]:
    """Calculate the GWI for the entire system under the chosen strategy."""
    pipeline = setup.pipeline
    ccts = setup.ccts_system
    strategy = setup.strategy

    vol_per_failure = setup.co2_volume_per_failure
    co2_tpa = setup.pipe_trsp_tonne_per_hr * 8760
    dist = ccts.distance_km

    def normal_chain_gwi(unhandled_vol_fraction: float) -> float:
        """Normal-chain GWI scaled by how much of each stage still operates. """
        uncaptured = 1 - unhandled_vol_fraction
        uncompressed = 1 - pipeline.failure_duration_days / 365

        gwi_capture = (ccts.gwi_fix_capture +
                   ccts.gwi_var_capture * uncaptured) * co2_tpa
        gwi_compression = (ccts.gwi_fix_compression +
                           ccts.gwi_var_compression * uncompressed) * co2_tpa
        gwi_pipeline = pipeline.calc_gwi(co2_tpa, co2_tpa * uncompressed, dist)
        gwi_geol_storage = (ccts.gwi_fix_geol_storage +
                            ccts.gwi_var_geol_storage * uncaptured) * co2_tpa

        return gwi_capture + gwi_compression + gwi_pipeline + gwi_geol_storage

    def backup_transport_gwi(backup_transport: BackupTransportParams):
        """GWI for a backup transport mode"""
        covered_volume = min(
            backup_transport.capacity_tonne_per_hr *
            pipeline.failure_duration_days * 24.0,
            vol_per_failure,
        )
        unhandled_vol_per_failure = vol_per_failure - covered_volume
        unhandled_vol_fraction = unhandled_vol_per_failure / co2_tpa

        gwi_backup = backup_transport.calc_gwi(co2_tpa, covered_volume, dist)
        gwi_liqu_buffer = gwi_liqu_and_buffer(setup, co2_tpa, covered_volume,
                                              ccts.captured_tonne_per_hr * 24)
        gwi_total = (gwi_backup + gwi_liqu_buffer +
                     normal_chain_gwi(unhandled_vol_fraction)) / 1000
        return gwi_total, unhandled_vol_per_failure

    if strategy == "do_nothing":
        unhandled_vol_fraction = vol_per_failure / co2_tpa
        gwi_noc = normal_chain_gwi(unhandled_vol_fraction) / 1000
        return gwi_noc, vol_per_failure
    elif strategy == "truck":
        return backup_transport_gwi(setup.truck)
    elif strategy == "train":
        return backup_transport_gwi(setup.train)
    elif strategy == "barge":
        return backup_transport_gwi(setup.barge)
    elif strategy == "ship":
        return backup_transport_gwi(setup.ship)
    elif strategy == "storage":
        unhandled_vol_per_failure = 0  # assumption: storage always covers failure
        gwi_liqu_buffer = gwi_liqu_and_buffer(setup, co2_tpa, vol_per_failure,
                                              vol_per_failure)
        gwi_tonnes_per_year = (normal_chain_gwi(0) + gwi_liqu_buffer) / 1000
        return gwi_tonnes_per_year, unhandled_vol_per_failure
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")
