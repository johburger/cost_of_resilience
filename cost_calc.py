"""CAPEX and OPEX functions for CO₂ transport modes: truck, train, barge, and ship.

This module provides functions to calculate capital expenditure (CAPEX, €) and
annual operating expenditure (OPEX, €/yr) for four CO₂ transport modes as a
function of transported volume and distance.
The formulas are derived from the techno-economic model in:

    Sievert, K., Stefanescu, A. S., Oeuvray, P., & Steffen, B. (2025).
    The impact of financing structures on the cost of carbon dioxide transport.
    Energy Economics, 143, 108253. https://doi.org/10.1016/j.eneco.2025.108253

All cost parameters are sourced from the "Techno-economic" worksheet of the
Supplementary Data. If underlying assumptions change, update the corresponding value.

Units:
    capacity:    tCO₂/yr  — design throughput the infrastructure is sized for
    flow:        tCO₂     — actual CO₂ transported during a failure event
    distance:    km
    capex_*      return:  €
    opex_*       return:  (capacity_opex €, flow_opex €)
                   capacity_opex — standing costs that depend only on capacity
                   flow_opex     — costs that scale with actual flow
"""

import math

import numpy as np

hicp_index = {
    2015: 1.296,
    2016: 1.294,
    2017: 1.277,
    2018: 1.258,
    2019: 1.243,
    2020: 1.236,
    2021: 1.207,
    2022: 1.115,
    2023: 1.051,
    2024: 1.025,
    2025: 1.0,
}


def storage_capex(co2_tpa: float, days: int = 1) -> float:
    """Buffer storage CAPEX (€) shared across all transport modes."""
    ref_capacity = 1000  # tCO₂; TE!C172
    ref_capex = 2_000_000 * hicp_index[2022]  # €; TE!C173
    scaling_exp = 0.9  # TE!C174
    storage_capacity_tonne = co2_tpa * days / 365
    return ref_capex * (storage_capacity_tonne / ref_capacity) ** scaling_exp


def storage_opex(co2_tpa: float, days: int = 1) -> float:
    """Buffer storage OPEX (€/yr) shared across all transport modes."""
    OPEX_RATE = 0.06  # fraction of CAPEX/yr; TE!C177
    return storage_capex(co2_tpa, days) * OPEX_RATE


def liqu_lp_capex(co2_tpa):
    """Liquefaction CAPEX (€); LOW pressure: Liquefaction @ 7 barg, -50°C."""
    base_capacity = 1_000_000  # tCO₂/yr; TE!C161
    base_cost = 4.5864 * hicp_index[2022]  # €/t/yr × lifetime; TE!C162
    lifetime = 25  # yr; aligned with the rest of the model
    exponent = 0.85  # TE!C165

    rel_capacity = co2_tpa / base_capacity
    return base_cost * base_capacity * lifetime * rel_capacity ** exponent


def liqu_lp_opex(co2_tpa, flow):
    """Liquefaction OPEX (€/yr), LOW pressure."""
    energy_use = 96.3  # kWh/t; TE!C164
    efficiency = 0.9  # conditioning efficiency; TE!C168
    cooling_water = 0.5905 * hicp_index[2022]  # €/t; TE!C167
    fixed_opex_fraction = 0.06  # fraction of CAPEX/yr; TE!C163
    # Eurostat: 2015 - 2025 average prices from 20 000 MWh to 69 999 MWh - band IE:
    el_cost_per_kwh = 0.13530476  # €/kWh,

    energy_cost = flow * (energy_use / efficiency) * el_cost_per_kwh
    cooling_cost = cooling_water * flow
    om_cost = fixed_opex_fraction * liqu_lp_capex(co2_tpa)

    return om_cost + energy_cost + cooling_cost


def liqu_mp_capex(co2_tpa):
    """Liquefaction CAPEX (€); MEDIUM pressure: Liquefaction @ 15 barg, -30°C."""
    base_capacity = 1_000_000  # tCO₂/yr; TE!C161
    base_cost = 4.368 * hicp_index[2022]  # €/t/yr × lifetime; TE!C162
    lifetime = 25  # yr; aligned with the rest of the model
    exponent = 0.85  # TE!C165

    rel_capacity = co2_tpa / base_capacity
    return base_cost * base_capacity * lifetime * rel_capacity ** exponent


def liqu_mp_opex(co2_tpa, flow):
    """Liquefaction OPEX (€/yr), MEDIUM pressure."""
    energy_use = 90.4  # kWh/t; TE!C164
    efficiency = 0.9  # conditioning efficiency; TE!C168
    cooling_water = 0.7086 * hicp_index[2022]  # €/t; TE!C167
    fixed_opex_fraction = 0.06  # fraction of CAPEX/yr; TE!C163
    el_cost_per_kwh = 0.13530476  # €/kWh, from Eurostat 2015 - 2025 prices

    energy_cost = flow * (energy_use / efficiency) * el_cost_per_kwh
    cooling_cost = cooling_water * flow
    om_cost = fixed_opex_fraction * liqu_mp_capex(co2_tpa)

    return om_cost + energy_cost + cooling_cost


# ---------------------------------------------------------------------------
# Truck
# ---------------------------------------------------------------------------
def _truck_loading_stations(capacity):
    """Number of loading stations; shared by capex_truck and opex_truck.
    Based on Oeuvray et al. 2024"""
    truck_capacity = 26  # tCO₂
    operating_hours = 8500  # h/yr
    loading_time_per_truck = 1.75  # h; sum of loading and unloading
    shipments_per_year = np.ceil(capacity / truck_capacity)
    return np.ceil(shipments_per_year * loading_time_per_truck / operating_hours)


def _truck_fleet_size(capacity, distance):
    """Number of trucks needed; shared by capex_truck and opex_truck.
    Based on Oeuvray et al. 2024."""
    truck_capacity = 26  # tCO₂
    truck_speed = 80  # km/h; Shoman et al. 2023
    loading_time_per_truck = 1.75  # h; sum of loading and unloading
    operating_hours = 8500  # h/yr

    roundtrip_duration = loading_time_per_truck + (distance / truck_speed) * 2  # h
    shipments_per_year = np.ceil(capacity / truck_capacity)
    trucks_needed = np.ceil(shipments_per_year * roundtrip_duration / operating_hours)
    return trucks_needed


def capex_truck(capacity, distance):
    """Truck CAPEX (€);
    From Oeuvray et al. 2024 and generalized to fit a generic connection.

    Args:
        capacity: CO₂ design throughput (tCO₂/yr)
        distance: distance (km)
    """
    # Assuming one truck due to discontinuous operation, despite long depreciation time
    trucks_cost = (100_000 + 200_000) * hicp_index[2021]  # € per truck + trailer
    loading_station_cost = 120_000 * hicp_index[2022]  # € per station; Sievert 2025

    trucks_needed = _truck_fleet_size(capacity, distance)
    truck_capex = trucks_needed * trucks_cost  # €
    loading_station_capex = (
            _truck_loading_stations(capacity) * loading_station_cost
    )  # €
    return truck_capex + loading_station_capex


def opex_truck(capacity, flow, distance):
    """Truck OPEX split into capacity-based and flow-based components.
    If not specified, values are taken from Oeuvray et al. 2024 and generalized to
    fit a generic connection.

    Args:
        capacity: CO₂ design throughput (tCO₂/yr)
        flow:     CO₂ actually transported during a failure event (tCO₂)
        distance: distance (km)

    Returns:
        capacity_opex + flow_opex - annual opex in €
    """
    # 1 worker per station
    labor_cost_per_station = 65_000 * hicp_index[2022]  # €/yr; Sievert et al. 2025
    truck_speed = 80  # km/h; Shoman et al. 2023
    loading_time_per_truck = 1.75  # h; sum of loading and unloading
    gamma_fuel = 0.29  # L/km
    c_fuel = 1.5850  # €/L; fuel-prices.eu, avg. from 2021 to 2026
    c_maint = 0.16 * hicp_index[2021]  # €/km;
    truck_capacity = 26  # tCO₂
    driver_salary = 21.9 * hicp_index[2021]  # €/h
    # heavy goods vehicle tax from Comité National Routier 2025. Medium value within
    # range of European countries to simplify calculation.
    hgvt = 0.35 * hicp_index[2021]  # €/km; heavy goods vehicle tax;

    roundtrip_duration = loading_time_per_truck + (distance / truck_speed) * 2  # h

    trucks_needed = _truck_fleet_size(capacity, distance)
    # storage_opex_cost = storage_opex(capacity)  # €/yr; Eq. 18
    # €/yr; Assuming worker only required during failure event.
    labor_loading_station = (labor_cost_per_station * _truck_loading_stations(capacity)
                             ) * flow / capacity
    # insurance, tax, infrastructure, administration, and tires; Eq. 24 & 25
    # Tires scale with actual utilization
    misc_cost = trucks_needed * (13800 + 3300 + 9000 + 29552
                                 + 11440 * flow / capacity) * hicp_index[2021]  # €/yr
    capacity_opex = labor_loading_station + misc_cost # + storage_opex_cost
    shipments_per_year = np.ceil(flow / truck_capacity)
    maint_fuel_cost = (
            shipments_per_year * (2 * distance) * (gamma_fuel * c_fuel + c_maint)
    )  # €/yr; Eq. 20 & 21
    heavy_duty_vehicle_tax = shipments_per_year * distance * hgvt  # €/yr; Eq. 22
    driver_labor_cost = (
            shipments_per_year * roundtrip_duration * driver_salary
    )  # €/yr; Eq. 23
    flow_opex = maint_fuel_cost + heavy_duty_vehicle_tax + driver_labor_cost

    return capacity_opex + flow_opex


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------


def _train_loading_stations(capacity):
    """Number of loading stations; shared by capex_train and opex_train."""
    wagon_capacity = 50  # tCO₂; TE!C23
    operating_hours = 8520  # h/yr; TE!C24
    unloading_time_per_wagon = 2  # h; TE!C20
    shipments_per_year = capacity / wagon_capacity
    return 2 * shipments_per_year * unloading_time_per_wagon / operating_hours


def capex_train(capacity, distance):
    """Train CAPEX (€); corresponds to CAPEX OPEX!D21

    Args:
        capacity: CO₂ design throughput (tCO₂/yr); CAPEX OPEX!C4
        distance: distance (km); CAPEX OPEX!C6
    """
    train_speed = 18  # km/h; TE!C17
    loading_time = 6  # h; TE!C18
    unloading_time = 12  # h; TE!C19
    wagon_capacity = 50  # tCO₂; TE!C23
    wagon_cost = 4520.18 * hicp_index[2022]  # €/tCO₂; TE!C16
    operating_hours = 8520  # h/yr; TE!C24
    loading_station_cost = 120_000 * hicp_index[2022]  # €; TE!C22

    roundtrip_duration = (loading_time + unloading_time) + (
            distance / train_speed) * 2  # h
    roundtrips_per_year = operating_hours / roundtrip_duration
    wagons_needed = math.ceil(capacity / (roundtrips_per_year * wagon_capacity))
    wagon_capex = wagons_needed * wagon_capacity * wagon_cost  # €
    loading_station_capex = (
            _train_loading_stations(capacity) * loading_station_cost
    )  # €
    return wagon_capex + loading_station_capex


def opex_train(capacity, flow, distance):
    """Train OPEX split into capacity-based and flow-based components.
    Corresponds to CAPEX OPEX!D26

    Args:
        capacity: CO₂ design throughput (tCO₂/yr); CAPEX OPEX!C4
        flow:     CO₂ actually transported during a failure event (tCO₂)
        distance: distance (km); CAPEX OPEX!C6

    Returns:
        capacity_opex + flow_opex - annual opex in €
    """
    labor_per_station = 1  # workers per loading station; TE!C25
    labor_cost_per_worker = 65_000 * hicp_index[2022]  # €/yr per worker; TE!C26
    transport_cost_fixed = 650 * hicp_index[2022]  # €/roundtrip; TE!C27
    transport_cost_var = 2.5 * hicp_index[2022]  # €/km (one-way distance); TE!C28
    wagon_capacity = 50  # tCO₂; TE!C23

    # Capacity-based: station labor and buffer storage
    labor_cost = (labor_per_station * _train_loading_stations(
        capacity) * labor_cost_per_worker)  # €/yr
    capacity_opex = labor_cost # + storage_opex(capacity)

    # Flow-based: transport cost scales with shipments
    shipments = flow / wagon_capacity
    transport_cost_per_shipment = transport_cost_fixed + transport_cost_var * distance
    flow_opex = transport_cost_per_shipment * shipments  # €

    return capacity_opex + flow_opex


# ---------------------------------------------------------------------------
# Barge
# ---------------------------------------------------------------------------
def _barge_fleet_capex(capacity, distance):
    """Barge fleet CAPEX (€); shared by capex_barge and opex_barge."""
    barge_speed = 11.9  # km/h; TE!C41
    loading_unloading_time = 24  # h (2 × 12h); 2 × TE!C40
    operating_hours = 8400  # h/yr; TE!C43
    barge_capacity = 3192  # tCO₂ (8×380 CBM × 1050 kg/m³ / 1000); TE!C34×C35/1000
    utilization_factor = 0.65  # ; TE!C42
    barge_cost = 16_300_000 * hicp_index[2022]  # € per barge; TE!C33

    roundtrip_duration = (2 * distance / barge_speed) + loading_unloading_time  # h
    roundtrips_per_year = math.floor(operating_hours / roundtrip_duration)
    barges_needed = capacity / (
            barge_capacity * utilization_factor * roundtrips_per_year
    )
    return barges_needed * barge_cost  # €


def capex_barge(capacity, distance):
    """Barge CAPEX (€); corresponds to CAPEX OPEX!D37

    Args:
        capacity: CO₂ design throughput (tCO₂/yr); CAPEX OPEX!C4
        distance: distance (km); CAPEX OPEX!C6
    """
    loading_cost_per_tco2 = 2.87196 * hicp_index[2022]  # €/tCO₂; TE!C47

    fleet_capex = _barge_fleet_capex(capacity, distance)
    loading_station_capex = loading_cost_per_tco2 * 2 * capacity  # €
    return fleet_capex + loading_station_capex # + storage_capex(capacity)


def opex_barge(capacity, flow, distance):
    """Barge OPEX split into capacity-based and flow-based components.
    Corresponds to CAPEX OPEX!D41

    Args:
        capacity: CO₂ design throughput (tCO₂/yr); CAPEX OPEX!C4
        flow:     CO₂ actually transported during a failure event (tCO₂)
        distance: distance (km); CAPEX OPEX!C6

    Returns:
        capacity_opex + flow_opex - annual opex in €
    """
    opex_rate = 0.074  # fraction of barge CAPEX/yr; TE!C36
    fuel_consumption = 10.4877864788785  # g diesel / (tCO₂·km); TE!C37
    fuel_cost_per_tonne = 500 * hicp_index[2022]  # €/t diesel; TE!C38

    capacity_opex = _barge_fleet_capex(capacity, distance) * opex_rate
    flow_opex = (
            fuel_consumption * flow * distance * fuel_cost_per_tonne / 1_000_000
    )  # €

    return capacity_opex + flow_opex


# ---------------------------------------------------------------------------
# Ship
# ---------------------------------------------------------------------------


def _ship_fleet_capex(capacity, distance):
    """Ship fleet CAPEX (€); shared by capex_ship and opex_ship."""
    ship_speed = 27.78  # km/h; TE!C58
    loading_time = 15  # h per ship; TE!C51
    additional_time = 2  # h; TE!C54
    turnaround_time = 2 * (loading_time + additional_time)  # 34 h total
    operating_hours = 8400  # h/yr; TE!C55
    ship_capacity = 50_000  # tCO₂; TE!C62
    ship_cost = 92_468_700 * hicp_index[2022]  # € per ship; TE!C63

    roundtrip_duration = 2 * distance / ship_speed + turnaround_time  # h
    roundtrips_per_year = math.floor(operating_hours / roundtrip_duration)
    ships_needed = math.ceil(capacity / (roundtrips_per_year * ship_capacity))
    return ships_needed * ship_cost  # €


def _ship_loading_capex(capacity):
    """Ship loading station CAPEX (€); shared by capex_ship and opex_ship."""
    loading_cost_per_tco2 = 2.87196 * hicp_index[2022]  # €/tCO₂; TE!C52
    return loading_cost_per_tco2 * 2 * capacity  # €


def capex_ship(capacity, distance):
    """Ship Low Pressure CAPEX (€); corresponds to CAPEX OPEX!D51

    Args:
        capacity: CO₂ design throughput (tCO₂/yr); CAPEX OPEX!C4
        distance: distance (km); CAPEX OPEX!C6
    """
    return (
            _ship_fleet_capex(capacity, distance)
            + _ship_loading_capex(capacity)
    )


def opex_ship(capacity, flow, distance):
    """Ship Low Pressure OPEX split into capacity-based and flow-based components.
    Corresponds to CAPEX OPEX!D57

    Args:
        capacity: CO₂ design throughput (tCO₂/yr); CAPEX OPEX!C4
        flow:     CO₂ actually transported during a failure event (tCO₂)
        distance: distance (km); CAPEX OPEX!C6

    Returns:
        capacity_opex + flow_opex - annual opex in €
    """
    fuel_consumption = 5.19  # g / (tCO₂·km); TE!C64
    fuel_cost_per_tonne = 325 * hicp_index[2022]  # €/t bunker fuel; TE!C67
    opex_rate = 0.05  # fraction of ship CAPEX/yr; TE!C65
    misc_cost_fixed = 1.2012 * hicp_index[2022]  # €/tCO₂/yr; TE!C66
    loading_opex_rate = 0.02  # fraction of loading CAPEX/yr; TE!C53
    harbor_fee = 0.26 * hicp_index[2022]  # €/tCO₂; TE!C39

    ship_capex = _ship_fleet_capex(capacity, distance)
    loading_capex = _ship_loading_capex(capacity)

    # Capacity-based: fleet maintenance + loading station maintenance + storage
    capacity_opex = (
            ship_capex * opex_rate
            + loading_capex * loading_opex_rate
    )  # €

    # Flow-based: fuel, per-tonne misc, and harbor fees
    fuel_cost = (
            fuel_consumption * flow * distance * fuel_cost_per_tonne / 1_000_000
    )  # €
    flow_opex = fuel_cost + misc_cost_fixed * flow + harbor_fee * flow  # €

    return capacity_opex + flow_opex
