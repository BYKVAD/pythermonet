from __future__ import annotations

from pythermonet.components.ground_loads import ground_loads_from_heat_pumps
from pythermonet.optimizer.fake_heat_pump import build_fake_heat_pump
from pythermonet.core.material import Material
from pythermonet.core.heat_carrier import HeatCarrier
from pythermonet.core.soil import Soil
from pythermonet.input.read_pipe_catalogue import read_pipe_catalogue
from pythermonet.input.read_topology import read_undimensioned_topology_tsv_to_network
from pythermonet.input.read_heat_pumps import read_heat_pumps_tsv
from pythermonet.input.df_from_sources import df_from_csv
from pythermonet.dimensioning.hydraulic_dimensioning import run_pipedimensioning

from pythermonet.components.vhe_field import VHEField
from pythermonet.core.annulus import Annulus
from pythermonet.core.pipe_segment import PipeSegment

from pythermonet.dimensioning.BHE.bhe_workflow import run_bhe_sizing_workflow, print_bhe_results, BHEWorkflowResult
from pythermonet.dimensioning.sizing_parameters import SizingParameters



import sys
from pathlib import Path
from typing import List



BUILDING_HP_COP = 3.3


def _calc_building_hp_electricity_MWh(building_hp_list) -> float:
    """
    Annual electricity drawn by the building heat pumps to deliver their
    annual heating load (assumed COP = BUILDING_HP_COP). Independent of
    ground regeneration — building demand does not change when a
    supplementary injector reduces ground extraction.
    """
    annual_heat_W = sum(hp.annualHeatingLoad for hp in building_hp_list)
    annual_heat_MWh = annual_heat_W * 8760.0 / 1_000_000.0
    return annual_heat_MWh / BUILDING_HP_COP


def _define_materials() -> tuple[List[Annulus], Material, HeatCarrier, Soil]:
    pipe_catalogue = read_pipe_catalogue(
        df_from_csv(r"C:\software\pythermonetII\src\pythermonet\resources\pipe_catalogue.csv")
    )

    pipe_material_dist = Material(
        rho=975,
        c=1900,
        thermalCond=0.4,
    )

    brine = HeatCarrier(
        rho=965,
        c=4450,
        thermalCond=0.45,
        dynamicViscosity=5e-3,
    )

    soil = Soil(
        rho=2500,
        c=1000,
        thermalCond=2.36,
        thermalCondShallowHeating=1.25,
        thermalCondShallowCooling=1.25,
        Qgeo=0.0185,
        surfaceTemp=9.03,
        surfaceTempAmp=7.9,
    )

    return pipe_catalogue, pipe_material_dist, brine, soil


def _define_undim_distribution_network(pipe_material_dist: Material):
    return read_undimensioned_topology_tsv_to_network(
        df_from_csv(r"C:\software\pythermonetII\examples\silkeborg_bhe_full_dimensioning_heat\data\silkeborg_topology.dat", sep=r"\t+"),
        pipe_material=pipe_material_dist,
        roughness_height=1e-6,
        burial_depth=1.2,
        pipe_distance=0.3,
        n_parallel_pipes=2,
    )


def _setup_borefield(n_boreholes: int = 6, spacing_m: float = 15.0):
    if n_boreholes < 1:
        raise ValueError("n_boreholes must be >= 1")
    coordinates = [[0.0, i * spacing_m] for i in range(n_boreholes)]
    borehole_diameter_m = 0.152
    u_pipe_outer_diameter_m = 0.04
    u_pipe_sdr = 11.0
    grout = Material(rho=1500, c=2e3, thermalCond=1.75)
    pipe_material_bhe = Material(rho=1000, c=2e3, thermalCond=0.4)
    borehole = Annulus(outerDiameter=borehole_diameter_m, SDR=1000.0)

    upipe = PipeSegment(
        outerDiameter=u_pipe_outer_diameter_m,
        SDR=u_pipe_sdr,
        material=pipe_material_bhe,
        roughnessHeight=1e-6,
        ID=0,
        length=100,  # placeholder
    )

    BHEfield = VHEField(
        ID = 1,
        HE='1U',
        pipe = upipe,
        borehole = borehole,
        grout = grout,
        coordinates=coordinates,
        shankSpacing=0.015 + 2 * 0.02,
        H_m=120.0,
        D_m=1.0,
        tilt_rad=0.0,
        orientation_rad=0.0,
    )

    return BHEfield


_GROUND_LOADS_KWARGS = dict(
    peak_heating_h=4.0,
    peak_fraction_heating_mode="incremental",
    peak_fraction_heating=1.0,
    peak_cooling_h=4.0,
    peak_fraction_cooling_mode="incremental",
    peak_fraction_cooling=1.0,
)


def _setup_heatpump():
    hp_list = read_heat_pumps_tsv(df_from_csv(r"C:\software\pythermonetII\examples\silkeborg_bhe_full_dimensioning_heat\data\silkeborg_heat_pump_heat_only.dat",sep=r"\t+"))

    sizing = SizingParameters(time_horizon_years=30.0)

    return hp_list, sizing


def setup_init_model(n_boreholes: int = 6):
    pipe_catalogue, pipe_material_dist, brine, soil = _define_materials()
    undim_distrib_network = _define_undim_distribution_network(pipe_material_dist)
    BHEField = _setup_borefield(n_boreholes=n_boreholes)
    hp_list, sizing = _setup_heatpump()

    return pipe_catalogue, brine, soil, undim_distrib_network, BHEField, hp_list, sizing



def execute_dimensioning(
    rated_power_W: float,
    capacity_factor: float | None = None,
    summer_capacity_factor: float | None = None,
    n_boreholes: int = 6,
):

    sys.stdout.reconfigure(encoding="utf-8")


    pipe_catalogue, brine, soil, distribution_network_undimensioned, BHEfield, building_hp_list, sizing = setup_init_model(n_boreholes=n_boreholes)

    # The fake HP is an external heat injector — not part of the topology, so it
    # must NOT enter hydraulic pipe sizing. It DOES contribute to the ground
    # heat balance, so it goes into the load aggregation only.
    hp_list_for_loads = list(building_hp_list)
    if rated_power_W > 0:
        hp_list_for_loads.append(build_fake_heat_pump(
            rated_power_W,
            capacity_factor=capacity_factor,
            summer_capacity_factor=summer_capacity_factor,
        ))

    loads = ground_loads_from_heat_pumps(hp_list_for_loads, brine, **_GROUND_LOADS_KWARGS)
    # -----------------------------------------------------------------------------
    # 5) Brine temperature limits
    # -----------------------------------------------------------------------------
    T_BRINE_MIN_HEAT = -3.0   # HP evaporator inlet limit [°C]
    T_BRINE_MAX_COOL = 20.0   # HP condenser inlet limit [°C]

    # -----------------------------------------------------------------------------
    # 6) Hydraulic pipe network sizing (mode-specific)
    # -----------------------------------------------------------------------------
    hydraulic = run_pipedimensioning(
        pipe_catalogue,
        brine,
        distribution_network_undimensioned,
        building_hp_list,
    )

    # -----------------------------------------------------------------------------
    # 8) BHE sizing workflow + results
    # -----------------------------------------------------------------------------
    result = run_bhe_sizing_workflow(
        ground_loads=loads,
        vhe_field=BHEfield,
        hydraulic=hydraulic,
        brine=brine,
        soil=soil,
        sizing=sizing,
        T_brine_min_heat=T_BRINE_MIN_HEAT,
        T_brine_max_cool=T_BRINE_MAX_COOL,
    )

    mwh = _calc_building_hp_electricity_MWh(building_hp_list)

    annual_heat_MWh = sum(hp.annualHeatingLoad for hp in building_hp_list) * 8760.0 / 1_000_000.0
    building_ground_extraction_MWh = sum(
        hp.annualHeating_ground_load for hp in building_hp_list
    ) * 8760.0 / 1_000_000.0

    return {
        "source_length": result.sizing.L_m,
        "num_boreholes": BHEfield.n_boreholes,
        "elec_consump": mwh,
        "heat_consump": annual_heat_MWh,
        "building_ground_extraction_MWh": building_ground_extraction_MWh,
    }

    