from __future__ import annotations

from pythermonet.components.ground_loads import ground_loads_from_heat_pumps
from pythermonet.optimizer.heat_pump import create_heat_pump
from pythermonet.core.material import Material
from pythermonet.core.heat_carrier import HeatCarrier
from pythermonet.core.soil import Soil
from pythermonet.input.read_pipe_catalogue import read_pipe_catalogue
from pythermonet.input.read_topology import read_undimensioned_topology_tsv_to_network
from pythermonet.input.df_from_sources import df_from_csv
from pythermonet.dimensioning.hydraulic_dimensioning import run_pipedimensioning

from pythermonet.components.vhe_field import VHEField
from pythermonet.components.distribution_network import DistributionNetwork
from pythermonet.components.pipe_infrastructure import PipeInfrastructure
from pythermonet.core.annulus import Annulus
from pythermonet.core.pipe_segment import PipeSegment

from pythermonet.dimensioning.BHE.bhe_workflow import run_bhe_sizing_workflow, print_bhe_results, BHEWorkflowResult

from pythermonet.components.heat_pump import HeatPump


import numpy as np
import sys
from pathlib import Path
from typing import List


def _filter_network_to_active_hps(
    network: DistributionNetwork, active_hp_ids: set[int]
) -> DistributionNetwork:
    """Return a topology-shape copy with per-section HP ID lists filtered
    down to `active_hp_ids`. Sections whose filtered HP list is empty are
    dropped across all parallel arrays.

    For "per-HP replicated" sections — where `N_traces` originally equals
    the number of HP IDs listed for that section (e.g. one service-line
    copy per building) — `N_traces` is scaled down to the filtered HP
    count so `N_HP_per_trace` stays at 1. For shared trunk sections
    (`N_traces == 1`, one physical run serving many HPs), `N_traces`
    stays as-is.
    """
    keep_idx: list[int] = []
    new_hp_id_trace: list[np.ndarray] = []
    new_N_traces: list[int] = []
    for i, ids in enumerate(network.hp_id_trace):
        filtered = np.array([int(hid) for hid in ids if int(hid) in active_hp_ids], dtype=int)
        if filtered.size == 0:
            continue
        keep_idx.append(i)
        new_hp_id_trace.append(filtered)

        original_n_ids = int(len(ids))
        original_n_traces = int(network.N_traces[i])
        if original_n_traces == original_n_ids:
            new_N_traces.append(int(filtered.size))
        else:
            new_N_traces.append(original_n_traces)

    infra = network.infrastructure
    new_infra = PipeInfrastructure(
        NParallelPipes=infra.NParallelPipes,
        traceSegments=[infra.traceSegments[i] for i in keep_idx],
        pipeDistance=infra.pipeDistance,
        burialDepth=infra.burialDepth,
    )
    return DistributionNetwork(
        infrastructure=new_infra,
        trace_names=[network.trace_names[i] for i in keep_idx],
        hp_id_trace=new_hp_id_trace,
        max_pressure_loss_trace=network.max_pressure_loss_trace[keep_idx],
        SDR=network.SDR[keep_idx],
        L_traces=network.L_traces[keep_idx],
        N_traces=np.array(new_N_traces, dtype=int),
    )


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





def setup_init_model(n_boreholes: int = 6):
    pipe_catalogue, pipe_material_dist, brine, soil = _define_materials()
    undim_distrib_network = _define_undim_distribution_network(pipe_material_dist)
    BHEField = _setup_borefield(n_boreholes=n_boreholes)

    return pipe_catalogue, brine, soil, undim_distrib_network, BHEField





def execute_dimensioning(
    hp_list: List[HeatPump],
    sizing,
    n_boreholes: int = 6,
    active_hp_ids: set[int] | None = None,
):

    sys.stdout.reconfigure(encoding="utf-8")


    pipe_catalogue, brine, soil, distribution_network_undimensioned, BHEfield = setup_init_model(n_boreholes=n_boreholes)

    if active_hp_ids is not None:
        distribution_network_undimensioned = _filter_network_to_active_hps(
            distribution_network_undimensioned, active_hp_ids
        )

    # The fake HP is an external heat injector — not part of the topology, so it
    # must NOT enter hydraulic pipe sizing. It DOES contribute to the ground
    # heat balance, so it goes into the load aggregation only.

    loads = ground_loads_from_heat_pumps(hp_list, brine, **_GROUND_LOADS_KWARGS)
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
        hp_list,
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

    # mwh = _calc_building_hp_electricity_MWh(hp_list)

    # annual_heat_MWh = sum(hp.annualHeatingLoad for hp in hp_list) * 8760.0 / 1_000_000.0
    # building_ground_extraction_MWh = sum(
    #     hp.annualHeating_ground_load for hp in hp_list
    # ) * 8760.0 / 1_000_000.0

    return {
        "source_length": result.sizing.L_m,
        "num_boreholes": BHEfield.n_boreholes
        # "elec_consump": elec_demand,
        # "heat_consump": annual_heat_MWh,
        # "building_ground_extraction_MWh": building_ground_extraction_MWh,
    }

    