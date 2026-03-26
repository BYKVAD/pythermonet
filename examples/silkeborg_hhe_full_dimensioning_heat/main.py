from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from pythermonet.components.aggregated_heat_pumps import AggregatedHeatPumps
from pythermonet.components.pipe_infrastructure import PipeInfrastructure
from pythermonet.core.material import Material
from pythermonet.core.heat_carrier import HeatCarrier
from pythermonet.core.pipe_segment import PipeSegment
from pythermonet.core.soil import Soil
from pythermonet.dimensioning.ground_field import HHEGroundField
from pythermonet.dimensioning.HHE.hhe_workflow import run_hhe_sizing_workflow, print_hhe_results
from pythermonet.dimensioning.sizing_parameters import SizingParameters
from pythermonet.input.read_dimensioned_topology import read_dimensioned_topology_tsv_to_hydraulic
from pythermonet.input.read_aggregated_load import read_aggregated_load_tsv

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent

agg_load_file  = PROJECT_DIR / "data/silkeborg_hhe_aggregated_load_heat.dat"
topology_file  = PROJECT_DIR / "data/silkeborg_hhe_topology_dimensioned.dat"

# -----------------------------------------------------------------------------
# 1) Define materials, brine, soil
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# 2) Distribution network (pre-dimensioned — no hydraulic sizing needed)
# -----------------------------------------------------------------------------
_, hydraulic = read_dimensioned_topology_tsv_to_hydraulic(
    topology_file,
    pipe_material=pipe_material_dist,
    brine=brine,
    burial_depth=1.2,
    pipe_distance=0.3,
    n_parallel_pipes=2,
)

# -----------------------------------------------------------------------------
# 3) HHE field  (20 parallel pipes = 10 loops, horizontal, 1.2 m burial depth)
# -----------------------------------------------------------------------------
hhe_pipe_material = Material(rho=950, c=2300, thermalCond=0.4)

hhe_segment = PipeSegment(
    outerDiameter=0.040,    # 40 mm OD
    SDR=17.0,
    material=hhe_pipe_material,
    roughnessHeight=1e-6,
    ID=0,
    length=100.0,           # placeholder — will be sized
)

pipe_infrastructure = PipeInfrastructure(
    NParallelPipes=20,      # 10 loops (outgoing + return)
    traceSegments=[hhe_segment],
    pipeDistance=1.5,       # lateral spacing between pipes [m]
    burialDepth=1.2,        # m
)

hhe_field = HHEGroundField(
    pipe_infrastructure=pipe_infrastructure,
    k_s=float(soil.thermalCondShallowHeating),
)

# -----------------------------------------------------------------------------
# 4) Aggregated heat pump loads
# -----------------------------------------------------------------------------
agg_load_input = read_aggregated_load_tsv(agg_load_file)

heat_pumps = AggregatedHeatPumps(
    load_input=agg_load_input,
    brine=brine,
    f_peak_heating=1.0,
    f_peak_cooling=1.0,
    peak_heating_h=4.0,
    peak_cooling_h=4.0,
)

sizing = SizingParameters(time_horizon_years=30.0)

# -----------------------------------------------------------------------------
# 5) Brine temperature limits
# -----------------------------------------------------------------------------
T_BRINE_MIN_HEAT = -3.0   # HP evaporator inlet limit [°C]
T_BRINE_MAX_COOL = 20.0   # HP condenser inlet limit [°C]

# -----------------------------------------------------------------------------
# 6) HHE sizing workflow + results
# -----------------------------------------------------------------------------
result = run_hhe_sizing_workflow(
    heat_pumps=heat_pumps,
    hhe_field=hhe_field,
    pipe_infrastructure=pipe_infrastructure,
    hydraulic=hydraulic,
    brine=brine,
    soil=soil,
    sizing=sizing,
    T_brine_min_heat=T_BRINE_MIN_HEAT,
    T_brine_max_cool=T_BRINE_MAX_COOL,
)
print_hhe_results(result, pipe_infrastructure)
