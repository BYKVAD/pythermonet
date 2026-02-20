from __future__ import annotations

from pathlib import Path

from pythermonet.components.heat_pumps import HeatPumps
from pythermonet.core.material import Material
from pythermonet.core.heat_carrier import HeatCarrier
from pythermonet.core.soil import Soil
from pythermonet.input.read_pipe_catalogue import read_pipe_catalogue
from pythermonet.input.read_topology import read_undimensioned_topology_tsv_to_network
from pythermonet.input.read_heat_pumps import read_heat_pumps_tsv
from pythermonet.dimensioning.hydraulic_dimensioning import run_pipedimensioning

from pythermonet.components.vhe_field import VHEField
from pythermonet.core.annulus import Annulus
from pythermonet.core.pipe_segment import PipeSegment

from pythermonet.gfunctions.models import GFunctionRequest
from pythermonet.gfunctions.service import GFunctionsService

from pythermonet.physics.bhe_resistance import compute_rb_for_vhe_field
from pythermonet.simulation.run_distribution_pipe_thermal import run_distribution_pipe_thermal


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parents[2]

pipe_catalogue_file = REPO_ROOT / "PythermonetII/src/pythermonet/resources/pipe_catalogue.csv"
heat_pump_file = PROJECT_DIR / "data/silkeborg_heat_pump_heat.dat"
topology_file = PROJECT_DIR / "data/silkeborg_topology.dat"


# -----------------------------------------------------------------------------
# Inputs: catalogue + materials + fluids + soil
# -----------------------------------------------------------------------------
pipe_catalogue = read_pipe_catalogue(pipe_catalogue_file)

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
    rho=2650,
    c=1000,
    thermalCond=2.36,
    thermalCondShallowHeating=1.25,
    thermalCondShallowCooling=1.25,
    Qgeo=0.01,
    surfaceTemp=9,
    surfaceTempAmp=7.9,
)


# -----------------------------------------------------------------------------
# 1) Undimensioned distribution topology
# -----------------------------------------------------------------------------
undim_network = read_undimensioned_topology_tsv_to_network(
    topology_file,
    pipe_material=pipe_material_dist,
    roughness_height=1e-6,
    burial_depth=1.2,
    pipe_distance=0.3,
    n_parallel_pipes=2,
)


# -----------------------------------------------------------------------------
# 2) Heat pumps (carrier ONLY in HeatPumps)
# -----------------------------------------------------------------------------
hp_list = read_heat_pumps_tsv(path=heat_pump_file)

hp = HeatPumps(
    heatPumpList=hp_list,
    brine=brine,
    peak_heating_h=4.0,
    peak_fraction_heating_mode="incremental",
    peak_fraction_heating=1.0,
    peak_cooling_h=4.0,
    peak_fraction_cooling_mode="incremental",
    peak_fraction_cooling=1.0,
)

print("N HP:", hp.n_heat_pumps)
print("Diversity:", hp.diversity_factor)
print("ΔT_sys_heat [K]:", hp.deltaT_sys_heat_C)
print("ΔT_sys_cool [K]:", hp.deltaT_sys_cool_C)


# -----------------------------------------------------------------------------
# 3) Hydraulic pipe sizing
# -----------------------------------------------------------------------------
distribution_network = run_pipedimensioning(
    pipe_catalogue=pipe_catalogue,
    brine=brine,
    network=undim_network,
    heat_pumps=hp,
)

print("Dimensioned distribution network:", distribution_network.infrastructure.traceSegments)


# -----------------------------------------------------------------------------
# 4) Borehole field + g-functions
# -----------------------------------------------------------------------------
n_boreholes = 6
spacing_m = 15.0
borehole_diameter_m = 0.152
r_b_m = borehole_diameter_m / 2.0
H_m = 150.0
D_m = 1.5

times_s = [4 * 3600, 86400 * 365.25 / 4, 30 * 365.25 * 86400]
alpha_m2_s = soil.thermalCond / (soil.rho * soil.c)

# U-pipe geometry (legacy)
u_pipe_outer_diameter_m = 0.04
u_pipe_sdr = 11.0

grout = Material(rho=1500, c=2e3, thermalCond=1.75)

pipe_material_bhe = Material(rho=1000, c=2e3, thermalCond=0.4)

borehole = Annulus(outerDiameter=borehole_diameter_m, SDR=1000.0)

pipe = PipeSegment(
    outerDiameter=u_pipe_outer_diameter_m,
    SDR=u_pipe_sdr,
    material=pipe_material_bhe,
    roughnessHeight=1e-6,
    ID=0,
    length=1.0,  # placeholder
)

coordinates = [[0.0, i * spacing_m, 0.0] for i in range(n_boreholes)]

vhe_field = VHEField(
    ID=1,
    HE="1U",
    pipe=pipe,
    borehole=borehole,
    grout=grout,
    coordinates=coordinates,
    shankSpacing=0.015 + 2 * 0.02,
)

boreholes = vhe_field.to_pygfunction_boreholes(
    H_m=H_m,
    D_m=D_m,
    r_b_m=r_b_m,
    use_z_as_depth=False,
)

req = GFunctionRequest(
    times_s=times_s,
    boreholes=boreholes,
    alpha_m2_s=alpha_m2_s,
    method="equivalent",
    boundary_condition="UHTR",
    options={},
)

gset = GFunctionsService(cache=None).compute(req)
print("Computed g-functions:", gset.g_values[:5])


# -----------------------------------------------------------------------------
# 5) BHE thermal resistance (Rb)
# -----------------------------------------------------------------------------
Q_peak_m3_s = hp.aggregated_q_peak_heat_m3_s
print("Peak volumetric flow rate [m3/s]:", Q_peak_m3_s)

rb = compute_rb_for_vhe_field(
    vhe_field=vhe_field,
    brine=brine,
    soil=soil,
    L_bhe_m=H_m,
    m_dot_kg_s=hp.aggregated_mdot_peak_heat_kg_s / n_boreholes,
    use_flow_length_correction=True,
)

print("Rb [K*m/W] =", rb.Rb_K_m_W)
print("Re, Pr     =", rb.Re, rb.Pr)


# -----------------------------------------------------------------------------
# 6) Distribution pipe thermal model
# -----------------------------------------------------------------------------
dist_therm = run_distribution_pipe_thermal(
    network=distribution_network,
    brine=brine,
    soil=soil,
    heat_pumps=hp,
    peak_heating_h=4.0,
    T_brine_min_heat_C=-3.0,
    T_brine_max_cool_C=25.0,
)

print("T_dimv (heating) [°C]:", dist_therm.heating.T_dimv_C)
print("F_total (heating):", dist_therm.heating.F_total)