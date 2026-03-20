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

from pythermonet.physics.bhe_resistance import compute_rb_for_vhe_field
from pythermonet.simulation.run_distribution_pipe_thermal_model import compute_distribution_pipe_thermal_capacity, print_pipe_thermal_table
from pythermonet.dimensioning.borehole_length import size_borehole_length_heating_cooling, apply_annual_balance
from pythermonet.dimensioning.system_temperatures import compute_system_brine_temperatures
import numpy as np

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parents[2]

pipe_catalogue_file = REPO_ROOT / "PythermonetII/src/pythermonet/resources/pipe_catalogue.csv"
heat_pump_file = PROJECT_DIR / "data/silkeborg_heat_pump_heat.dat" #test commit
topology_file = PROJECT_DIR / "data/silkeborg_topology.dat"


# -----------------------------------------------------------------------------
# 1) Read pipe catalogue, define materials + fluids + soil
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
    Qgeo=0.0185,
    surfaceTemp=9.03,
    surfaceTempAmp=7.9,
)

# -----------------------------------------------------------------------------
# 2) Distribution network
# -----------------------------------------------------------------------------
distribution_network_undimensioned = read_undimensioned_topology_tsv_to_network(
    topology_file,
    pipe_material=pipe_material_dist,
    roughness_height=1e-6,
    burial_depth=1.2,
    pipe_distance=0.3,
    n_parallel_pipes=2,
)

# -----------------------------------------------------------------------------
# 3) Borehole field (only identical boreholes supported for now)
# -----------------------------------------------------------------------------
n_boreholes = 6
spacing_m = 15.0
coordinates = [[0.0, i * spacing_m] for i in range(n_boreholes)]
borehole_diameter_m = 0.152
r_b_m = borehole_diameter_m / 2.0
H_m = 150.0
D_m = 1.5
u_pipe_outer_diameter_m = 0.04
u_pipe_sdr = 11.0
grout = Material(rho=1500, c=2e3, thermalCond=1.75)
pipe_material_bhe = Material(rho=1000, c=2e3, thermalCond=0.4)
borehole = Annulus(outerDiameter=borehole_diameter_m, SDR=1000.0)

# Create the U-pipe for the borehole
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
    r_b_m=0.075,
    tilt_rad=0.0,
    orientation_rad=0.0,
)

# -----------------------------------------------------------------------------
# 4) Heat pumps
# -----------------------------------------------------------------------------
hp_list = read_heat_pumps_tsv(path=heat_pump_file)

heat_pumps = HeatPumps(
    heatPumpList=hp_list,
    brine=brine,
    peak_heating_h=4.0,
    peak_fraction_heating_mode="incremental",
    peak_fraction_heating=1.0,
    peak_cooling_h=4.0,
    peak_fraction_cooling_mode="incremental",
    peak_fraction_cooling=1.0,
)

# -----------------------------------------------------------------------------
# 5) Brine temperature limits
# -----------------------------------------------------------------------------
T_BRINE_MIN_HEAT = -3.0   # HP evaporator inlet limit [°C]
T_BRINE_MAX_COOL = 25.0   # HP condenser inlet limit [°C]

# -----------------------------------------------------------------------------
# 6) Simulation time vector
# -----------------------------------------------------------------------------
times_s = [
    4 * 3600,
    4 * 3600 + 86400 * 365.25 / 4,
    4 * 3600 + 86400 * 365.25 / 4 + 30 * 365.25 * 86400,
]

# -----------------------------------------------------------------------------
# 7) Hydraulic pipe network sizing
# -----------------------------------------------------------------------------
hydraulic = run_pipedimensioning(
    pipe_catalogue,
    brine,
    distribution_network_undimensioned,
    heat_pumps,
)

# -----------------------------------------------------------------------------
# 7) Distribution pipe thermal simulation
# -----------------------------------------------------------------------------
_P_heat_full = np.asarray(heat_pumps.heating_ground_load_W, dtype=float)
_P_cool_full = np.asarray(heat_pumps.cooling_ground_load_W, dtype=float) if heat_pumps.has_cooling else None
if _P_cool_full is not None:
    _P_heat_full, _P_cool_full = apply_annual_balance(_P_heat_full.copy(), _P_cool_full.copy())
print(f"Full heating ground load after annual balance [annual / winter / peak] [W]: "
      f"{_P_heat_full[0]:.0f} / {_P_heat_full[1]:.0f} / {_P_heat_full[2]:.0f}")
if _P_cool_full is not None:
    print(f"Full cooling ground load after annual balance [annual / winter / peak] [W]: "
          f"{_P_cool_full[0]:.0f} / {_P_cool_full[1]:.0f} / {_P_cool_full[2]:.0f}")

dist_thermal = compute_distribution_pipe_thermal_capacity(
    hydraulic=hydraulic,
    brine=brine,
    soil=soil,
    heat_pumps=heat_pumps,
    times_heat_s=np.flip(times_s),
    times_cool_s=np.flip(times_s) if heat_pumps.has_cooling else None,
    T_brine_min_heat=T_BRINE_MIN_HEAT,
    T_brine_max_cool=T_BRINE_MAX_COOL if heat_pumps.has_cooling else None,
)

# -----------------------------------------------------------------------------
# 8) Compute fractions of thermal loads supplied by the boreholes
# BHE loads = balanced full loads × (1 − distribution fraction)
# Balance was already applied once before distribution pipe simulation.
# -----------------------------------------------------------------------------
P_heating = (1 - dist_thermal["heating"].F_total) * _P_heat_full
P_cooling = (
    (1 - dist_thermal["cooling"].F_total) * _P_cool_full
    if heat_pumps.has_cooling else None
)

print(f"BHE heating loads after annual balance [annual / winter / peak] [W]: "
      f"{P_heating[0]:.0f} / {P_heating[1]:.0f} / {P_heating[2]:.0f}")
if P_cooling is not None:
    print(f"BHE cooling loads after annual balance [annual / winter / peak] [W]: "
          f"{P_cooling[0]:.0f} / {P_cooling[1]:.0f} / {P_cooling[2]:.0f}")

# -----------------------------------------------------------------------------
# 9) Size borehole length from heating and cooling constraints
# -----------------------------------------------------------------------------
sizing = size_borehole_length_heating_cooling(
    T_fluid_min=T_BRINE_MIN_HEAT - 0.5 * heat_pumps.deltaT_sys_heat,
    P_heating_W=P_heating,
    times_heat_s=times_s,
    T_fluid_max=T_BRINE_MAX_COOL + (0.5 * heat_pumps.deltaT_sys_cool if heat_pumps.has_cooling else 0.0),
    P_cooling_W=P_cooling,
    times_cool_s=times_s if heat_pumps.has_cooling else None,
    vhe_field=BHEfield,
    brine=brine,
    soil=soil,
    m_dot_per_borehole_heat_kg_s=heat_pumps.aggregated_mdot_peak_heat_kg_s / n_boreholes,
    m_dot_per_borehole_cool_kg_s=(
        heat_pumps.aggregated_mdot_peak_cool_kg_s / n_boreholes
        if heat_pumps.has_cooling else None
    ),
    pre_balanced=True,
)
print(f"Required borehole length: {sizing.H_m:.8f} m (governed by {sizing.governing})")
print(f"Rb heating [K·m/W]: {sizing.rb_heating.Rb_K_m_W:.4f}")
if sizing.rb_cooling is not None:
    print(f"Rb cooling [K·m/W]: {sizing.rb_cooling.Rb_K_m_W:.4f}")

# -----------------------------------------------------------------------------
# 12) System mean brine temperatures
# -----------------------------------------------------------------------------
sys_temps = compute_system_brine_temperatures(
    sizing=sizing,
    vhe_field=BHEfield,
    hydraulic=hydraulic,
    heat_pumps=heat_pumps,
    P_heating_W=P_heating,
    times_heat_s=times_s,
    P_cooling_W=P_cooling,
    times_cool_s=times_s if heat_pumps.has_cooling else None,
    soil=soil,
    dist_thermal_heat=dist_thermal["heating"],
    dist_thermal_cool=dist_thermal["cooling"] if heat_pumps.has_cooling else None,
    pre_balanced=True,
)
print(f"BHE  mean temp heating   [annual / winter / peak]: "
      f"{sys_temps.T_bhe_heat_annual_C:.2f} / {sys_temps.T_bhe_heat_winter_C:.2f} / {sys_temps.T_bhe_heat_peak_C:.2f} °C")
if sys_temps.T_dist_heat_annual_C is not None:
    print(f"Dist mean temp heating   [annual / winter / peak]: "
          f"{sys_temps.T_dist_heat_annual_C:.2f} / {sys_temps.T_dist_heat_winter_C:.2f} / {sys_temps.T_dist_heat_peak_C:.2f} °C")
print(f"System mean temp heating [annual / winter / peak]: "
      f"{sys_temps.T_avg_heat_annual_C:.2f} / {sys_temps.T_avg_heat_winter_C:.2f} / {sys_temps.T_avg_heat_peak_C:.2f} °C")
if sys_temps.T_avg_cool_peak_C is not None:
    print(f"BHE  mean temp cooling   [annual / winter / peak]: "
          f"{sys_temps.T_bhe_cool_annual_C:.2f} / {sys_temps.T_bhe_cool_winter_C:.2f} / {sys_temps.T_bhe_cool_peak_C:.2f} °C")
    if sys_temps.T_dist_cool_annual_C is not None:
        print(f"Dist mean temp cooling   [annual / winter / peak]: "
              f"{sys_temps.T_dist_cool_annual_C:.2f} / {sys_temps.T_dist_cool_winter_C:.2f} / {sys_temps.T_dist_cool_peak_C:.2f} °C")
    print(f"System mean temp cooling [annual / winter / peak]: "
          f"{sys_temps.T_avg_cool_annual_C:.2f} / {sys_temps.T_avg_cool_winter_C:.2f} / {sys_temps.T_avg_cool_peak_C:.2f} °C")
print(f"BHE fluid volume:  {sys_temps.V_bhe_m3:.3f} m³  ({sys_temps.bhe_fraction*100:.1f}% of total)")
print(f"Dist fluid volume: {sys_temps.V_dist_m3:.3f} m³  ({(1-sys_temps.bhe_fraction)*100:.1f}% of total)")