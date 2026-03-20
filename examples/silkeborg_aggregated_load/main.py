from __future__ import annotations

from pathlib import Path

import numpy as np

from pythermonet.core.material import Material
from pythermonet.core.heat_carrier import HeatCarrier
from pythermonet.core.soil import Soil
from pythermonet.components.vhe_field import VHEField
from pythermonet.core.annulus import Annulus
from pythermonet.core.pipe_segment import PipeSegment
from pythermonet.components.aggregated_heat_pumps import AggregatedHeatPumps
from pythermonet.input.read_aggregated_load import read_aggregated_load_tsv
from pythermonet.input.read_dimensioned_topology import read_dimensioned_topology_tsv_to_hydraulic
from pythermonet.simulation.run_distribution_pipe_thermal_model import compute_distribution_pipe_thermal_capacity
from pythermonet.dimensioning.borehole_length import size_borehole_length_heating_cooling, apply_annual_balance
from pythermonet.dimensioning.system_temperatures import compute_system_brine_temperatures

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
agg_load_file  = PROJECT_DIR / "data/silkeborg_aggregated_load_heat.dat"
topology_file  = PROJECT_DIR / "data/silkeborg_topology_dimensioned_heat.dat"

# -----------------------------------------------------------------------------
# 1) Materials, brine, soil
# -----------------------------------------------------------------------------
pipe_material = Material(rho=975, c=1900, thermalCond=0.4)

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
# 2) Distribution network (pre-sized, no hydraulic dimensioning)
# -----------------------------------------------------------------------------
network, hydraulic = read_dimensioned_topology_tsv_to_hydraulic(
    topology_file,
    pipe_material=pipe_material,
    brine=brine,
    burial_depth=1.2,
    pipe_distance=0.3,
    n_parallel_pipes=2,
)

# -----------------------------------------------------------------------------
# 3) Borehole field
# -----------------------------------------------------------------------------
n_boreholes   = 6
spacing_m     = 15.0
coordinates   = [[0.0, i * spacing_m] for i in range(n_boreholes)]

pipe_material_bhe = Material(rho=1000, c=2e3, thermalCond=0.4)
grout         = Material(rho=1500, c=2e3, thermalCond=1.75)
borehole      = Annulus(outerDiameter=0.152, SDR=1000.0)
upipe         = PipeSegment(
    outerDiameter=0.04,
    SDR=11.0,
    material=pipe_material_bhe,
    roughnessHeight=1e-6,
    ID=0,
    length=100,   # placeholder
)

BHEfield = VHEField(
    ID=1,
    HE="1U",
    pipe=upipe,
    borehole=borehole,
    grout=grout,
    coordinates=coordinates,
    shankSpacing=0.015 + 2 * 0.02,
    H_m=120.0,
    D_m=1.0,
    r_b_m=0.075,
    tilt_rad=0.0,
    orientation_rad=0.0,
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
)

# -----------------------------------------------------------------------------
# 5) Brine temperature limits
# -----------------------------------------------------------------------------
T_BRINE_MIN_HEAT = -3.0    # HP evaporator inlet limit [°C]
T_BRINE_MAX_COOL = 25.0    # HP condenser inlet limit [°C]

# -----------------------------------------------------------------------------
# 6) Simulation time vector
# -----------------------------------------------------------------------------
times_s = [
    4 * 3600,
    4 * 3600 + 86400 * 365.25 / 4,
    4 * 3600 + 86400 * 365.25 / 4 + 30 * 365.25 * 86400,
]

# -----------------------------------------------------------------------------
# 7) Distribution pipe thermal simulation
# Apply annual balance once — before both the distribution pipe model and
# the BHE sizing.
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
# 8) BHE loads = balanced full loads × (1 − distribution fraction)
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
# 9) Size borehole length
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
# 10) System mean brine temperatures
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
