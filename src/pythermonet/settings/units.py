"""Per-field unit strings for every settings-file-registered domain type.

Load-bearing, not display metadata: `save_settings` cannot write a field's
`{value, unit}` entry without an entry here, and `load_settings` rejects any
file whose declared unit doesn't match. Unit strings are sourced verbatim
from each dataclass's own trailing `# unit` comment. Dimensionless numeric
fields (ratios, counts) and non-physical categorical fields (mode strings,
type tags) both use ``"-"`` — neither carries a real physical unit.
"""

from __future__ import annotations

from pythermonet.components.distribution_network import DistributionNetworkParameters
from pythermonet.components.ground_loads import HeatPumpPeakSupplyParameters
from pythermonet.components.heat_pump import BrineTemperatureLimits
from pythermonet.components.pipe_infrastructure import PipeInfrastructureParameters
from pythermonet.components.vhe_field import VHEFieldParameters
from pythermonet.core.annulus import Annulus
from pythermonet.core.heat_carrier import HeatCarrier
from pythermonet.core.material import Material
from pythermonet.core.pipe_segment import PipeSegmentParameters
from pythermonet.core.soil import Soil
from pythermonet.dimensioning.sizing_parameters import SizingParameters

UNITS_BY_TYPE: dict[type, dict[str, str]] = {
    Material: {
        "density": "kg/m^3",
        "specific_heat": "J/kg/K",
        "thermal_conductivity": "W/m/K",
    },
    Soil: {
        "density": "kg/m^3",
        "specific_heat": "J/kg/K",
        "thermal_conductivity": "W/m/K",
        "thermal_conductivity_shallow_heating": "W/m/K",
        "thermal_conductivity_shallow_cooling": "W/m/K",
        "geothermal_heat_flux": "W/m^2",
        "temperature_surface_mean": "degC",
        "temperature_surface_amplitude": "K",
    },
    HeatCarrier: {
        "density": "kg/m^3",
        "specific_heat": "J/kg/K",
        "thermal_conductivity": "W/m/K",
        "dynamic_viscosity": "Pa*s",
    },
    DistributionNetworkParameters: {
        "roughness": "m",
        "burial_depth": "m",
        "pipe_spacing": "m",
        "n_pipes_parallel": "-",
    },
    HeatPumpPeakSupplyParameters: {
        "peak_hours_heating": "h",
        "peak_fraction_heating_mode": "-",
        "peak_fraction_heating": "-",
        "peak_hours_cooling": "h",
        "peak_fraction_cooling_mode": "-",
        "peak_fraction_cooling": "-",
    },
    SizingParameters: {
        "thermal_dimensioning_lifetime": "years",
    },
    BrineTemperatureLimits: {
        "temperature_brine_min_heating": "degC",
        "temperature_brine_max_cooling": "degC",
    },
    PipeSegmentParameters: {
        "diameter_outer": "m",
        "sdr": "-",
        "roughness": "m",
    },
    PipeInfrastructureParameters: {
        "n_pipes_parallel": "-",
        "pipe_spacing": "m",
        "burial_depth": "m",
    },
    Annulus: {
        "diameter_outer": "m",
        "sdr": "-",
    },
    VHEFieldParameters: {
        "shank_spacing": "m",
        "burial_depth": "m",
        "tilt_rad": "rad",
        "orientation_rad": "rad",
        "heat_exchanger_type": "-",
    },
}
