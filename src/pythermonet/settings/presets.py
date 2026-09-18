"""Reference values shown as hints when a settings file is missing a field.

All values below are taken verbatim from the `silkeborg_bhe_full_dimensioning_heat`
and `silkeborg_hhe_full_dimensioning_heat` example projects. They are illustrative
reference points from real projects, not universal engineering constants — see
`PRESETS_SOURCE_NOTE`.
"""

from __future__ import annotations

from pythermonet.components.distribution_network import DistributionNetworkParameters
from pythermonet.components.ground_loads import HeatPumpPeakSupplyParameters
from pythermonet.components.heat_pump import BrineTemperatureLimits
from pythermonet.components.pipe_infrastructure import HHEFieldParameters
from pythermonet.components.vhe_field import VHEFieldParameters
from pythermonet.core.annulus import Annulus
from pythermonet.core.heat_carrier import HeatCarrier
from pythermonet.core.material import Material
from pythermonet.core.pipe_segment import PipeSegmentParameters
from pythermonet.core.soil import Soil
from pythermonet.dimensioning.sizing_parameters import SizingParameters

PRESETS_SOURCE_NOTE = (
    "values from the Silkeborg BHE and HHE example projects "
    "(silkeborg_bhe_full_dimensioning_heat, silkeborg_hhe_full_dimensioning_heat), "
    "not universal constants"
)

MATERIAL_PRESETS: dict[str, Material] = {
    "pipe_material_dist": Material(
        density=975,
        specific_heat=1900,
        thermal_conductivity=0.4,
    ),
    "grout": Material(
        density=1500,
        specific_heat=2000,
        thermal_conductivity=1.75,
    ),
    "pipe_material_bhe": Material(
        density=1000,
        specific_heat=2000,
        thermal_conductivity=0.4,
    ),
    "pipe_material_hhe": Material(
        density=975,
        specific_heat=1900,
        thermal_conductivity=0.4,
    ),
}

SOIL_PRESETS: dict[str, Soil] = {
    "soil": Soil(
        density=2500,
        specific_heat=1000,
        thermal_conductivity=2.36,
        thermal_conductivity_shallow_heating=1.25,
        thermal_conductivity_shallow_cooling=1.25,
        geothermal_heat_flux=0.0185,
        temperature_surface_mean=9.03,
        temperature_surface_amplitude=7.9,
    ),
}

HEAT_CARRIER_PRESETS: dict[str, HeatCarrier] = {
    "brine": HeatCarrier(
        density=965,
        specific_heat=4450,
        thermal_conductivity=0.45,
        dynamic_viscosity=5e-3,
    ),
}

DISTRIBUTION_NETWORK_PARAMETERS_PRESETS: dict[str, DistributionNetworkParameters] = {
    "distribution_network_parameters": DistributionNetworkParameters(
        roughness=1e-6,
        burial_depth=1.2,
        pipe_spacing=0.3,
        n_pipes_parallel=2,
    ),
}

HEAT_PUMP_PEAK_SUPPLY_PARAMETERS_PRESETS: dict[str, HeatPumpPeakSupplyParameters] = {
    "heat_pump_peak_supply_parameters": HeatPumpPeakSupplyParameters(
        peak_hours_heating=4.0,
        peak_fraction_heating_mode="incremental",
        peak_fraction_heating=1.0,
        peak_hours_cooling=4.0,
        peak_fraction_cooling_mode="incremental",
        peak_fraction_cooling=1.0,
    ),
}

SIZING_PARAMETERS_PRESETS: dict[str, SizingParameters] = {
    "sizing_parameters": SizingParameters(thermal_dimensioning_lifetime=30.0),
}

BRINE_TEMPERATURE_LIMITS_PRESETS: dict[str, BrineTemperatureLimits] = {
    "brine_temperature_limits": BrineTemperatureLimits(
        temperature_brine_min_heating=-3.0,
        temperature_brine_max_cooling=20.0,
    ),
}

PIPE_SEGMENT_PARAMETERS_PRESETS: dict[str, PipeSegmentParameters] = {
    "pipe_segment_hhe": PipeSegmentParameters(
        diameter_outer=0.040,
        sdr=17.0,
        roughness=1e-6,
    ),
    "pipe_segment_bhe": PipeSegmentParameters(
        diameter_outer=0.040,
        sdr=11.0,
        roughness=1e-6,
    ),
}

HHE_FIELD_PARAMETERS_PRESETS: dict[str, HHEFieldParameters] = {
    "pipe_infrastructure_hhe": HHEFieldParameters(
        n_pipes_parallel=20,
        pipe_spacing=1.5,
        burial_depth=1.2,
        length_element=100.0,
    ),
}

ANNULUS_PRESETS: dict[str, Annulus] = {
    "borehole": Annulus(diameter_outer=0.152, sdr=1000.0),
}

VHE_FIELD_PARAMETERS_PRESETS: dict[str, VHEFieldParameters] = {
    "vhe_field_parameters": VHEFieldParameters(
        shank_spacing=0.015 + 2 * 0.02,
        burial_depth=1.0,
        tilt_rad=0.0,
        orientation_rad=0.0,
        heat_exchanger_type="1U",
    ),
}

PRESETS_BY_TYPE: dict[type, dict[str, object]] = {
    Material: MATERIAL_PRESETS,
    Soil: SOIL_PRESETS,
    HeatCarrier: HEAT_CARRIER_PRESETS,
    DistributionNetworkParameters: DISTRIBUTION_NETWORK_PARAMETERS_PRESETS,
    HeatPumpPeakSupplyParameters: HEAT_PUMP_PEAK_SUPPLY_PARAMETERS_PRESETS,
    SizingParameters: SIZING_PARAMETERS_PRESETS,
    BrineTemperatureLimits: BRINE_TEMPERATURE_LIMITS_PRESETS,
    PipeSegmentParameters: PIPE_SEGMENT_PARAMETERS_PRESETS,
    HHEFieldParameters: HHE_FIELD_PARAMETERS_PRESETS,
    Annulus: ANNULUS_PRESETS,
    VHEFieldParameters: VHE_FIELD_PARAMETERS_PRESETS,
}
