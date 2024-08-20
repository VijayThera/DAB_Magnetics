"""DTOs for the stacked transformer optimization."""
# python libraries
from dataclasses import dataclass
from typing import List, Union

# 3rd party libraries
import numpy as np
from materialdatabase.dtos import MaterialCurve
from femmt.enumerations import *


@dataclass
class DABStoInsulation:
    """Insulation definition for stacked transformer optimization."""

    iso_top_core: float
    iso_bot_core: float
    iso_left_core_min: float
    iso_right_core: float
    iso_primary_to_primary: float
    iso_secondary_to_secondary: float
    iso_primary_to_secondary: float
    iso_primary_inner_bobbin: float


@dataclass
class DABStoSingleInputConfig:
    """
    Configuration to simulate a stacked transformer.

    Input parameters are the target parameters, current vectors and the parameters to sweep.
    Also specifies the working directory where to store the results.
    """

    # target parameters
    l_s_target: float
    l_h_target: float
    target_inductance_percent_tolerance: int
    n_target: float

    # operating point: current waveforms and temperature
    time_current_1_vec: np.ndarray
    time_current_2_vec: np.ndarray

    temperature: float
    frequency: int
    air_gap_min: float
    air_gap_max:float

    # sweep parameters: geometry and materials
    material_list: list
    core_inner_diameter_min_max_list: list
    window_w_min_max_list: list
    window_h_bot_min_max_list: list
    window_h_top_min_max_list: list

    # conductor
    primary_litz_wire_list: list
    # N_p_top_min_max_list: list
    # N_p_bot_min_max_list: list
    # primary_fill_factor: float

    secondary_litz_wire_list: list
    # N_s_top_min_max_list: list
    # N_s_bot_min_max_list: list
    # secondary_fill_factor: float

    # maximum limitation for transformer total height and core volume
    max_transformer_total_height: float
    max_core_volume: float

    # fix parameters: insulations
    insulations: DABStoInsulation

    # misc
    working_directory: str
    fft_filter_value_factor: float
    mesh_accuracy: float

    # data sources
    permeability_datasource: MaterialDataSource
    permeability_datatype: MeasurementDataType
    permeability_measurement_setup: MeasurementSetup
    permittivity_datasource: MaterialDataSource
    permittivity_datatype: MeasurementDataType
    permittivity_measurement_setup: MeasurementSetup


@dataclass
class ThermalConfig:
    """Thermal configuration file for the stacked transformer optimization."""

    thermal_conductivity_dict: dict
    case_gap_top: float
    case_gap_right: float
    case_gap_bot: float
    boundary_temperatures: dict
    boundary_flags: dict


@dataclass
class WorkingDirectories:
    """Working directories for an integrated transformer optimization."""

    fem_working_directory: str
    reluctance_model_results_directory: str
    fem_simulation_results_directory: str
    fem_simulation_filtered_results_directory: str
    fem_thermal_simulation_results_directory: str
    fem_thermal_filtered_simulation_results_directory: str


@dataclass
class DABStoTargetAndFixedParameters:
    """
    Stacked-transformer optimization target and fixed parameters.

    These parameters are calculated from the stacked-transformer input configuration (StoSingleInputConfig).
    """

    i_rms_1: float
    i_rms_2: float
    # i_peak_1: float
    # i_peak_2: float
    # i_phase_deg_1: float
    # i_phase_deg_2: float
    material_dto_curve_list: List[MaterialCurve]
    time_extracted_vec: List
    current_extracted_top_vec: List
    current_extracted_bot_vec: List
    fundamental_frequency: float
    target_inductance_matrix: np.ndarray
    working_directories: WorkingDirectories


@dataclass
class DABCurrentWorkingPoint:
    """Stores the working point of currents together with a human-readable name."""

    name: str
    time_current_1_vec: Union[np.ndarray, list]
    time_current_2_vec: Union[np.ndarray, list]

@dataclass
class DABStoSingleResultFile:
    """
    Dataclass to store the reluctance model simulation results.

    Contains concrete geometry parameters as well as the calculated results.
    """

    case: int
    # geometry parameters
    air_gap_top: float
    air_gap_bot: float
    n_p_top: int
    n_p_bot: int
    # n_s_top: int
    n_s_bot: int
    window_h_top: float
    window_h_bot: float
    window_w: float
    core_material: str
    core_inner_diameter: float
    primary_litz_wire: str
    secondary_litz_wire: str

    # reluctance model results
    # flux_top_max: float
    # flux_bot_max: float
    # flux_stray_max: float
    # flux_density_top_max: float
    # flux_density_bot_max: float
    # flux_density_stray_max: float
    p_hyst: float
    primary_litz_wire_loss: float
    secondary_litz_wire_loss: float
    core_2daxi_total_volume: float
    total_loss: float
