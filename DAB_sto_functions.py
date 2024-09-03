"""Functions for the integrated transformer optimization."""
# python libraries
import shutil
import os
import csv

from typing import List, Tuple
import inspect

import femmt
# 3rd party libraries
import pandas as pd
# femmt libraries
from DAB_sto_dtos import *
import DAB_sto as dab
import femmt.functions_reluctance as fr
import femmt.functions as ff


def stacked_transformer_fem_simulation_from_result_dto(config_dto: DABStoSingleInputConfig,
                                                       dto: DABStoSingleResultFile,
                                                       fem_working_directory: str,
                                                       fundamental_frequency: float,
                                                       time_current_vectors,
                                                       visualize: bool = False,
                                                       show_last_fem_simulation: bool = False):
    """FEM simulation for the integrated transformer from a result DTO."""
    # 1. chose simulation type
    geo = femmt.MagneticComponent(component_type=femmt.ComponentType.IntegratedTransformer,
                                  working_directory=fem_working_directory,
                                  verbosity=femmt.Verbosity.Silent)

    core_dimensions = femmt.dtos.StackedCoreDimensions(core_inner_diameter=dto.core_inner_diameter,
                                                       window_w=dto.window_w,
                                                       window_h_top=dto.window_h_top, window_h_bot=dto.window_h_bot)

    if isinstance(dto.core_material, str):
        material = femmt.Material(dto.core_material)
    else:
        material = dto.core_material

    core = femmt.Core(core_type=femmt.CoreType.Stacked,
                      core_dimensions=core_dimensions,
                      material=material,
                      temperature=config_dto.temperature,
                      frequency=fundamental_frequency,
                      permeability_datasource=femmt.MaterialDataSource.ManufacturerDatasheet,
                      permittivity_datasource=femmt.MaterialDataSource.ManufacturerDatasheet)

    geo.set_core(core)

    # 3. set air gap parameters
    air_gaps = femmt.AirGaps(femmt.AirGapMethod.Stacked, core)
    air_gaps.add_air_gap(femmt.AirGapLegPosition.CenterLeg, dto.air_gap_top, stacked_position=femmt.StackedPosition.Top)
    air_gaps.add_air_gap(femmt.AirGapLegPosition.CenterLeg, dto.air_gap_bot, stacked_position=femmt.StackedPosition.Bot)
    geo.set_air_gaps(air_gaps)

    # 4. set insulations
    insulation = femmt.Insulation()
    insulation.add_core_insulations(0.0015, 0.0015, 0.0015, 0.0015)
    insulation.add_winding_insulations([[0.0002, 0.0002],
                                        [0.0002, 0.0002]])
    geo.set_insulation(insulation)

    winding_window_top, winding_window_bot = femmt.create_stacked_winding_windows(core, insulation)

    vww_top = winding_window_top.split_window(femmt.WindingWindowSplit.NoSplit)
    vww_bot = winding_window_bot.split_window(femmt.WindingWindowSplit.NoSplit)

    # fill_factor = 0.78539
    # 5. set conductor parameters
    primary_litz = ff.litz_database()[dto.primary_litz_wire]
    secondary_litz = ff.litz_database()[dto.secondary_litz_wire]

    winding1 = femmt.Conductor(0, femmt.Conductivity.Copper)
    winding1.set_litz_round_conductor(primary_litz["conductor_radii"], primary_litz["strands_numbers"],
                                      primary_litz["strand_radii"], None, femmt.ConductorArrangement.Square)

    winding2 = femmt.Conductor(1, femmt.Conductivity.Copper)
    winding2.set_litz_round_conductor(secondary_litz["conductor_radii"], secondary_litz["strands_numbers"],
                                      secondary_litz["strand_radii"], None, femmt.ConductorArrangement.Square)

    # 6. add conductor to vww and add winding window to MagneticComponent
    vww_top.set_interleaved_winding(winding1, dto.n_p_top, winding2, 0,
                                    femmt.InterleavedWindingScheme.HorizontalAlternating)
    vww_bot.set_interleaved_winding(winding1, dto.n_p_bot, winding2, dto.n_s_bot,
                                    femmt.InterleavedWindingScheme.HorizontalAlternating)

    geo.set_winding_windows([winding_window_top, winding_window_bot])

    # 8. start simulation with given frequency, currents and phases
    geo.create_model(freq=fundamental_frequency, pre_visualize_geometry=visualize)

    geo.stacked_core_study(number_primary_coil_turns=dto.n_p_top, time_current_vectors=time_current_vectors,
                           plot_waveforms=False, fft_filter_value_factor=0.05, show_last_fem_simulation=show_last_fem_simulation)

    difference_l_h = 669e-6 - geo.L_h_conc
    difference_l_s = 125e-6 - geo.L_s_conc
    deviation = 100 * (abs(difference_l_h / 669e-6) +
                       abs(difference_l_s / 125e-6))
    print(f'geo.L_h_conc:{geo.L_h_conc} H,\ngeo.L_s_conc:{geo.L_s_conc} H,\ndeviation: {deviation} %')
    print(f'p_hyst: {dto.p_hyst} W,\nn_target: {dto.n_p_bot/dto.n_s_bot}, n_conc: {geo.n_conc}')

    # Define the directory and file path
    directory = os.path.join(os.path.dirname(__file__), "example_results",
                             f'optuna_stacked_transformer_optimization_flxcore')
    csv_file = os.path.join(directory, 'output_data.csv')

    # Ensure the directory exists; if not, create it
    os.makedirs(directory, exist_ok=True)

    # Define the headers and data to save
    headers = ['case_number',
               'l_s_conc',
               'l_h_conc',
               'deviation',
               'n_conc',
               'n_target',
               'RM_p_hyst',
               'RM_p_winding',
               'volume']
    data = [dto.case,
            geo.L_s_conc,
            geo.L_h_conc,
            deviation,
            geo.n_conc,
            (dto.n_p_bot/dto.n_s_bot),
            dto.p_hyst,
            (dto.primary_litz_wire_loss+dto.secondary_litz_wire_loss),
            dto.core_2daxi_total_volume]

    # Check if file exists to decide whether to write headers
    file_exists = os.path.isfile(csv_file)

    # Open the file in append mode to preserve previous data
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write headers only if the file does not already exist
        if not file_exists:
            writer.writerow(headers)

        # Write the new row of data
        writer.writerow(data)

    print(f'Data saved to {csv_file}')


    return geo


def stacked_transformer_fem_simulations_from_result_dtos(config_dto: DABStoSingleInputConfig,
                                                         simulation_dto_list: List[DABStoSingleResultFile],
                                                         max_loss: int,
                                                         visualize: bool = False,
                                                         ):
    """FEM simulation for the integrated transformer from a result DTO."""
    ito_target_and_fixed_parameters_dto = dab.DABStackedTransformerOptimization.calculate_fix_parameters(
        config_dto)

    # time_extracted, current_extracted_1_vec = fr.time_vec_current_vec_from_time_current_vec(
    #     config_dto.time_current_1_vec)
    # time_extracted, current_extracted_2_vec = fr.time_vec_current_vec_from_time_current_vec(
    #     config_dto.time_current_2_vec)

    fundamental_frequency = 200000

    waveforms = pd.read_csv('currents_shifted.csv', delimiter=',')
    time = waveforms['# t'].to_numpy() - waveforms['# t'][0]
    i_ls = waveforms['i_Ls'].to_numpy() - np.mean(waveforms['i_Ls'])
    i_hf2 = waveforms['i_HF2'].to_numpy() - np.mean(waveforms['i_HF2'])
    time_current_vectors = [[time, i_ls], [time, -i_hf2]]

    # phase_deg_1, phase_deg_2 = fr.phases_deg_from_time_current(time_extracted, current_extracted_1_vec,
    #                                                            current_extracted_2_vec)
    # i_peak_1, i_peak_2 = fr.max_value_from_value_vec(current_extracted_1_vec, current_extracted_2_vec)

    for count, dto in enumerate(simulation_dto_list):
        if dto.total_loss < max_loss:
            print(f"FEM simulation {count} of {len(simulation_dto_list)}, case_{dto.case}")
            try:
                stacked_transformer_fem_simulation_from_result_dto(
                    config_dto=config_dto,
                    dto=dto,
                    fem_working_directory=ito_target_and_fixed_parameters_dto.working_directories.fem_working_directory,
                    fundamental_frequency=fundamental_frequency,
                    time_current_vectors=time_current_vectors,
                    visualize=visualize)

                source_json_file = os.path.join(
                    ito_target_and_fixed_parameters_dto.working_directories.fem_working_directory,
                    "results", "log_electro_magnetic.json")
                destination_json_file = os.path.join(
                    ito_target_and_fixed_parameters_dto.working_directories.fem_simulation_results_directory,
                    f'case_{dto.case}.json')

                shutil.copy(source_json_file, destination_json_file)

            except Exception as e:
                print(f"Exception: {e}")

