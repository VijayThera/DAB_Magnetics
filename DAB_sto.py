"""stacked transformer optimization."""
# Python libraries
import os
import json
import itertools
import shutil
import dataclasses
from typing import List, Dict, Tuple

# 3rd party library import
import materialdatabase as mdb
import numpy as np
from scipy import optimize
import optuna
import math
import logging
import gmsh
import os
import matplotlib.pyplot as plt

# femmt import
import femmt.functions as ff
import femmt.functions_reluctance as fr
import femmt.optimization.functions_optimization as fo
# from femmt import DABStoSingleResultFile
from femmt.optimization.ito_dtos import *
import femmt.optimization.optuna_femmt_parser as op
import femmt.optimization.ito_functions as itof
import femmt
from DAB_sto_dtos import *
import DAB_sto_functions as dab_func
from femmt.model import *
import magnethub as mh
import pandas as pd
from datetime import datetime


class MyJSONEncoder(json.JSONEncoder):
    """
    Class to transform dicts with numpy arrays to json.

    This class is used as cls=MyJSONEncoder by json.dump

    See Also
    --------
    https://python-forum.io/thread-35245.html
    """

    def default(self, o):
        """Transform the dictionary to a .json file."""
        try:
            return o.tolist()  # works with any object that has .tolist() method
        except AttributeError:
            pass
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, o)


class DABStackedTransformerOptimization:
    """Perform different optimization methods for the stacked transformer."""

    @staticmethod
    def plot(valid_design_list: List[DABStoSingleResultFile]) -> None:
        """
        Plot the pareto diagram out of the reluctance model calculation.

        :param valid_design_list:
        :type valid_design_list: List[DABStoSingleResultFile]
        :return: Plot
        :rtype: None
        """
        volume_list = []
        core_hyst_loss_list = []
        annotation_list = []

        for result in valid_design_list:
            volume_list.append(result.core_2daxi_total_volume)
            core_hyst_loss_list.append(result.total_loss)
            annotation_list.append(result.case)

        fo.plot_2d(volume_list, core_hyst_loss_list, "Volume in m³", "Losses in W", "Pareto Diagram",
                   plot_color="red", annotations=annotation_list)

    @staticmethod
    def calculate_fix_parameters(config: DABStoSingleInputConfig) -> DABStoTargetAndFixedParameters:
        """
        Calculate fix parameters what can be derived from the input configuration.

        return values are:

            i_rms_1
            i_rms_2
            time_extracted_vec
            current_extracted_top_vec
            current_extracted_bot_vec
            material_dto_curve_list
            fundamental_frequency
            target_inductance_matrix
            fem_working_directory
            fem_simulation_results_directory
            reluctance_model_results_directory
            fem_thermal_simulation_results_directory

        :param config: configuration file
        :type config: DABStoSingleInputConfig
        :return: calculated target and fix parameters
        :rtype: ItoTargetAndFixedParameters
        """
        # currents
        time_extracted, current_extracted_top_vec = fr.time_vec_current_vec_from_time_current_vec(
            config.time_current_1_vec)
        time_extracted, current_extracted_bot_vec = fr.time_vec_current_vec_from_time_current_vec(
            config.time_current_2_vec)
        fundamental_frequency = config.frequency  # 1 / time_extracted[-1]

        i_rms_1 = fr.i_rms(config.time_current_1_vec)
        i_rms_2 = fr.i_rms(config.time_current_2_vec)

        # target inductances
        target_inductance_matrix = fr.calculate_inductance_matrix_from_ls_lh_n(config.l_s_target, config.l_h_target,
                                                                               config.n_target)

        # material properties
        material_db = mdb.MaterialDatabase(is_silent=True)

        material_data_list = []
        for material_name in config.material_list:
            material_dto = material_db.material_data_interpolation_to_dto(material_name, fundamental_frequency,
                                                                          config.temperature)
            material_data_list.append(material_dto)

        # set up working directories
        working_directories = itof.set_up_folder_structure(config.working_directory)

        # finalize data to dto
        target_and_fix_parameters = DABStoTargetAndFixedParameters(
            i_rms_1=i_rms_1,
            i_rms_2=i_rms_2,
            time_extracted_vec=time_extracted,
            current_extracted_top_vec=current_extracted_top_vec,
            current_extracted_bot_vec=current_extracted_bot_vec,
            material_dto_curve_list=material_data_list,
            fundamental_frequency=fundamental_frequency,
            target_inductance_matrix=target_inductance_matrix,
            working_directories=working_directories
        )
        # print(
        #     f'time_extracted:{time_extracted}\ncurrent_extracted_top_vec:{current_extracted_top_vec}\ncurrent_extracted_bot_vec:{current_extracted_bot_vec}')

        return target_and_fix_parameters

    class ReluctanceModel:
        """Create and calculate the reluctance model for the Stacked transformer."""

        class NSGAII:
            """NSGAII algorithm to find the pareto front."""

            ##############################
            # simulation
            ##############################

            @staticmethod
            def objective(trial, config: DABStoSingleInputConfig,
                          target_and_fixed_parameters: DABStoTargetAndFixedParameters,
                          standard_core: bool = False):
                """
                Objective function to optimize.

                Using optuna. Some hints:

                 * returning failed trails by using return float('nan'), float('nan'),
                   see https://optuna.readthedocs.io/en/stable/faq.html#how-are-nans-returned-by-trials-handled
                 * speed up the search for NSGA-II algorithm with dynamic alter the search space, see https://optuna.readthedocs.io/en/stable/faq.html#id10


                :param standard_core:
                :param trial: parameter suggesting by optuna
                :param config: input configuration file
                :type config: DABStoSingleInputConfig
                :param target_and_fixed_parameters: target and fix parameters
                :type target_and_fixed_parameters: DABStoTargetAndFixedParameters
                """
                # pass multiple arguments to the objective function used by optuna
                # https://www.kaggle.com/general/261870

                #########################################################
                # set core geometry optimization parameters
                #########################################################

                core_inner_diameter = trial.suggest_categorical('core_inner_diameter',
                                                                config.core_inner_diameter_min_max_list)

                # dictionary mapping core_inner_diameter to window_w
                core_inner_diameter_to_window_w_map = {
                    12e-3: (22.5 - 12) / 2 * 1e-3,
                    13.45e-3: (27.5 - 13.45) / 2 * 1e-3,
                    14.35e-3: (32 - 14.35) / 2 * 1e-3,
                    14.9e-3: (37 - 14.9) / 2 * 1e-3
                }
                # Retrieve window_w if core_inner_diameter matches any key
                window_w = core_inner_diameter_to_window_w_map.get(core_inner_diameter)
                # window_w = trial.suggest_categorical('window_w', config.window_w_min_max_list)

                # dictionary mapping core_inner_diameter to window_h
                core_inner_diameter_to_window_h_map = {
                    12e-3: 16.1e-3,
                    13.45e-3: 11.5e-3,
                    14.35e-3: 25e-3,
                    14.9e-3: 20e-3
                }
                window_h = core_inner_diameter_to_window_h_map.get(core_inner_diameter)

                window_h_bot = trial.suggest_float('window_h_bot', config.window_h_bot_min_max_list[0] * window_h,
                                                   config.window_h_bot_min_max_list[1] * window_h)

                # core_inner_diameter = trial.suggest_float('core_inner_diameter', 0.015, 0.030)
                # window_w = trial.suggest_float('window_w', 0.008, 0.020)
                # window_h_bot = trial.suggest_float('window_h_bot', 0.010, 0.040)

                material = trial.suggest_categorical('material', config.material_list)
                primary_litz_wire = trial.suggest_categorical('primary_litz_wire', config.primary_litz_wire_list)
                secondary_litz_wire = trial.suggest_categorical('secondary_litz_wire', config.secondary_litz_wire_list)

                L_s = config.l_s_target
                L_h = config.l_h_target

                # cross-section comparison is according to a square for round wire.
                # this approximation is more realistic
                # insulation
                insulation_distance = 1.5e-3
                # insulation_cross_section_top = 2 * insulation_distance * (window_w + window_h_top)
                insulation_cross_section_bot = 2 * insulation_distance * (window_w + window_h_bot)

                litz_database = ff.litz_database()

                primary_litz = litz_database[primary_litz_wire]
                secondary_litz = litz_database[secondary_litz_wire]

                # total_available_window_cross_section_top = window_h_top * window_w - insulation_cross_section_top
                total_available_window_cross_section_bot = window_h_bot * window_w - insulation_cross_section_bot

                #########################################################
                # set dynamic wire count parameters as optimization parameters
                #########################################################
                # set the winding search space dynamic
                # https://optuna.readthedocs.io/en/stable/faq.html#what-happens-when-i-dynamically-alter-a-search-space

                # n_p_top suggestion
                # n_p_top_max = total_available_window_cross_section_top / (2 * primary_litz['conductor_radii']) ** 2
                # if n_p_top_max < 0:
                #     n_p_top_max = 0
                n_p_top = trial.suggest_int('n_p_top', 0, config.n_p_top_max)
                n_s_top = 0

                avaliable_window_w = window_w - 2 * insulation_distance
                turns_one_row_top = math.floor(avaliable_window_w / (2 * primary_litz['conductor_radii'] + 0.0002))
                turns_one_col_top = math.ceil(n_p_top / turns_one_row_top)
                window_h_top = turns_one_col_top * (
                        2 * primary_litz['conductor_radii'] + 0.0002) + 2 * insulation_distance

                # n_p_bot suggestion
                # n_bot_max = total_available_window_cross_section_bot / (2 * primary_litz['conductor_radii']) ** 2
                # if n_bot_max < 0:
                #     n_bot_max = 0
                n_p_bot = trial.suggest_int('n_p_bot', 0, config.n_p_bot_max)
                n_s_bot = round(n_p_bot / config.n_target)

                conductor_radii = max(primary_litz['conductor_radii'], secondary_litz['conductor_radii'])
                turns_one_row_bot = math.floor(avaliable_window_w / (2 * conductor_radii + 0.0002))
                turns_one_col_bot = math.ceil((n_p_bot + n_s_bot) / turns_one_row_bot)
                window_h_bot_min = turns_one_col_bot * (2 * conductor_radii + 0.0002) + 2 * insulation_distance

                if (total_available_window_cross_section_bot >=
                    (n_p_bot * ((2 * primary_litz['conductor_radii']) ** 2)) +
                    (n_s_bot * ((2 * secondary_litz['conductor_radii']) ** 2))) and window_h_bot > window_h_bot_min:
                    for material_dto in target_and_fixed_parameters.material_dto_curve_list:
                        if material_dto.material_name == material:
                            material_data = material_dto

                        material_mu_r_initial = material_data.material_mu_r_abs
                        flux_density_data_vec = material_data.material_flux_density_vec
                        mu_r_imag_data_vec = material_data.material_mu_r_imag_vec

                        core_top_bot_height = core_inner_diameter / 4
                        core_cross_section = (core_inner_diameter / 2) ** 2 * np.pi

                        t2_winding_matrix = [[n_p_top, n_s_top], [n_p_bot, n_s_bot]]

                        target_inductance_matrix = fr.calculate_inductance_matrix_from_ls_lh_n(L_s,
                                                                                               L_h,
                                                                                               config.n_target)
                        t2_reluctance_matrix = fr.calculate_reluctance_matrix(t2_winding_matrix,
                                                                              target_inductance_matrix)

                        # print(f't2_reluctance_matrix: {t2_reluctance_matrix}')

                        core_2daxi_total_volume = fr.calculate_core_2daxi_total_volume(core_inner_diameter,
                                                                                       (
                                                                                               window_h_bot + window_h_top + core_inner_diameter / 4),
                                                                                       window_w)

                        if np.linalg.det(t2_reluctance_matrix) != 0 and np.linalg.det(
                                np.transpose(t2_winding_matrix)) != 0 and np.linalg.det(target_inductance_matrix) != 0:

                            r_center_topcore = fr.r_core_round(core_inner_diameter, window_h_top,
                                                               material_data.material_mu_r_abs)

                            r_top_topcore = fr.r_core_top_bot_radiant(core_inner_diameter, window_w,
                                                                      material_data.material_mu_r_abs,
                                                                      core_top_bot_height)

                            r_center_botcore = fr.r_core_round(core_inner_diameter, window_h_bot,
                                                               material_data.material_mu_r_abs)

                            r_top_botcore = fr.r_core_top_bot_radiant(core_inner_diameter, window_w,
                                                                      material_data.material_mu_r_abs,
                                                                      core_top_bot_height)

                            r_middle_target = -t2_reluctance_matrix[0][1]
                            r_top_target = t2_reluctance_matrix[0][0] - r_middle_target  # - r_top_topcore
                            r_bot_target = t2_reluctance_matrix[1][1] - r_middle_target  # - r_top_botcore

                            r_air_gap_top_target = r_top_target - 2 * r_center_topcore - r_top_topcore
                            r_air_gap_bot_target = r_bot_target - 2 * r_center_botcore - r_top_botcore

                            if r_air_gap_top_target > 0 and r_air_gap_bot_target > 0:
                                # Note: a minimum air gap length of zero is not allowed. This will lead to failure calculation
                                # when trying to solve (using brentq) r_gap_round_round-function. Calculating an air gap
                                # reluctance with length of zero is not realistic.
                                minimum_air_gap_length = config.air_gap_min
                                maximum_air_gap_length = config.air_gap_max
                                minimum_sort_out_air_gap_length = 0
                                try:
                                    # solving brentq needs to be in try/except statement,
                                    # as it can be that there is no sign changing in the given interval
                                    # to search for the zero.
                                    # Note: setting full output to true and taking object [0] is only
                                    # to avoid linting error!
                                    l_top_air_gap = optimize.brentq(fr.r_air_gap_round_inf_sct, minimum_air_gap_length,
                                                                    maximum_air_gap_length,
                                                                    args=(core_inner_diameter, window_h_top,
                                                                          r_air_gap_top_target), full_output=True)[0]

                                    l_bot_air_gap = \
                                        optimize.brentq(fr.r_air_gap_round_round_sct, minimum_air_gap_length,
                                                        maximum_air_gap_length,
                                                        args=(core_inner_diameter, window_h_bot / 2,
                                                              window_h_bot / 2, r_air_gap_bot_target),
                                                        full_output=True)[0]

                                except ValueError:
                                    # ValueError is raised in case of an air gap with length of zero
                                    return float('nan'), float('nan')

                                if l_top_air_gap > core_inner_diameter or l_bot_air_gap > core_inner_diameter:
                                    return float('nan'), float('nan')

                                if l_top_air_gap >= minimum_sort_out_air_gap_length and l_bot_air_gap >= minimum_sort_out_air_gap_length and n_p_bot / n_s_bot > 4.2:

                                    # =======================================================================================
                                    mdl = mh.loss.LossModel(material='3C95', team='paderborn')

                                    waveforms = pd.read_csv('currents_shifted.csv')
                                    times = waveforms['# t'].to_numpy() - waveforms['# t'][0]
                                    i_ls = waveforms['i_Ls'].to_numpy() - np.mean(waveforms['i_Ls'])
                                    i_hf2 = waveforms['i_HF2'].to_numpy() - np.mean(waveforms['i_HF2'])
                                    # step_size = round(len(times) / 1024)
                                    # i_ls_sampled = np.array(i_ls[::step_size][:1024])
                                    # i_hf2_sampled = np.array(i_hf2[::step_size][:1024])

                                    # i_matrix = np.array([i_ls_sampled, i_hf2_sampled])
                                    i_matrix = np.array([i_ls, -i_hf2])
                                    flux_matrix = fr.calculate_flux_matrix(reluctance_matrix=abs(t2_reluctance_matrix),
                                                                           winding_matrix=t2_winding_matrix,
                                                                           current_matrix=i_matrix)
                                    flux_top = flux_matrix[0]
                                    flux_bot = flux_matrix[1]

                                    # flux_top = (n_p_top * i_ls_sampled) / t2_reluctance_matrix[0][0]
                                    # flux_bot = ((n_p_bot * i_ls_sampled - n_s_bot * i_hf2_sampled) /
                                    #             t2_reluctance_matrix[1][1])
                                    flux_mid = flux_top + flux_bot

                                    B_top = flux_top / core_cross_section
                                    B_bot = flux_bot / core_cross_section
                                    B_mid = flux_mid / core_cross_section

                                    step_size = round(len(B_top) / 1024)

                                    B_top_sampled = np.array(B_top[::step_size][:1024])
                                    B_bot_sampled = np.array(B_bot[::step_size][:1024])
                                    B_mid_sampled = np.array(B_mid[::step_size][:1024])

                                    p_hyst_den_top, _ = mdl(B_top_sampled, config.frequency, config.temperature)
                                    p_hyst_den_bot, _ = mdl(B_bot_sampled, config.frequency, config.temperature)
                                    p_hyst_den_mid, _ = mdl(B_mid_sampled, config.frequency, config.temperature)

                                    core_width = fr.calculate_r_outer(core_inner_diameter, window_w)
                                    inner_leg_width = core_inner_diameter / 2

                                    core_height_top = window_h_top + core_inner_diameter / 4
                                    core_height_bot = window_h_bot + core_inner_diameter / 4

                                    air_gap_volume_top = np.pi * inner_leg_width ** 2 * l_top_air_gap
                                    air_gap_volume_bot = np.pi * inner_leg_width ** 2 * l_bot_air_gap

                                    core_volume_top = (np.pi * (core_width ** 2 * core_height_top -
                                                                (inner_leg_width + window_w) ** 2 * window_h_top +
                                                                inner_leg_width ** 2 * window_h_top) -
                                                       air_gap_volume_top)

                                    core_volume_bot = (np.pi * (core_width ** 2 * core_height_bot -
                                                                (inner_leg_width + window_w) ** 2 * window_h_bot +
                                                                inner_leg_width ** 2 * window_h_bot) -
                                                       air_gap_volume_bot)

                                    core_volume_mid = np.pi * core_width ** 2 * (core_inner_diameter / 4)

                                    p_hyst_top = p_hyst_den_top * core_volume_top
                                    p_hyst_bot = p_hyst_den_bot * core_volume_bot
                                    p_hyst_mid = p_hyst_den_mid * core_volume_mid

                                    p_hyst = p_hyst_top + p_hyst_bot + p_hyst_mid

                                    primary_effective_conductive_cross_section = primary_litz['strands_numbers'] * \
                                                                                 primary_litz[
                                                                                     'strand_radii'] ** 2 * np.pi
                                    primary_effective_conductive_radius = np.sqrt(
                                        primary_effective_conductive_cross_section / np.pi)
                                    primary_resistance = fr.resistance_solid_wire(core_inner_diameter, window_w,
                                                                                  n_p_top + n_p_bot,
                                                                                  primary_effective_conductive_radius,
                                                                                  material='Copper')
                                    # print(primary_resistance)
                                    primary_dc_loss = primary_resistance * target_and_fixed_parameters.i_rms_1 ** 2

                                    secondary_effective_conductive_cross_section = secondary_litz['strands_numbers'] * \
                                                                                   secondary_litz[
                                                                                       'strand_radii'] ** 2 * np.pi
                                    secondary_effective_conductive_radius = np.sqrt(
                                        secondary_effective_conductive_cross_section / np.pi)
                                    secondary_resistance = fr.resistance_solid_wire(core_inner_diameter, window_w,
                                                                                    n_s_top + n_s_bot,
                                                                                    secondary_effective_conductive_radius,
                                                                                    material='Copper')
                                    # print(secondary_resistance)
                                    secondary_dc_loss = secondary_resistance * target_and_fixed_parameters.i_rms_2 ** 2

                                    total_loss = p_hyst + primary_dc_loss + secondary_dc_loss
                                    trial.set_user_attr('window_w', window_w)
                                    trial.set_user_attr('window_h_top', window_h_top)
                                    trial.set_user_attr('n_s_bot', n_s_bot)
                                    trial.set_user_attr('air_gap_top', l_top_air_gap)
                                    trial.set_user_attr('air_gap_bot', l_bot_air_gap)

                                    trial.set_user_attr('flux_top_max', np.nanmax(flux_top))
                                    trial.set_user_attr('flux_bot_max', np.nanmax(flux_bot))
                                    trial.set_user_attr('flux_mid_max', np.nanmax(flux_mid))
                                    trial.set_user_attr('flux_density_top_max', np.nanmax(B_top))
                                    trial.set_user_attr('flux_density_bot_max', np.nanmax(B_bot))
                                    trial.set_user_attr('flux_density_mid_max', np.nanmax(B_mid))
                                    trial.set_user_attr('p_hyst', p_hyst)
                                    trial.set_user_attr('primary_litz_wire_loss', primary_dc_loss)
                                    trial.set_user_attr('secondary_litz_wire_loss', secondary_dc_loss)

                                    # print(f'STO: successfully calculated trial {trial.number}')
                                    return core_2daxi_total_volume, total_loss
                                else:
                                    return float('nan'), float('nan')
                            else:
                                return float('nan'), float('nan')
                        else:
                            return float('nan'), float('nan')
                else:
                    return float('nan'), float('nan')

            @staticmethod
            def start_study(study_name: str, config: DABStoSingleInputConfig, number_trials: int,
                            storage: str = None) -> None:
                """
                Start a study to optimize an Stacked transformer.

                Note: Due to performance reasons, the study is calculated in RAM.
                After finishing the study, the results are copied to sqlite or mysql database by the use of a new study.

                :param study_name: Name of the study
                :type study_name: str
                :param config: simulation configuration
                :type config: DABStoSingleInputConfig
                :param number_trials: number of trials
                :type number_trials: int
                :param storage: "sqlite" or "mysql"
                :type storage: str
                """
                if os.path.exists(f"{config.working_directory}/study_{study_name}.sqlite3"):
                    print("Existing study found. Proceeding.")

                # introduce study in storage, e.g. sqlite or mysql
                if storage == 'sqlite':
                    # Note: for sqlite operation, there needs to be three slashes '///' even before the path '/home/...'
                    # Means, in total there are four slashes including the path itself '////home/.../database.sqlite3'
                    storage = f"sqlite:///{config.working_directory}/study_{study_name}.sqlite3"
                elif storage == 'mysql':
                    storage = "mysql://monty@localhost/mydb",

                # set logging verbosity:
                # https://optuna.readthedocs.io/en/stable/reference/generated/optuna.logging.set_verbosity.html#optuna.logging.set_verbosity
                # .INFO: all messages (default)
                # .WARNING: fails and warnings
                # .ERROR: only errors
                optuna.logging.set_verbosity(optuna.logging.ERROR)

                # calculate the target and fixed parameters
                # and generate the folder structure inside this function
                target_and_fixed_parameters = DABStackedTransformerOptimization.calculate_fix_parameters(config)

                # Wrap the objective inside a lambda and call objective inside it
                func = lambda trial: DABStackedTransformerOptimization.ReluctanceModel.NSGAII.objective(trial,
                                                                                                        config,
                                                                                                        target_and_fixed_parameters,
                                                                                                        False)
                directions = ['minimize', 'minimize']
                sampler = optuna.samplers.NSGAIIISampler()
                study_in_storage = optuna.create_study(directions=directions, study_name=study_name, storage=storage,
                                                       load_if_exists=True, sampler=sampler)

                # Pass func to Optuna studies
                study_in_memory = optuna.create_study(directions=directions, study_name=study_name, sampler=sampler)
                print(f"Sampler is {study_in_memory.sampler.__class__.__name__}")
                study_in_memory.add_trials(study_in_storage.trials)
                study_in_memory.optimize(func, n_trials=number_trials, show_progress_bar=True)
                #n_jobs=-1, gc_after_trial=False,
                study_in_storage.add_trials(study_in_memory.trials[-number_trials:])

                # # fig = optuna.visualization.plot_pareto_front(study, targets=lambda t: (
                # #     t.values[0] if error_difference_inductance_sum_percent > t.values[2] else None,
                # #     t.values[1] if error_difference_inductance_sum_percent > t.values[2] else None),
                # #                                              target_names=["volume in m³", "loss in W"])
                # # if t.values[1] < 2000 else None
                # # in-memory calculation is shown before saving the data to database
                # fig = optuna.visualization.plot_pareto_front(study_in_memory,
                #                                              targets=lambda t: (
                #                                                  t.values[0],
                #                                                  t.values[1]),
                #                                              target_names=["volume", "losses"])
                # fig.show()
                # # Current timestamp
                # timestamp = datetime.now().strftime("%m-%d__%H-%M")
                # # Create a unique filename for the Pareto front plot
                # filename = f"Pareto_Front__Trials-{len(study_in_memory.trials)}__{timestamp}.html"
                # # Specify the directory to save the file
                # save_dir = '../DAB_Magnetics/example_results/pareto'
                # os.makedirs(save_dir, exist_ok=True)
                # # Combine directory and filename
                # file_path = os.path.join(save_dir, filename)
                # fig.write_html(file_path)
                # # Print the file path for reference
                # print('file_path:', file_path)

            @staticmethod
            def proceed_study(study_name: str, config: DABStoSingleInputConfig, number_trials: int) -> None:
                """
                Proceed a study which is stored as sqlite database.

                :param study_name: Name of the study
                :type study_name: str
                :param config: Simulation configuration
                :type config: DABStoSingleInputConfig
                :param number_trials: Number of trials adding to the existing study
                :type number_trials: int
                """
                target_and_fixed_parameters = DABStackedTransformerOptimization.calculate_fix_parameters(
                    config)

                # Wrap the objective inside a lambda and call objective inside it
                func = lambda \
                        trial: DABStackedTransformerOptimization.ReluctanceModel.NSGAII.objective(
                    trial,
                    config,
                    target_and_fixed_parameters)

                study = optuna.create_study(study_name=study_name, storage=f"sqlite:///study_{study_name}.sqlite3",
                                            load_if_exists=True)
                study.optimize(func, n_trials=number_trials)

            @staticmethod
            def show_study_results(study_name: str, config: DABStoSingleInputConfig, losses: int) -> None:
                """
                Show the results of a study.

                :param losses:
                :param study_name: Name of the study
                :type study_name: str
                :param config: stacked transformer configuration file
                :type config: DABStoSingleInputConfig
                """
                study = optuna.create_study(study_name=study_name,
                                            storage=f"sqlite:///{config.working_directory}/study_{study_name}.sqlite3",
                                            load_if_exists=True)

                fig = optuna.visualization.plot_pareto_front(study,
                                                             targets=lambda t: (
                                                                 t.values[0],
                                                                 t.values[1] if t.values[1] < losses else None),
                                                             target_names=["volume", "losses"])
                fig.show()

            ##############################
            # load
            ##############################

            @staticmethod
            def load_study_to_dto(study_name: str, config: DABStoSingleInputConfig) -> List[DABStoSingleResultFile]:
                """
                Load all trials of a study to a DTO-list.

                :param study_name: Name of the study
                :type study_name: str
                :param config: stacked transformer configuration file
                :type config: DABStoSingleInputConfig
                :return: List of all trials
                :rtype: List[DABStoSingleResultFile]

                """
                study = optuna.create_study(study_name=study_name,
                                            storage=f"sqlite:///{config.working_directory}/study_{study_name}.sqlite3",
                                            load_if_exists=True)

                dto_list = [OptunaFemmtParser.parse(frozen_object) for frozen_object in study.trials \
                            if frozen_object.state == optuna.trial.TrialState.COMPLETE]

                return dto_list

            @staticmethod
            def load_study_best_trials_to_dto(study_name: str, config: DABStoSingleInputConfig) -> List[
                DABStoSingleResultFile]:
                """
                Load the best trials (Pareto front) of a study.

                :param study_name: Name of the study
                :type study_name: str
                :param config: stacked transformer configuration file
                :type config: DABStoSingleInputConfig
                :return: List of the best trials.
                :rtype: List[DABStoSingleResultFile]

                """
                study = optuna.create_study(study_name=study_name,
                                            storage=f"sqlite:///{config.working_directory}/study_{study_name}.sqlite3",
                                            load_if_exists=True)

                print(study.best_trials[0])

                dto_list = [OptunaFemmtParser.parse(frozen_object) for frozen_object in study.best_trials]

                return dto_list

        @staticmethod
        def filter_loss_list(valid_design_list: List[DABStoSingleResultFile], factor_min_dc_losses: float = 1.2) -> \
                List[DABStoSingleResultFile]:
            """Remove designs with too high losses compared to the minimum losses."""
            # figure out pareto front
            # pareto_volume_list, pareto_core_hyst_list, pareto_dto_list = self.pareto_front(volume_list, core_hyst_loss_list, valid_design_list)

            x_pareto_vec, y_pareto_vec = fo.pareto_front_from_dtos(valid_design_list)

            vector_to_sort = np.array([x_pareto_vec, y_pareto_vec])

            # sorting 2d array by 1st row
            # https://stackoverflow.com/questions/49374253/sort-a-numpy-2d-array-by-1st-row-maintaining-columns
            sorted_vector = vector_to_sort[:, vector_to_sort[0].argsort()]
            x_pareto_vec = sorted_vector[0]
            y_pareto_vec = sorted_vector[1]

            total_losses_list = []
            filtered_design_dto_list = []

            for dto in valid_design_list:
                total_losses_list.append(dto.total_loss)

            min_total_dc_losses = total_losses_list[np.argmin(total_losses_list)]
            loss_offset = factor_min_dc_losses * min_total_dc_losses

            for dto in valid_design_list:
                ref_loss = np.interp(dto.core_2daxi_total_volume, x_pareto_vec, y_pareto_vec) + loss_offset
                if dto.total_loss < ref_loss:
                    filtered_design_dto_list.append(dto)

            return filtered_design_dto_list

        @staticmethod
        def save_dto_list(result_dto_list: List[DABStoSingleResultFile], filepath: str):
            """
            Save the DABStoSingleResultFile-List to the file structure.

            :param result_dto_list:
            :type result_dto_list: List[DABStoSingleResultFile]
            :param filepath: filepath
            :type filepath: str
            """
            if not os.path.exists(filepath):
                os.mkdir(filepath)

            for _, dto in enumerate(result_dto_list):
                file_name = os.path.join(filepath, f"case_{dto.case}.json")

                result_dict = dataclasses.asdict(dto)
                with open(file_name, "w+", encoding='utf-8') as outfile:
                    json.dump(result_dict, outfile, indent=2, ensure_ascii=False, cls=MyJSONEncoder)

        @staticmethod
        def load_filtered_results(working_directory: str) -> list[DABStoSingleResultFile]:
            """
            Load the results of the reluctance model and returns the DABStoSingleResultFiles as a list.

            :param working_directory: working directory
            :type working_directory: str
            :return: List of DABStoSingleResultFiles
            :rtype: List[DABStoSingleResultFile]
            """
            Stacked_transformer_reluctance_model_results_directory = os.path.join(working_directory,
                                                                                  "01_reluctance_model_results_filtered")
            print(f"Read results from {Stacked_transformer_reluctance_model_results_directory}")
            return DABStackedTransformerOptimization.ReluctanceModel.load_list(
                Stacked_transformer_reluctance_model_results_directory)

        @staticmethod
        def load_list(filepath: str) -> List[DABStoSingleResultFile]:
            """
            Load the list of the reluctance models from the folder structure.

            :param filepath: filepath
            :type filepath: str
            :return: List of DABStoSingleResultFiles
            :rtype: List[DABStoSingleResultFile]
            """
            valid_design_list = []
            for file in os.listdir(filepath):
                if file.endswith(".json"):
                    json_file_path = os.path.join(filepath, file)
                    with open(json_file_path, "r") as fd:
                        loaded_data_dict = json.loads(fd.read())

                    valid_design_list.append(result_file_dict_to_dto(loaded_data_dict))
            if len(valid_design_list) == 0:
                raise ValueError("Specified file path is empty")

            return valid_design_list

    class FemSimulation:
        """Group functions to perform FEM simulations."""

        @staticmethod
        def simulate(config_dto: DABStoSingleInputConfig, simulation_dto_list: List[DABStoSingleResultFile],
                     max_loss: int,
                     visualize: bool = False):
            """Perform the FEM simulation."""
            dab_func.stacked_transformer_fem_simulations_from_result_dtos(config_dto, simulation_dto_list, max_loss,
                                                                          visualize)


def result_file_dict_to_dto(result_file_dict):
    """Translate the result file dictionary to a data transfer object (DTO)."""
    result_file_dto = DABStoSingleResultFile(
        case=result_file_dict["case"],
        air_gap_top=result_file_dict["air_gap_top"],
        air_gap_bot=result_file_dict["air_gap_bot"],
        # air_gap_middle=result_file_dict["air_gap_middle"],
        n_p_top=int(result_file_dict["n_p_top"]),
        n_p_bot=int(result_file_dict["n_p_bot"]),
        # n_s_top=int(result_file_dict["n_s_top"]),
        n_s_bot=int(result_file_dict["n_s_bot"]),
        window_h_top=result_file_dict["window_h_top"],
        window_h_bot=result_file_dict["window_h_bot"],
        window_w=result_file_dict["window_w"],
        core_material=result_file_dict["core_material"],
        core_inner_diameter=result_file_dict["core_inner_diameter"],
        flux_top_max=result_file_dict["flux_top_max"],
        flux_bot_max=result_file_dict["flux_bot_max"],
        flux_mid_max=result_file_dict["flux_mid_max"],
        flux_density_top_max=result_file_dict["flux_density_top_max"],
        flux_density_bot_max=result_file_dict["flux_density_bot_max"],
        flux_density_mid_max=result_file_dict["flux_density_mid_max"],
        p_hyst=result_file_dict["p_hyst"],
        core_2daxi_total_volume=result_file_dict["core_2daxi_total_volume"],
        primary_litz_wire=result_file_dict["primary_litz_wire"],
        secondary_litz_wire=result_file_dict["secondary_litz_wire"],
        primary_litz_wire_loss=result_file_dict["primary_litz_wire_loss"],
        secondary_litz_wire_loss=result_file_dict["secondary_litz_wire_loss"],
        total_loss=result_file_dict["total_loss"]
    )
    return result_file_dto


class OptunaFemmtParser:
    """Parser to bring optuna results to DABStoSingleResultFile format."""

    @staticmethod
    def parse(frozen_trial: optuna.trial.FrozenTrial) -> DABStoSingleResultFile:
        """Parse the optuna trial to DABStoSingleResultFile.

        :param frozen_trial: frozen trial (by optuna)
        :type frozen_trial: optuna.trial.FrozenTrial
        """
        return DABStoSingleResultFile(
            case=frozen_trial.number,
            # geometry parameters
            air_gap_top=frozen_trial.user_attrs["air_gap_top"],
            air_gap_bot=frozen_trial.user_attrs["air_gap_bot"],
            n_p_top=frozen_trial.params["n_p_top"],
            n_p_bot=frozen_trial.params["n_p_bot"],
            n_s_bot=frozen_trial.user_attrs["n_s_bot"],
            window_h_top=frozen_trial.user_attrs["window_h_top"],
            window_h_bot=frozen_trial.params["window_h_bot"],
            window_w=frozen_trial.user_attrs["window_w"],
            core_material=frozen_trial.params["material"],
            core_inner_diameter=frozen_trial.params["core_inner_diameter"],
            primary_litz_wire=frozen_trial.params["primary_litz_wire"],
            secondary_litz_wire=frozen_trial.params["secondary_litz_wire"],

            # reluctance model results
            flux_top_max=frozen_trial.user_attrs["flux_top_max"],
            flux_bot_max=frozen_trial.user_attrs["flux_bot_max"],
            flux_mid_max=frozen_trial.user_attrs["flux_mid_max"],
            flux_density_top_max=frozen_trial.user_attrs["flux_density_top_max"],
            flux_density_bot_max=frozen_trial.user_attrs["flux_density_bot_max"],
            flux_density_mid_max=frozen_trial.user_attrs["flux_density_mid_max"],
            p_hyst=frozen_trial.user_attrs["p_hyst"],
            primary_litz_wire_loss=frozen_trial.user_attrs["primary_litz_wire_loss"],
            secondary_litz_wire_loss=frozen_trial.user_attrs["secondary_litz_wire_loss"],
            core_2daxi_total_volume=frozen_trial.values[0],
            total_loss=frozen_trial.values[1],
        )


def study_to_df(study_name: str, database_url: str):
    """Create a dataframe from a study.

    :param study_name: name of study
    :type study_name: str
    :param database_url: url of database
    :type database_url: str
    """
    loaded_study = optuna.create_study(study_name=study_name, storage=database_url, load_if_exists=True)
    df = loaded_study.trials_dataframe()
    df.to_csv(f'{study_name}.csv')


def load_fem_simulation_results(working_directory: str):
    """
    Load FEM simulation results from given working directory.

    param working_directory: Sets the working directory
    :type working_directory: str
    """
    working_directories = []
    labels = []
    fem_simulation_results_directory = os.path.join(working_directory, '02_fem_simulation_results')
    print("##########################")
    print(f"{fem_simulation_results_directory=}")
    print("##########################")
    file_names = [f for f in os.listdir(fem_simulation_results_directory) if
                  os.path.isfile(os.path.join(fem_simulation_results_directory, f))]

    counter = 0
    for name in file_names:
        temp_var = os.path.join(fem_simulation_results_directory, name)
        working_directories.append(temp_var)
        labels.append(name.split('_')[1].removesuffix('.json'))
        counter = counter + 1

    zip_iterator = zip(file_names, working_directories)
    logs = dict(zip_iterator)

    # After the simulations the sweep can be analyzed
    # This could be done using the FEMMTLogParser:
    log_parser = femmt.FEMMTLogParser(logs)

    # In this case the self inductivity of winding1 will be analyzed
    inductivities = []
    total_winding_loss = []
    total_Hys_loss = []
    total_volume = []
    total_cost = []
    for _, data in log_parser.data.items():
        inductivities.append(data.sweeps[0].windings[0].flux_over_current)
        total_winding_loss.append(data.total_winding_losses)
        total_Hys_loss.append(data.total_core_losses)
        total_volume.append(data.core_2daxi_total_volume)
        total_cost.append(data.total_cost)

    # real_inductance = []
    # for i in range(len(total_DC_loss)):
    #     real_inductance.append(inductivities[i].real)
    #     print(f'{labels[i]} -- total_loss: {total_DC_loss[i]} W -- core_2daxi_total_volume: {total_volume[i]}')

    return total_winding_loss, total_Hys_loss, total_volume, labels


def load_dab_dto_from_study(working_directory: str, study_name: str, trail_number: int | None = None):
    """
    Load a DAB-DTO from an optuna study.

    :param working_directory: directory
    :type working_directory: str
    :param study_name: study name to load
    :type study_name: str
    :param trail_number: trial number to load to the DTO
    :type trail_number: int
    :return:
    """
    if trail_number is None:
        raise NotImplementedError("needs to be implemented")

    optuna.logging.set_verbosity(optuna.logging.WARNING)  # Only warnings and errors will be shown
    loaded_study = optuna.create_study(study_name=study_name,
                                       storage=f"sqlite:///{working_directory}/study_{study_name}.sqlite3",
                                       load_if_exists=True)
    logging.info(f"The study '{study_name}' contains {len(loaded_study.trials)} trials.")
    # print(f'The study {study_name} contains {len(loaded_study.trials)} trials.')
    trials_dict_params = loaded_study.trials[trail_number].params
    trials_dict_user = loaded_study.trials[trail_number].user_attrs

    result_file_dto = DABStoSingleResultFile(
        case=trail_number,
        air_gap_top=trials_dict_user["air_gap_top"],
        air_gap_bot=trials_dict_user["air_gap_bot"],
        n_p_top=trials_dict_params["n_p_top"],
        n_p_bot=trials_dict_params["n_p_bot"],
        n_s_bot=trials_dict_user["n_s_bot"],
        window_h_top=trials_dict_user["window_h_top"],
        window_h_bot=trials_dict_params["window_h_bot"],
        window_w=trials_dict_user["window_w"],
        core_material=trials_dict_params["material"],
        core_inner_diameter=trials_dict_params["core_inner_diameter"],
        flux_top_max=trials_dict_user["flux_top_max"],
        flux_bot_max=trials_dict_user["flux_bot_max"],
        flux_mid_max=trials_dict_user["flux_mid_max"],
        flux_density_top_max=trials_dict_user["flux_density_top_max"],
        flux_density_bot_max=trials_dict_user["flux_density_bot_max"],
        flux_density_mid_max=trials_dict_user["flux_density_mid_max"],
        p_hyst=trials_dict_user["p_hyst"],
        primary_litz_wire=trials_dict_params["primary_litz_wire"],
        secondary_litz_wire=trials_dict_params["secondary_litz_wire"],
        primary_litz_wire_loss=trials_dict_user["primary_litz_wire_loss"],
        secondary_litz_wire_loss=trials_dict_user["secondary_litz_wire_loss"],
        total_loss=loaded_study.trials[trail_number].values[1],
        core_2daxi_total_volume=loaded_study.trials[trail_number].values[0]
    )
    return result_file_dto


def plot_2d(x_value: list, y_value: list, x_label: str, y_label: str, title: str, plot_color: str, z_value: list = None,
            z_label: str = None, inductance_value: list = None, annotations: list = None):
    """
    Visualize data in 2d plot with popover next to mouse position.

    :param x_value: Data points for x-axis
    :type x_value: list
    :param y_value: Data points for y-axis
    :type y_value: list
    :param z_value: Data points for z-axis
    :type z_value: list
    :param x_label: x-axis label
    :type x_label: str
    :param y_label: y-axis label
    :type y_label: str
    :param z_label: z-axis label
    :type z_label: str
    :param title: Title of the graph
    :type title: str
    :param inductance_value: Data points for inductance value corresponding to the (x, y, z): (Optional)
    :type inductance_value: list
    :param annotations: Annotations corresponding to the 3D points
    :type annotations: list
    :param plot_color: Color of the plot (the colors are based on 'femmt.colors_femmt_default')
    :type annotations: str
    """
    if annotations is None:
        names = [str(x) for x in list(range(len(x_value)))]
    else:
        temp_var = [int(x) for x in annotations]
        names = [str(x) for x in temp_var]

    if inductance_value is not None:
        l_label = 'L / H'

    if z_value is not None:
        z_value_str = [str(round(z, 3)) for z in z_value]

    if inductance_value is not None:
        l_value_str = [str(round(i_inductance, 6)) for i_inductance in inductance_value]

    x_value_str = [str(round(x, 6)) for x in x_value]
    y_value_str = [str(round(y, 3)) for y in y_value]

    fig, ax = plt.subplots()
    femmt.plt.title(title)
    femmt.plt.xlabel(x_label)
    femmt.plt.ylabel(y_label)

    # c = np.random.randint(1, 5, size=len(y_value))
    # norm = plt.Normalize(1, 4)
    # cmap = plt.cm.RdYlGn

    # sc = plt.scatter(x_value, y_value, c=c, s=50, cmap=cmap, norm=norm)
    if z_value is None:
        sc = plt.scatter(x_value, y_value, c='#%02x%02x%02x' % femmt.colors_femmt_default[plot_color])
    else:
        sc = plt.scatter(x_value, y_value, c=z_value, cmap=plot_color)
        cbar = plt.colorbar(sc)
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel(z_label, rotation=270)

    # # ===============================
    # # Adding annotations directly on the points
    # for i, txt in enumerate(names):
    #     ax.annotate(txt, (x_value[i], y_value[i]), textcoords="offset points", xytext=(0, 5), ha='center',
    #                 fontsize=7, rotation=90)
    #
    # ax.grid()
    # plt.show()
    # # ===============================

    annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annot(ind):
        """
        Create popover annotations in 2d plot.

        :param ind:
        :type ind:
        """
        pos = sc.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        text = ""
        if z_label is None and inductance_value is None:
            text = "case: {}\n{}: {}\n{}:{}". \
                format(" ".join([names[n] for n in ind["ind"]]),
                       x_label, " ".join([x_value_str[n] for n in ind["ind"]]),
                       y_label, " ".join([y_value_str[n] for n in ind["ind"]]))
        elif z_label is not None and inductance_value is None:
            text = "case: {}\n{}: {}\n{}:{}\n{}:{}". \
                format(" ".join([names[n] for n in ind["ind"]]),
                       x_label, " ".join([x_value_str[n] for n in ind["ind"]]),
                       y_label, " ".join([y_value_str[n] for n in ind["ind"]]),
                       z_label, " ".join([z_value_str[n] for n in ind["ind"]]))
        elif z_label is None and inductance_value is not None:
            text = "case: {}\n{}: {}\n{}:{}\n{}:{}". \
                format(" ".join([names[n] for n in ind["ind"]]),
                       x_label, " ".join([x_value_str[n] for n in ind["ind"]]),
                       y_label, " ".join([y_value_str[n] for n in ind["ind"]]),
                       l_label, " ".join([l_value_str[n] for n in ind["ind"]]))
        else:
            text = "case: {}\n{}: {}\n{}:{}\n{}:{}\n{}:{}". \
                format(" ".join([names[n] for n in ind["ind"]]),
                       x_label, " ".join([x_value_str[n] for n in ind["ind"]]),
                       y_label, " ".join([y_value_str[n] for n in ind["ind"]]),
                       z_label, " ".join([z_value_str[n] for n in ind["ind"]]),
                       l_label, " ".join([l_value_str[n] for n in ind["ind"]]))
        annot.set_text(text)
        # annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
        annot.get_bbox_patch().set_alpha(0.8)

    def hover(event):
        """
        Event that is triggered when mouse is hovered. Shows text annotation over data point closest to mouse.

        :param event:
        :type event:
        """
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)
    ax.grid()
    plt.show()
