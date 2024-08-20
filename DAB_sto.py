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
import gmsh
import os

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
                          show_geometries: bool = False):
                """
                Objective function to optimize.

                Using optuna. Some hints:

                 * returning failed trails by using return float('nan'), float('nan'),
                   see https://optuna.readthedocs.io/en/stable/faq.html#how-are-nans-returned-by-trials-handled
                 * speed up the search for NSGA-II algorithm with dynamic alter the search space, see https://optuna.readthedocs.io/en/stable/faq.html#id10


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
                core_inner_diameter = trial.suggest_categorical("core_inner_diameter",
                                                                config.core_inner_diameter_min_max_list)
                window_w = trial.suggest_categorical("window_w", config.window_w_min_max_list)
                window_h_top = trial.suggest_float("window_h_top", config.window_h_top_min_max_list[0],
                                                   config.window_h_top_min_max_list[1])
                window_h_bot = trial.suggest_float("window_h_bot", config.window_h_bot_min_max_list[0],
                                                   config.window_h_bot_min_max_list[1])

                material = trial.suggest_categorical("material", config.material_list)
                primary_litz_wire = trial.suggest_categorical("primary_litz_wire", config.primary_litz_wire_list)
                secondary_litz_wire = trial.suggest_categorical("secondary_litz_wire", config.secondary_litz_wire_list)

                tolerance = config.target_inductance_percent_tolerance / 100
                L_s = trial.suggest_float("L_s", (1 - tolerance) * config.l_s_target,
                                          (1 + tolerance) * config.l_s_target)
                L_h = trial.suggest_float("L_h", (1 - tolerance) * config.l_h_target,
                                          (1 + tolerance) * config.l_h_target)

                # cross-section comparison is according to a square for round wire.
                # this approximation is more realistic
                # insulation
                insulation_distance = 1.5e-3
                insulation_cross_section_top = 2 * insulation_distance * (window_w + window_h_top)
                insulation_cross_section_bot = 2 * insulation_distance * (window_w + window_h_bot)

                litz_database = ff.litz_database()

                primary_litz = litz_database[primary_litz_wire]
                secondary_litz = litz_database[secondary_litz_wire]

                total_available_window_cross_section_top = window_h_top * window_w - insulation_cross_section_top
                total_available_window_cross_section_bot = window_h_bot * window_w - insulation_cross_section_bot

                #########################################################
                # set dynamic wire count parameters as optimization parameters
                #########################################################
                # set the winding search space dynamic
                # https://optuna.readthedocs.io/en/stable/faq.html#what-happens-when-i-dynamically-alter-a-search-space

                # n_p_top suggestion
                n_p_top_max = total_available_window_cross_section_top / (2 * primary_litz["conductor_radii"]) ** 2
                if n_p_top_max < 0:
                    n_p_top_max = 0
                n_p_top = trial.suggest_int("n_p_top", 0, n_p_top_max)

                # n_s_top_suggestion
                winding_cross_section_n_p_top_max = n_p_top * (2 * primary_litz["conductor_radii"]) ** 2
                # n_s_top_max = int((total_available_window_cross_section_top - winding_cross_section_n_p_top_max) / (
                #         2 * secondary_litz["conductor_radii"]) ** 2)
                n_s_top = 0

                # n_p_bot suggestion
                n_p_bot_max = total_available_window_cross_section_bot / (2 * primary_litz["conductor_radii"]) ** 2
                if n_p_bot_max < 0:
                    n_p_bot_max = 0
                n_p_bot = trial.suggest_int("n_p_bot", 0, n_p_bot_max)

                # n_s_bot suggestion
                # winding_cross_section_n_p_bot_max = n_p_bot * (2 * primary_litz["conductor_radii"]) ** 2
                # n_s_bot_max = int((total_available_window_cross_section_bot - winding_cross_section_n_p_bot_max) / (
                #         2 * secondary_litz["conductor_radii"]) ** 2)
                # if n_s_bot_max < 0:
                #     n_s_bot_max = 0
                n_s_bot = round(n_p_bot / config.n_target)

                winding_cross_section_top = n_p_top * (2 * primary_litz["conductor_radii"]) ** 2 + n_s_top * (
                        2 * secondary_litz["conductor_radii"]) ** 2
                winding_cross_section_bot = n_p_bot * (2 * primary_litz["conductor_radii"]) ** 2 + n_s_bot * (
                        2 * secondary_litz["conductor_radii"]) ** 2

                thousand_simulations = trial.number / 1000

                if thousand_simulations.is_integer():
                    print(f"simulation count: {trial.number}")

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
                    t2_reluctance_matrix = fr.calculate_reluctance_matrix(t2_winding_matrix, target_inductance_matrix)

                    core_2daxi_total_volume = fr.calculate_core_2daxi_total_volume(core_inner_diameter,
                                                                                   (
                                                                                           window_h_bot + window_h_top + core_inner_diameter / 4),
                                                                                   window_w)

                    if np.linalg.det(t2_reluctance_matrix) != 0 and np.linalg.det(
                            np.transpose(t2_winding_matrix)) != 0 and np.linalg.det(target_inductance_matrix) != 0:
                        # calculate the flux
                        flux_top_vec, flux_bot_vec, flux_stray_vec = fr.flux_vec_from_current_vec(
                            target_and_fixed_parameters.current_extracted_top_vec,
                            target_and_fixed_parameters.current_extracted_bot_vec,
                            t2_winding_matrix,
                            target_inductance_matrix)

                        # calculate maximum values
                        flux_top_max, flux_bot_max, flux_stray_max = fr.max_value_from_value_vec(flux_top_vec,
                                                                                                 flux_bot_vec,
                                                                                                 flux_stray_vec)

                        flux_density_top_max = flux_top_max / core_cross_section
                        flux_density_bot_max = flux_bot_max / core_cross_section
                        flux_density_middle_max = flux_stray_max / core_cross_section

                        r_middle_target = -t2_reluctance_matrix[0][1]
                        r_top_target = t2_reluctance_matrix[0][0] - r_middle_target
                        r_bot_target = t2_reluctance_matrix[1][1] - r_middle_target

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

                        r_air_gap_top_target = r_top_target - 2 * r_center_topcore - 2 * r_top_topcore
                        r_air_gap_bot_target = r_bot_target - 2 * r_center_botcore - 2 * r_top_botcore

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

                                l_bot_air_gap = optimize.brentq(fr.r_air_gap_round_round_sct, minimum_air_gap_length,
                                                                maximum_air_gap_length,
                                                                args=(core_inner_diameter, window_h_bot / 2,
                                                                      window_h_bot / 2, r_air_gap_bot_target),
                                                                full_output=True)[0]

                            except ValueError:
                                # ValueError is raised in case of an air gap with length of zero
                                return float('nan'), float('nan')

                            if l_top_air_gap > core_inner_diameter or l_bot_air_gap > core_inner_diameter:
                                return float('nan'), float('nan')

                            if l_top_air_gap >= minimum_sort_out_air_gap_length and l_bot_air_gap >= minimum_sort_out_air_gap_length:

                                #=======================================================================================
                                mdl = mh.loss.LossModel(material="3C95", team="paderborn")

                                df = pd.read_csv('currents_shifted.csv')
                                i_top = df['i_Ls']
                                i_bot = df['i_Lc2_']
                                total_points = len(i_top)
                                step_size = total_points // 1024
                                i_top_sampled = np.array(i_top[::step_size][:1024])
                                i_bot_sampled = np.array(i_bot[::step_size][:1024])

                                core_width = fr.calculate_r_outer(core_inner_diameter, window_w)
                                inner_leg_width = core_inner_diameter / 2

                                core_height_top = window_h_top + core_inner_diameter / 2
                                core_height_bot = window_h_bot + core_inner_diameter / 2

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

                                flux_top_vec, flux_bot_vec, flux_stray_vec = fr.flux_vec_from_current_vec(
                                    i_top_sampled, i_bot_sampled, t2_winding_matrix, target_inductance_matrix)

                                B_top = (n_p_top / (r_top_target * core_cross_section)) * i_top_sampled
                                # B_top = flux_top_vec / np.array(core_cross_section)
                                # print(np.nanmax(B_top))
                                B_bot = ((n_p_bot / (r_bot_target * core_cross_section)) * i_top_sampled -
                                         ((config.n_target * n_s_bot / (r_bot_target * core_cross_section)) *
                                          (i_top_sampled - i_bot_sampled)))
                                # B_bot = flux_bot_vec / np.array(core_cross_section)
                                # print(np.nanmax(B_bot))

                                p_hyst_den_top, _ = mdl(B_top, config.frequency, config.temperature)
                                p_hyst_den_bot, _ = mdl(B_bot, config.frequency, config.temperature)

                                p_hyst_top = p_hyst_den_top * core_volume_top
                                p_hyst_bot = p_hyst_den_bot * core_volume_bot
                                # # =======================================================================================
                                # p_hyst_top = fr.hyst_losses_core_half_mu_r_imag(core_inner_diameter, window_h_top,
                                #                                                 window_w,
                                #                                                 material_data.material_mu_r_abs,
                                #                                                 flux_top_max,
                                #                                                 target_and_fixed_parameters.fundamental_frequency,
                                #                                                 material_data.material_flux_density_vec,
                                #                                                 material_data.material_mu_r_imag_vec)
                                #
                                # p_hyst_bot = 2 * fr.hyst_losses_core_half_mu_r_imag(core_inner_diameter, window_h_bot,
                                #                                                     window_w,
                                #                                                     material_data.material_mu_r_abs,
                                #                                                     flux_bot_max,
                                #                                                     target_and_fixed_parameters.fundamental_frequency,
                                #                                                     material_data.material_flux_density_vec,
                                #                                                     material_data.material_mu_r_imag_vec)
                                # #=======================================================================================
                                p_hyst = p_hyst_top + p_hyst_bot

                                # print(f'p_hyst_top:{p_hyst_top}')
                                # print(f'p_hyst_bot:{p_hyst_bot}')

                                primary_effective_conductive_cross_section = primary_litz["strands_numbers"] * \
                                                                             primary_litz["strand_radii"] ** 2 * np.pi
                                primary_effective_conductive_radius = np.sqrt(
                                    primary_effective_conductive_cross_section / np.pi)
                                primary_resistance = fr.resistance_solid_wire(core_inner_diameter, window_w,
                                                                              n_p_top + n_p_bot,
                                                                              primary_effective_conductive_radius,
                                                                              material='Copper')
                                # print(primary_resistance)
                                primary_dc_loss = primary_resistance * 4.0561 ** 2

                                secondary_effective_conductive_cross_section = secondary_litz["strands_numbers"] * \
                                                                               secondary_litz[
                                                                                   "strand_radii"] ** 2 * np.pi
                                secondary_effective_conductive_radius = np.sqrt(
                                    secondary_effective_conductive_cross_section / np.pi)
                                secondary_resistance = fr.resistance_solid_wire(core_inner_diameter, window_w,
                                                                                n_s_top + n_s_bot,
                                                                                secondary_effective_conductive_radius,
                                                                                material='Copper')
                                # print(secondary_resistance)
                                secondary_dc_loss = secondary_resistance * 19.0282 ** 2

                                total_loss = p_hyst + primary_dc_loss + secondary_dc_loss

                                trial.set_user_attr("n_s_bot", n_s_bot)
                                trial.set_user_attr("air_gap_top", l_top_air_gap)
                                trial.set_user_attr("air_gap_bot", l_bot_air_gap)

                                # trial.set_user_attr("flux_top_max", flux_top_max)
                                # trial.set_user_attr("flux_bot_max", flux_bot_max)
                                # trial.set_user_attr("flux_stray_max", flux_stray_max)
                                # trial.set_user_attr("flux_density_top_max", flux_density_top_max)
                                # trial.set_user_attr("flux_density_bot_max", flux_density_bot_max)
                                # trial.set_user_attr("flux_density_stray_max", flux_density_middle_max)
                                trial.set_user_attr("p_hyst", p_hyst)
                                trial.set_user_attr("primary_litz_wire_loss", primary_dc_loss)
                                trial.set_user_attr("secondary_litz_wire_loss", secondary_dc_loss)

                                difference_l_h = config.l_h_target - L_h
                                difference_l_s = config.l_s_target - L_s
                                deviation = 100 * (abs(difference_l_h / config.l_h_target) +
                                                   abs(difference_l_s / config.l_s_target))

                                trial.set_user_attr("deviation", deviation)

                                print(f"STO: successfully calculated trial {trial.number}")
                                return core_2daxi_total_volume, total_loss, deviation

                                # valid_design_dict = DABStoSingleResultFile(
                                #     case=trial.number,
                                #     air_gap_top=l_top_air_gap,
                                #     air_gap_bot=l_bot_air_gap,
                                #     n_p_top=n_p_top,
                                #     n_p_bot=n_p_bot,
                                #     n_s_top=n_s_top,
                                #     n_s_bot=n_s_bot,
                                #     window_h_top=window_h_top,
                                #     window_h_bot=window_h_bot,
                                #     window_w=window_w,
                                #     core_material=material_data.material_name,
                                #     core_inner_diameter=core_inner_diameter,
                                #     primary_litz_wire=primary_litz_wire,
                                #     secondary_litz_wire=secondary_litz_wire,
                                #     # results
                                #     flux_top_max=flux_top_max,
                                #     flux_bot_max=flux_bot_max,
                                #     flux_stray_max=flux_stray_max,
                                #     flux_density_top_max=flux_density_top_max,
                                #     flux_density_bot_max=flux_density_bot_max,
                                #     flux_density_stray_max=flux_density_middle_max,
                                #     p_hyst=p_hyst,
                                #     core_2daxi_total_volume=core_2daxi_total_volume,
                                #     primary_litz_wire_loss=primary_dc_loss,
                                #     secondary_litz_wire_loss=secondary_dc_loss,
                                #     total_loss=total_loss
                                # )

                                # if show_geometries:
                                #     verbosity = femmt.Verbosity.ToConsole
                                # else:
                                #     verbosity = femmt.Verbosity.Silent
                                #
                                # if total_loss <= 200:
                                #
                                #     print(f"FEM simulation for trial {trial.number}")
                                #     gmsh.initialize()
                                #
                                #     # 1. chose simulation type
                                #     working_directory_single_process = os.path.join(
                                #         target_and_fixed_parameters.working_directories.fem_working_directory,
                                #         f"process_{trial.number}")
                                #
                                #     # 1. chose simulation type
                                #     geo = femmt.MagneticComponent(
                                #         component_type=femmt.ComponentType.IntegratedTransformer,
                                #         working_directory=working_directory_single_process,
                                #         verbosity=femmt.Verbosity.ToConsole,
                                #         simulation_name=f"Case_{trial.number}")
                                #
                                #     # This line is for automated pytest running on GitHub only. Please ignore this line!
                                #     # if onelab_folder is not None:
                                #     #     geo.file_data.onelab_folder_path = onelab_folder
                                #
                                #     # 2. set core parameters
                                #     core_dimensions = femmt.dtos.StackedCoreDimensions(
                                #         core_inner_diameter=core_inner_diameter,
                                #         window_w=window_w,
                                #         window_h_top=window_h_top, window_h_bot=window_h_bot)
                                #     core = femmt.Core(core_type=femmt.CoreType.Stacked, core_dimensions=core_dimensions,
                                #                       material=material_data.material_name,
                                #                       temperature=config.temperature,
                                #                       frequency=target_and_fixed_parameters.fundamental_frequency,
                                #                       permeability_datasource=config.permeability_datasource,
                                #                       permeability_datatype=config.permeability_datatype,
                                #                       permeability_measurement_setup=config.permeability_measurement_setup,
                                #                       permittivity_datasource=config.permittivity_datasource,
                                #                       permittivity_datatype=config.permittivity_datatype,
                                #                       permittivity_measurement_setup=config.permittivity_measurement_setup)
                                #
                                #     geo.set_core(core)
                                #
                                #     # 3. set air gap parameters
                                #     air_gaps = femmt.AirGaps(femmt.AirGapMethod.Stacked, core)
                                #     air_gaps.add_air_gap(femmt.AirGapLegPosition.CenterLeg, l_top_air_gap,
                                #                          stacked_position=femmt.StackedPosition.Top)
                                #     air_gaps.add_air_gap(femmt.AirGapLegPosition.CenterLeg, l_bot_air_gap,
                                #                          stacked_position=femmt.StackedPosition.Bot)
                                #     geo.set_air_gaps(air_gaps)
                                #
                                #     # 4. set insulations
                                #     insulation = femmt.Insulation(flag_insulation=False)
                                #     insulation.add_core_insulations(config.insulations.iso_bot_core,
                                #                                     config.insulations.iso_top_core,
                                #                                     config.insulations.iso_left_core_min,
                                #                                     config.insulations.iso_right_core)  # [bot, top, left, right]
                                #     insulation.add_winding_insulations([[config.insulations.iso_primary_to_primary,
                                #                                          config.insulations.iso_primary_to_secondary],
                                #                                         [config.insulations.iso_primary_to_secondary,
                                #                                          config.insulations.iso_secondary_to_secondary]])
                                #     geo.set_insulation(insulation)
                                #
                                #     winding_window_top, winding_window_bot = femmt.create_stacked_winding_windows(core,
                                #                                                                                   insulation)
                                #
                                #     vww_top = winding_window_top.split_window(femmt.WindingWindowSplit.NoSplit)
                                #     vww_bot = winding_window_bot.split_window(femmt.WindingWindowSplit.NoSplit)
                                #
                                #     fill_factor = 0.78539
                                #     # 5. set conductor parameters
                                #     winding1 = femmt.Conductor(0, femmt.Conductivity.Copper)
                                #     winding1.set_litz_round_conductor(primary_litz["conductor_radii"],
                                #                                       primary_litz["strands_numbers"],
                                #                                       primary_litz["strand_radii"],
                                #                                       fill_factor,
                                #                                       femmt.ConductorArrangement.Hexagonal)
                                #
                                #     winding2 = femmt.Conductor(1, femmt.Conductivity.Copper)
                                #     winding2.set_litz_round_conductor(secondary_litz["conductor_radii"],
                                #                                       secondary_litz["strands_numbers"],
                                #                                       secondary_litz["strand_radii"],
                                #                                       fill_factor,
                                #                                       femmt.ConductorArrangement.Hexagonal)
                                #
                                #     # 6. add conductor to vww and add winding window to MagneticComponent
                                #     vww_top.set_interleaved_winding(winding1, n_p_top, winding2, n_s_top,
                                #                                     femmt.InterleavedWindingScheme.HorizontalAlternating)
                                #     vww_bot.set_interleaved_winding(winding1, n_p_bot, winding2, n_s_bot,
                                #                                     femmt.InterleavedWindingScheme.HorizontalAlternating)
                                #
                                #     geo.set_winding_windows([winding_window_top, winding_window_bot])
                                #
                                #     geo.create_model(freq=target_and_fixed_parameters.fundamental_frequency,
                                #                      pre_visualize_geometry=False)
                                #     geo.single_simulation(freq=target_and_fixed_parameters.fundamental_frequency,
                                #                           current=[5.5, 23], phi_deg=[0, 180],
                                #                           show_fem_simulation_results=False)
                                #
                                #     # center_tapped_study_excitation = geo.center_tapped_pre_study(
                                #     #     time_current_vectors=[[target_and_fixed_parameters.time_extracted_vec,
                                #     #                            target_and_fixed_parameters.current_extracted_top_vec],
                                #     #                           [target_and_fixed_parameters.time_extracted_vec,
                                #     #                            target_and_fixed_parameters.current_extracted_bot_vec]],
                                #     #     fft_filter_value_factor=config.fft_filter_value_factor)
                                #     #
                                #     # geo.stacked_core_center_tapped_study(center_tapped_study_excitation,
                                #     #                                      number_primary_coil_turns=n_p_top)
                                #
                                #     geo.get_inductances(I0=5.5, op_frequency=config.frequency, skin_mesh_factor=0.5,
                                #                         visualize_last_fem_simulation=False)
                                #
                                #     # copy result files to result-file folder
                                #     source_json_file = os.path.join(
                                #         target_and_fixed_parameters.working_directories.fem_working_directory,
                                #         f'process_{trial.number}',
                                #         "results", "log_electro_magnetic.json")
                                #     destination_json_file = os.path.join(
                                #         target_and_fixed_parameters.working_directories.fem_simulation_results_directory,
                                #         f'case_{trial.number}.json')
                                #
                                #     shutil.copy(source_json_file, destination_json_file)
                                #
                                #     # read result-log
                                #     with open(source_json_file, "r") as fd:
                                #         loaded_data_dict = json.loads(fd.read())
                                #
                                #     total_volume = loaded_data_dict["misc"]["core_2daxi_total_volume"]
                                #     total_loss = loaded_data_dict["total_losses"]["total_losses"]
                                #     # total_cost = loaded_data_dict["misc"]["total_cost_incl_margin"]
                                #
                                #     # Get inductance values
                                #     difference_l_h = config.l_h_target - geo.L_h_conc
                                #     difference_l_s = config.l_s_target - geo.L_s_conc
                                #
                                #     trial.set_user_attr("l_h", geo.L_h_conc)
                                #     trial.set_user_attr("l_s", geo.L_s_conc)
                                #
                                #     return total_volume, total_loss, 100 * (abs(difference_l_h / config.l_h_target) +
                                #                                             abs(difference_l_s / config.l_s_target))
                                # else:
                                #     return float('nan'), float('nan'), float('nan')
                            else:
                                return float('nan'), float('nan'), float('nan')
                        else:
                            return float('nan'), float('nan'), float('nan')
                    else:
                        return float('nan'), float('nan'), float('nan')

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
                # calculate the target and fixed parameters
                # and generate the folder structure inside this function
                target_and_fixed_parameters = DABStackedTransformerOptimization.calculate_fix_parameters(
                    config)

                # Wrap the objective inside a lambda and call objective inside it
                func = lambda trial: DABStackedTransformerOptimization.ReluctanceModel.NSGAII.objective(trial,
                                                                                                        config,
                                                                                                        target_and_fixed_parameters,
                                                                                                        False)

                # Pass func to Optuna studies
                study_in_memory = optuna.create_study(  #directions=["minimize", "minimize"],
                    directions=["minimize", "minimize", "minimize"],
                    # sampler=optuna.samplers.TPESampler(),
                    sampler=optuna.samplers.NSGAIISampler(),
                )

                # set logging verbosity:
                # https://optuna.readthedocs.io/en/stable/reference/generated/optuna.logging.set_verbosity.html#optuna.logging.set_verbosity
                # .INFO: all messages (default)
                # .WARNING: fails and warnings
                # .ERROR: only errors
                optuna.logging.set_verbosity(optuna.logging.ERROR)

                print(f"Sampler is {study_in_memory.sampler.__class__.__name__}")
                study_in_memory.optimize(func, n_trials=number_trials, n_jobs=-1, gc_after_trial=False)

                # fig = optuna.visualization.plot_pareto_front(study, targets=lambda t: (
                #     t.values[0] if error_difference_inductance_sum_percent > t.values[2] else None,
                #     t.values[1] if error_difference_inductance_sum_percent > t.values[2] else None),
                #                                              target_names=["volume in m³", "loss in W"])

                # in-memory calculation is shown before saving the data to database
                fig = optuna.visualization.plot_pareto_front(study_in_memory,
                                                             targets=lambda t: (
                                                                 t.values[0],
                                                                 t.values[1] if t.values[1] < 200 else None),
                                                             target_names=["volume", "losses"])
                # fig.show()

                # Current timestamp
                timestamp = datetime.now().strftime("%m-%d__%H-%M")

                # Create a unique filename for the Pareto front plot
                filename = f"Pareto_Front__Trials-{len(study_in_memory.trials)}__{timestamp}.html"

                # Specify the directory to save the file
                save_dir = '../DAB_Magnetics/example_results/pareto'
                os.makedirs(save_dir, exist_ok=True)

                # Combine directory and filename
                file_path = os.path.join(save_dir, filename)

                fig.write_html(file_path)
                # Print the file path for reference
                print('file_path:', file_path)

                # introduce study in storage, e.g. sqlite or mysql
                if storage == 'sqlite':
                    # Note: for sqlite operation, there needs to be three slashes '///' even before the path '/home/...'
                    # Means, in total there are four slashes including the path itself '////home/.../database.sqlite3'
                    storage = f"sqlite:///{config.working_directory}/study_{study_name}.sqlite3"
                elif storage == 'mysql':
                    storage = "mysql://monty@localhost/mydb",

                study_in_storage = optuna.create_study(directions=["minimize", "minimize", "minimize"],
                                                       study_name=study_name,
                                                       storage=storage)
                study_in_storage.add_trials(study_in_memory.trials)

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
                target_and_fixed_parameters = femmt.optimization.DABStackedTransformerOptimization.calculate_fix_parameters(
                    config)

                # Wrap the objective inside a lambda and call objective inside it
                func = lambda \
                        trial: femmt.optimization.DABStackedTransformerOptimization.ReluctanceModel.NSGAII.objective(
                    trial,
                    config,
                    target_and_fixed_parameters)

                study = optuna.create_study(study_name=study_name, storage=f"sqlite:///study_{study_name}.sqlite3",
                                            load_if_exists=True)
                study.optimize(func, n_trials=number_trials)

            @staticmethod
            def show_study_results(study_name: str, config: DABStoSingleInputConfig) -> None:
                """
                Show the results of a study.

                :param study_name: Name of the study
                :type study_name: str
                :param config: stacked transformer configuration file
                :type config: DABStoSingleInputConfig
                """
                study = optuna.create_study(study_name=study_name,
                                            storage=f"sqlite:///{config.working_directory}/study_{study_name}.sqlite3",
                                            load_if_exists=True)

                fig = optuna.visualization.plot_pareto_front(study, target_names=["volume", "losses", "deviation"])
                # fig.show()

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

                dto_list = [op.DABOptunaFemmtParser.parse(frozen_object) for frozen_object in study.trials \
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

                dto_list = [op.DABOptunaFemmtParser.parse(frozen_object) for frozen_object in study.best_trials]

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
        def load_filtered_results(working_directory: str) -> list[ItoSingleResultFile]:
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
            return femmt.DABStackedTransformerOptimization.ReluctanceModel.load_list(
                Stacked_transformer_reluctance_model_results_directory)

        @staticmethod
        def load_list(filepath: str) -> List[ItoSingleResultFile]:
            """
            Load the list of the reluctance models from the folder structure.

            :param filepath: filepath
            :type filepath: str
            :return: List of ItoSingleResultFiles
            :rtype: List[ItoSingleResultFile]
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
                     visualize: bool = False):
            """Perform the FEM simulation."""
            dab_func.stacked_transformer_fem_simulations_from_result_dtos(config_dto, simulation_dto_list, visualize)


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
        # flux_top_max=result_file_dict["flux_top_max"],
        # flux_bot_max=result_file_dict["flux_bot_max"],
        # flux_stray_max=result_file_dict["flux_stray_max"],
        # flux_density_top_max=result_file_dict["flux_density_top_max"],
        # flux_density_bot_max=result_file_dict["flux_density_bot_max"],
        # flux_density_stray_max=result_file_dict["flux_density_stray_max"],
        p_hyst=result_file_dict["p_hyst"],
        core_2daxi_total_volume=result_file_dict["core_2daxi_total_volume"],
        primary_litz_wire=result_file_dict["primary_litz_wire"],
        secondary_litz_wire=result_file_dict["secondary_litz_wire"],
        primary_litz_wire_loss=result_file_dict["primary_litz_wire_loss"],
        secondary_litz_wire_loss=result_file_dict["secondary_litz_wire_loss"],
        total_loss=result_file_dict["total_loss"]
    )
    return result_file_dto
