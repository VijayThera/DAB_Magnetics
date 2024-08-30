# python libraries
import os

# 3rd party libraries
import numpy as np
from datetime import datetime
import optuna
import pandas as pd

# femmt libraries
import femmt as fmt
import DAB_sto as dab
import DAB_sto_functions as dab_func
import DAB_sto_dtos as dabdtos

import materialdatabase as mdb

core_database = fmt.core_database()
pq2625 = core_database["PQ 26/25"]
pq3220 = core_database["PQ 32/20"]
pq3535 = core_database["PQ 35/35"]
pq4030 = core_database["PQ 40/30"]

i_1 = [[0.00000000e+00, 2.11162061e-12, 1.86446568e-08, 1.22988898e-06, 2.49995377e-06, 2.49995588e-06, 2.51859843e-06, 3.72984275e-06, 4.99990754e-06],
       [-0.6978413, -0.69782037, -0.41012329, 6.27606901, 0.6978413, 0.69782037, 0.41012329, -6.27606901, -0.6978413]]
i_2 = [[0.00000000e+00, 2.11162061e-12, 1.86446568e-08, 1.22988898e-06, 2.49995377e-06, 2.49995588e-06, 2.51859843e-06, 3.72984275e-06, 4.99990754e-06],
       [-4.08749914, -4.08741514, -2.91354639, 25.16846129, 4.08749914, 4.08741514, 2.91354639, -25.16846129, -4.08749914]]

# i_1 = [[0.0, 3.265248131976911e-07, 2.5e-06, 2.8265248131976912e-06, 5e-06],
#        [-0.9996115022426437, 4.975792579275104, 0.9996115022426446, -4.975792579275103, -0.9996115022426437]]
# i_2 = [[0.0, 3.265248131976911e-07, 2.5e-06, 2.8265248131976912e-06, 5e-06],
#        [-0.9196195846583147, -19.598444313231134, 0.9196195846583122, 19.59844431323113, -0.9196195846583147]]

StoInsulation_config = dabdtos.DABStoInsulation(
    iso_top_core=1.5e-3,
    iso_bot_core=1.5e-3,
    iso_left_core_min=1.5e-3,
    iso_right_core=1.5e-3,
    iso_primary_to_primary=0.0002,
    iso_secondary_to_secondary=0.0002,
    iso_primary_to_secondary=0.0002,
    iso_primary_inner_bobbin=0.0002,
)

DAB_transformer_config = dabdtos.DABStoSingleInputConfig(
    l_s_target=125e-6,
    l_h_target=669e-6,
    target_inductance_percent_tolerance=10,
    n_target=4.2,

    # operating point: current waveforms and temperature
    time_current_1_vec=np.array(i_1),
    time_current_2_vec=np.array(i_2),
    temperature=100,
    frequency=200000,
    air_gap_min=1e-4,
    air_gap_max=2e-3,

    # sweep parameters: geometry and materials
    material_list=[mdb.Material.N95],
    core_inner_diameter_min_max_list=[pq2625["core_inner_diameter"], pq3220["core_inner_diameter"],
                                      pq3535["core_inner_diameter"], pq4030["core_inner_diameter"]],
    window_w_min_max_list=[pq2625["window_w"], pq3220["window_w"], pq3535["window_w"], pq4030["window_w"]],
    window_h_top_min_max_list=[0.1, 0.5],
    window_h_bot_min_max_list=[0.25, 1],

    # conductors
    primary_litz_wire_list=["1.1x60x0.1", "1.4x200x0.071"],
    secondary_litz_wire_list=["1.1x60x0.1", "1.4x200x0.071"],
    n_p_top_max=25,
    n_p_bot_max=40,

    # maximum limitation for transformer total height and core volume
    max_transformer_total_height=0.06,
    max_core_volume=1000e-3,
    insulations=StoInsulation_config,

    # misc
    working_directory=os.path.join(os.path.dirname(__file__), "example_results",
                                   f'optuna_stacked_transformer_optimization2'),#_{datetime.now().strftime("%m-%d__%H-%M-%S")}'),
    fft_filter_value_factor=0.05,
    mesh_accuracy=0.8,

    # data sources
    permeability_datasource=fmt.MaterialDataSource.Measurement,
    permeability_datatype=fmt.MeasurementDataType.ComplexPermeability,
    permeability_measurement_setup=mdb.MeasurementSetup.LEA_LK,
    permittivity_datasource=fmt.MaterialDataSource.Measurement,
    permittivity_datatype=fmt.MeasurementDataType.ComplexPermittivity,
    permittivity_measurement_setup=mdb.MeasurementSetup.LEA_LK
)

task = 'start_study'
# task = 'filter_reluctance_model'
# task = 'fem_simulation_from_filtered_reluctance_model_results'
# task = 'plot_study_results'
# task = 'compare_results'
# task = 'load_single_trail'

max_loss = 1000
trail_number = 1924

# study_name = f'workflow_{datetime.now().strftime("%m-%d__%H-%M")}'
study_name = "workflow_test"

if __name__ == '__main__':

    time_start = datetime.now()

    if task == 'start_study':
        dab.DABStackedTransformerOptimization.ReluctanceModel.NSGAII.start_study(study_name,
                                                                                 DAB_transformer_config,
                                                                                 3000,
                                                                                 storage='sqlite')

    # ==========================================================================================
    # ==========================================================================================

    elif task == 'filter_reluctance_model':
        # load trials from reluctance model
        reluctance_result_list = dab.DABStackedTransformerOptimization.ReluctanceModel.NSGAII.load_study_to_dto(
            study_name, DAB_transformer_config)
        print(f"{len(reluctance_result_list)=}")

        # filter for Pareto front
        pareto_reluctance_dto_list = dab.DABStackedTransformerOptimization.ReluctanceModel.filter_loss_list(
            reluctance_result_list, factor_min_dc_losses=0.5)
        print(f"{len(pareto_reluctance_dto_list)=}")

        dab.DABStackedTransformerOptimization.plot(reluctance_result_list)
        dab.DABStackedTransformerOptimization.plot(pareto_reluctance_dto_list)

        # save results
        dab.DABStackedTransformerOptimization.ReluctanceModel.save_dto_list(pareto_reluctance_dto_list, os.path.join(
            DAB_transformer_config.working_directory,
            '01_reluctance_model_results_filtered'))

    # ==========================================================================================
    # ==========================================================================================

    elif task == 'fem_simulation_from_filtered_reluctance_model_results':
        # load filtered reluctance models
        pareto_reluctance_dto_list = dab.DABStackedTransformerOptimization.ReluctanceModel.load_filtered_results(
            DAB_transformer_config.working_directory)
        print(f"{len(pareto_reluctance_dto_list)=}")

        # start FEM simulation
        dab.DABStackedTransformerOptimization.FemSimulation.simulate(config_dto=DAB_transformer_config,
                                                                     simulation_dto_list=pareto_reluctance_dto_list,
                                                                     max_loss=max_loss, visualize=False)

    # ==========================================================================================
    # ==========================================================================================

    elif task == 'plot_study_results':
        dab.DABStackedTransformerOptimization.ReluctanceModel.NSGAII.show_study_results(study_name,
                                                                                        DAB_transformer_config,
                                                                                        max_loss)

    # ==========================================================================================
    # ==========================================================================================

    elif task == 'compare_results':
        losses, volume, labels = dab.load_fem_simulation_results(DAB_transformer_config.working_directory)

        for i in range(len(labels)):
            trail_dto = dab.load_dab_dto_from_study(working_directory=DAB_transformer_config.working_directory,
                                                    study_name=study_name, trail_number=int(labels[i]))
            # reluctance model results
            print(f'trail number: {labels[i]}'
                  f'\nRM - total losses: {trail_dto.total_loss:.2f} W, FEM - total losses: {losses[i]:.2f} W'
                  f'\n----------------------------------------------')
        dab.plot_2d(x_value=volume, y_value=losses,
                    x_label='Volume / m\u00b3', y_label='Loss / W',
                    title='Volume vs losses', plot_color='yellow', annotations=labels)

    # ==========================================================================================
    # ==========================================================================================

    elif task == 'load_single_trail':
        trail_dto = dab.load_dab_dto_from_study(working_directory=DAB_transformer_config.working_directory,
                                                study_name=study_name, trail_number=trail_number)
        print(trail_dto.total_loss, trail_dto.core_2daxi_total_volume)
        waveforms = pd.read_csv('currents_shifted.csv', delimiter=',')
        time = waveforms['# t'].to_numpy() - waveforms['# t'][0]
        i_ls = waveforms['i_Ls'].to_numpy() - np.mean(waveforms['i_Ls'])
        i_hf2 = waveforms['i_HF2'].to_numpy() - np.mean(waveforms['i_HF2'])
        time_current_vectors = [[time, i_ls], [time, -i_hf2]]
        dab_func.stacked_transformer_fem_simulation_from_result_dto(config_dto=DAB_transformer_config,
                                                                    dto=trail_dto,
                                                                    fem_working_directory=DAB_transformer_config.working_directory,
                                                                    fundamental_frequency=200e3,
                                                                    time_current_vectors=time_current_vectors,
                                                                    visualize=True)

    time_stop = datetime.now()

    time_difference = time_stop - time_start
    print(f"{time_difference=}")
