# python libraries
import os

# 3rd party libraries
import numpy as np
from datetime import datetime
import optuna

# femmt libraries
import femmt as fmt

import materialdatabase as mdb

core_database = fmt.core_database()
pq4040 = core_database["PQ 40/40"]

i_1 = [[0.0, 3.265248131976911e-07, 2.5e-06, 2.8265248131976912e-06, 5e-06],
       [-0.9996115022426437, 4.975792579275104, 0.9996115022426446, -4.975792579275103, -0.9996115022426437]]
i_2 = [[0.0, 3.265248131976911e-07, 2.5e-06, 2.8265248131976912e-06, 5e-06],
       [-0.9196195846583147, -19.598444313231134, 0.9196195846583122, 19.59844431323113, -0.9196195846583147]]

StoInsulation_config = fmt.DABStoInsulation(
    iso_top_core=1.5e-3,
    iso_bot_core=1.5e-3,
    iso_left_core_min=1.5e-3,
    iso_right_core=1.5e-3,
    iso_primary_to_primary=0.00005,
    iso_secondary_to_secondary=0.00005,
    iso_primary_to_secondary=0.00005,
    iso_primary_inner_bobbin=1.5e-3,
)

DAB_transformer_config = fmt.DABStoSingleInputConfig(
    l_s_target=125e-6,
    l_h_target=669e-6,
    n_target=4.2,

    # operating point: current waveforms and temperature
    time_current_1_vec=np.array(i_1),
    time_current_2_vec=np.array(i_2),
    temperature=100,

    # sweep parameters: geometry and materials
    material_list=[mdb.Material.N95],
    core_inner_diameter_min_max_list=[pq4040["core_inner_diameter"], pq4040["core_inner_diameter"]],
    window_w_min_max_list=[pq4040["window_w"], pq4040["window_w"]],
    window_h_top_min_max_list=[0.1 * pq4040["window_h"], 0.5 * pq4040["window_h"]],
    window_h_bot_min_max_list=[0.1 * pq4040["window_h"], pq4040["window_h"]],

    # conductors
    primary_litz_wire_list=["1.1x60x0.1"],
    primary_fill_factor=0.9,
    secondary_litz_wire_list=["1.1x60x0.1"],
    secondary_fill_factor=0.9,
    # maximum limitation for transformer total height and core volume
    max_transformer_total_height=0.06,
    max_core_volume=1000e-3,
    insulations=StoInsulation_config,

    # misc
    working_directory=os.path.join(os.path.dirname(__file__), "example_results",
                                   "optuna_stacked_transformer_optimization"),
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

study_name = f'workflow_{datetime.now().strftime("%m-%d__%H-%M")}'

if __name__ == '__main__':

    time_start = datetime.now()

    if task == 'start_study':
        fmt.DABStackedTransformerOptimization.ReluctanceModel.NSGAII.start_study(study_name, DAB_transformer_config,
                                                                                 10000,
                                                                                 storage='sqlite')
    # elif task == 'filter_reluctance_model':
    #     # load trials from reluctance model
    #     reluctance_result_list = fmt.DABStackedTransformerOptimization.ReluctanceModel.NSGAII.load_study_to_dto(study_name, DAB_transformer_config)
    #     print(f"{len(reluctance_result_list)=}")
    #
    #     # filter air gaps
    #     filtered_air_gaps_dto_list = fmt.DABStackedTransformerOptimization.ReluctanceModel.filter_min_air_gap_length(reluctance_result_list)
    #     print(f"{len(filtered_air_gaps_dto_list)=}")
    #
    #     # filter for Pareto front
    #     pareto_reluctance_dto_list = fmt.DABStackedTransformerOptimization.ReluctanceModel.filter_loss_list(
    #         filtered_air_gaps_dto_list, factor_min_dc_losses=0.5)
    #     print(f"{len(pareto_reluctance_dto_list)=}")
    #
    #     fmt.DABStackedTransformerOptimization.plot(reluctance_result_list)
    #     fmt.DABStackedTransformerOptimization.plot(pareto_reluctance_dto_list)
    #
    #     # save results
    #     fmt.DABStackedTransformerOptimization.ReluctanceModel.save_dto_list(pareto_reluctance_dto_list, os.path.join(DAB_transformer_config.working_directory,
    #                                                                                                                  '01_reluctance_model_results_filtered'))
    #
    # elif task == 'fem_simulation_from_filtered_reluctance_model_results':
    #     # load filtered reluctance models
    #     pareto_reluctance_dto_list = fmt.DABStackedTransformerOptimization.ReluctanceModel.load_filtered_results(DAB_transformer_config.working_directory)
    #     print(f"{len(pareto_reluctance_dto_list)=}")
    #
    #     # start FEM simulation
    #     fmt.DABStackedTransformerOptimization.FemSimulation.simulate(config_dto=DAB_transformer_config,
    #                                                                  simulation_dto_list=pareto_reluctance_dto_list)
    #
    # elif task == 'plot_study_results':
    #     fmt.DABStackedTransformerOptimization.ReluctanceModel.NSGAII.show_study_results(study_name, DAB_transformer_config)

    time_stop = datetime.now()

    time_difference = time_stop - time_start
    print(f"{time_difference=}")
