# python libraries
import os

# 3rd party libraries
import numpy as np
from datetime import datetime
import optuna

# femmt libraries
import femmt as fmt
import DAB_sto as dab

import materialdatabase as mdb

core_database = fmt.core_database()
pq3220 = core_database["PQ 32/20"]
pq3230 = core_database["PQ 32/30"]
pq3535 = core_database["PQ 35/35"]
pq4040 = core_database["PQ 40/40"]

i_1 = [[0, 4.000e-07, 8.000e-07, 1.366e-06, 2.211e-06, 2.635e-06, 3.073e-06, 4.108e-06, 5.0e-06],
       [-4.354, -0.210, 5.221, 4.781, 4.419, -0.331, -5.170, -4.696, -4.352]]
i_h = [[0, 4.000e-07, 8.000e-07, 1.366e-06, 2.211e-06, 2.635e-06, 3.073e-06, 4.108e-06, 5.0e-06],
       [-0.527, -0.947, -1.422, -0.453, 0.549, 1.058, 1.474, 0.395, -0.521]]
i_2 = [[0, 4.000e-07, 8.000e-07, 1.366e-06, 2.211e-06, 2.635e-06, 3.073e-06, 4.108e-06, 5.0e-06],
       [-16.0734, 3.0954, 27.9006, 21.9828, 16.254, -5.8338, -27.9048, -21.3822, -16.0902]]

# i_1 = [[0.0, 3.265248131976911e-07, 2.5e-06, 2.8265248131976912e-06, 5e-06],
#        [-0.9996115022426437, 4.975792579275104, 0.9996115022426446, -4.975792579275103, -0.9996115022426437]]
# i_2 = [[0.0, 3.265248131976911e-07, 2.5e-06, 2.8265248131976912e-06, 5e-06],
#        [-0.9196195846583147, -19.598444313231134, 0.9196195846583122, 19.59844431323113, -0.9196195846583147]]

StoInsulation_config = fmt.DABStoInsulation(
    iso_top_core=1.5e-3,
    iso_bot_core=1.5e-3,
    iso_left_core_min=1.5e-3,
    iso_right_core=1.5e-3,
    iso_primary_to_primary=0.0002,
    iso_secondary_to_secondary=0.0002,
    iso_primary_to_secondary=0.0002,
    iso_primary_inner_bobbin=0.0002,
)

DAB_transformer_config = fmt.DABStoSingleInputConfig(
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
    core_inner_diameter_min_max_list=[pq3220["core_inner_diameter"], pq3230["core_inner_diameter"],
                                      pq3535["core_inner_diameter"], pq4040["core_inner_diameter"]],
    window_w_min_max_list=[pq3220["window_w"], pq3230["window_w"], pq3535["window_w"],
                           pq4040["window_w"]],
    window_h_top_min_max_list=[0.1 * pq3220["window_h"], 0.5 * pq4040["window_h"]],
    window_h_bot_min_max_list=[0.25 * pq3220["window_h"], pq4040["window_h"]],

    # conductors
    primary_litz_wire_list=["1.1x60x0.1", "1.4x200x0.071"],
    secondary_litz_wire_list=["1.1x60x0.1", "1.4x200x0.071"],

    # maximum limitation for transformer total height and core volume
    max_transformer_total_height=0.06,
    max_core_volume=1000e-3,
    insulations=StoInsulation_config,

    # misc
    working_directory=os.path.join(os.path.dirname(__file__), "example_results",
                                   f'optuna_stacked_transformer_optimization'),
    #_{datetime.now().strftime("%m-%d__%H-%M-%S")}'),
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

# study_name = f'workflow_{datetime.now().strftime("%m-%d__%H-%M")}'
study_name = "workflow_test_01"

if __name__ == '__main__':

    time_start = datetime.now()

    if task == 'start_study':
        dab.DABStackedTransformerOptimization.ReluctanceModel.NSGAII.start_study(study_name, DAB_transformer_config,
                                                                                 200,
                                                                                 storage='sqlite')

    elif task == 'filter_reluctance_model':
        # load trials from reluctance model
        reluctance_result_list = dab.DABStackedTransformerOptimization.ReluctanceModel.NSGAII.load_study_to_dto(
            study_name, DAB_transformer_config)
        print(f"{len(reluctance_result_list)=}")

        # filter for Pareto front
        pareto_reluctance_dto_list = dab.DABStackedTransformerOptimization.ReluctanceModel.filter_loss_list(
            reluctance_result_list, factor_min_dc_losses=0.5)
        print(f"{len(pareto_reluctance_dto_list)=}")

        # dab.DABStackedTransformerOptimization.plot(reluctance_result_list)
        # dab.DABStackedTransformerOptimization.plot(pareto_reluctance_dto_list)

        # save results
        dab.DABStackedTransformerOptimization.ReluctanceModel.save_dto_list(pareto_reluctance_dto_list, os.path.join(
            DAB_transformer_config.working_directory,
            '01_reluctance_model_results_filtered'))

    elif task == 'fem_simulation_from_filtered_reluctance_model_results':
        # load filtered reluctance models
        pareto_reluctance_dto_list = dab.DABStackedTransformerOptimization.ReluctanceModel.load_filtered_results(
            DAB_transformer_config.working_directory)
        print(f"{len(pareto_reluctance_dto_list)=}")

        # start FEM simulation
        dab.DABStackedTransformerOptimization.FemSimulation.simulate(config_dto=DAB_transformer_config,
                                                                     simulation_dto_list=pareto_reluctance_dto_list)

    elif task == 'plot_study_results':
        dab.DABStackedTransformerOptimization.ReluctanceModel.NSGAII.show_study_results(study_name,
                                                                                        DAB_transformer_config)

    time_stop = datetime.now()

    time_difference = time_stop - time_start
    print(f"{time_difference=}")
