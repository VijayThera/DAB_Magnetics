# python libraries
import os

# 3rd party libraries
import numpy as np
import datetime
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

DAB_StoInsulation_config = fmt.DABStoInsulation(
    iso_top_core=8e-4,
    iso_bot_core=8e-4,
    iso_left_core_min=1e-4,
    iso_right_core=1e-4,
    iso_primary_to_primary=1e-4,
    iso_secondary_to_secondary=1e-4,
    iso_primary_to_secondary=1e-4,
    iso_primary_inner_bobbin=1e-4,
)

DAB_transformer_config = fmt.DABStoSingleInputConfig(
    # target parameters
    l_s12_target=84e-6,
    l_h_target=600e-6,
    n_target=4.2,

    # operating point: current waveforms and temperature
    time_current_1_vec=np.array(i_1),
    time_current_2_vec=np.array(i_2),
    temperature=70,

    # sweep parameters: geometry and materials
    core_inner_diameter_min_max_list=[pq4040["core_inner_diameter"], pq4040["core_inner_diameter"]],
    window_w_min_max_list=[pq4040["window_w"], pq4040["window_w"]],
    material_list=[mdb.Material.N95],
    window_h_bot_min_max_list=[0.5 * pq4040["window_h"], 1.0 * pq4040["window_h"]],
    window_h_top_min_max_list=[0.1 * pq4040["window_h"], 0.5 * pq4040["window_h"]],

    primary_litz_wire_list=["1.4x200x0.071"],
    N_p_top_min_max_list=[1, 20],
    N_p_bot_min_max_list=[1, 20],
    primary_fill_factor=0.1,

    secondary_litz_wire_list=["1.4x200x0.071"],
    N_s_top_min_max_list=[0, 0],
    N_s_bot_min_max_list=[1, 20],
    secondary_fill_factor=0.1,
    # maximum limitation for transformer total height-m and core volume-m3(l*b*h/1e9)
    max_transformer_total_height=0.1,
    max_core_volume=1.6e-4,

    insulations=DAB_StoInsulation_config,

    # misc
    working_directory=os.path.join(os.path.dirname(__file__), "example_results", "DAB_optuna_cto"),
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

study_name = "workflow_003"
if __name__ == '__main__':

    fmt.DABStackedTransformerOptimization.start_proceed_study(study_name, DAB_transformer_config,
                                                              3,
                                                              3,
                                                              storage='sqlite',
                                                              sampler=optuna.samplers.NSGAIISampler(),
                                                              show_geometries=True)
    # fmt.DABStackedTransformerOptimization.show_study_results3(study_name, DAB_transformer_config,
    #                                                          0.05,
    #                                                          storage='sqlite')
