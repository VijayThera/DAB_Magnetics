import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import magnethub as mh
import femmt as fmt
import femmt.functions_reluctance as fr

# #===================================================
# from matplotlib import rcParams
#
# # Setting font to Times New Roman and font size to 14
# rcParams['font.family'] = 'Times New Roman'
# rcParams['font.size'] = 14
#
# # Initialize the loss model
# mdl = mh.loss.LossModel(material="3C95", team="paderborn")
# temp = 100
#
# # Define constants and parameters
# mu_o = 4e-7 * np.pi
# mu_r = 3000
# num_samples = 1024
#
# # CSV file paths
# csv_file_paths = {
#     '100kHz': 'C:/Users/vijay/Desktop/UPB/Thesis/3C95_100kHz.csv',
#     '200kHz': 'C:/Users/vijay/Desktop/UPB/Thesis/3C95_200kHz.csv',
#     '400kHz': 'C:/Users/vijay/Desktop/UPB/Thesis/3C95_400kHz.csv'
# }
#
# # Frequencies and labels
# frequencies = [100e3, 200e3, 400e3]
# colors = ['r', 'b', 'g']
# labels = ['100kHz', '200kHz', '400kHz']
#
# # Initialize dictionaries to store power loss and amplitudes for different frequencies
# all_power_losses = {freq: [] for freq in frequencies}
# amplitudes_dict = {freq: [] for freq in frequencies}
#
#
# # Function to calculate power loss for given B values and frequency
# def calculate_power_loss(B_values, freq):
#     period = 1 / freq
#     t = np.linspace(0, period, num_samples, endpoint=False)
#     power_losses = []
#
#     for B in B_values:
#         b_wave = 1e-3 * B * np.sin(2 * np.pi * freq * t)
#         p, h = mdl(b_wave, freq, temp)
#         power_losses.append(p / 1000)
#     return power_losses
#
#
# for label, file_path in csv_file_paths.items():
#     freq = int(label[:-3]) * 1e3
#     csv_data = pd.read_csv(file_path)
#     B_values = csv_data['B'].values
#     amplitudes_dict[freq] = B_values
#     all_power_losses[freq] = calculate_power_loss(B_values, freq)
#
# for freq, color, label in zip(frequencies, colors, labels):
#     plt.plot(amplitudes_dict[freq], all_power_losses[freq], linestyle='-', marker='none', color=color)
#     plt.text(amplitudes_dict[freq][-1], all_power_losses[freq][-1], label, color=color)
#
# csv_data_100kHz = pd.read_csv(csv_file_paths['100kHz'])
# csv_data_200kHz = pd.read_csv(csv_file_paths['200kHz'])
# csv_data_400kHz = pd.read_csv(csv_file_paths['400kHz'])
#
# plt.plot(csv_data_100kHz['B'], csv_data_100kHz['P'], linestyle='--', marker='none', color='r')
# plt.plot(csv_data_200kHz['B'], csv_data_200kHz['P'], linestyle='--', marker='none', color='b')
# plt.plot(csv_data_400kHz['B'], csv_data_400kHz['P'], linestyle='--', marker='none', color='g')
#
# # Set the axis scales to logarithmic
# plt.xscale('log')
# plt.yscale('log')
#
# # Set the axis limits
# plt.xlim(10, 1e3)
# plt.ylim(10, 1e4)
#
# plt.grid(True, which="both", ls="--", linewidth=0.5)
#
# # Add a text box at the top left corner
# textstr = (' - Calc. power losses\n'
#            '-- Datasheet power losses')
#
# props = dict(boxstyle='round', facecolor='white', alpha=0.5)
# plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
#          verticalalignment='top', bbox=props)
#
# plt.title('Power Loss vs Flux Density')
# plt.xlabel('Flux Density (mT)')
# plt.ylabel('Power Loss (kW/m³)')
# plt.show()
# #==========================================================
# mdl = mh.loss.LossModel(material="3C95", team="paderborn")
# freq = 200e3
# temp = 100
# # Define constants and parameters
# mu_o = 4e-7 * np.pi
# mu_r = 3000
# num_samples = 1024
# amplitude_range_100kHz = np.linspace(70e-3, 300e-3, 10)
# amplitude_range_200kHz = np.linspace(40e-3, 200e-3, 10)
# amplitude_range_400kHz = np.linspace(25e-3, 100e-3, 10)
#
# # Initialize lists to store power loss and amplitudes
# power_losses_100kHz = []
# power_losses_200kHz = []
# power_losses_400kHz = []
# amplitudes_100kHz = []
# amplitudes_200kHz = []
# amplitudes_400kHz = []
#
# for amplitudes in amplitude_range_100kHz:
#     period = 1 / freq
#     t = np.linspace(0, period, num_samples, endpoint=False)
#     b_wave = amplitudes * np.sin(2 * np.pi * freq * t)
#     p, h = mdl(b_wave, 100e3, temp)
#     power_losses_100kHz.append(p / 1000)
#     amplitudes_100kHz.append(amplitudes * 1000)
#
# for amplitudes in amplitude_range_200kHz:
#     period = 1 / freq
#     t = np.linspace(0, period, num_samples, endpoint=False)
#     b_wave = amplitudes * np.sin(2 * np.pi * freq * t)
#     p, h = mdl(b_wave, 200e3, temp)
#     power_losses_200kHz.append(p / 1000)
#     amplitudes_200kHz.append(amplitudes * 1000)
#
# for amplitudes in amplitude_range_400kHz:
#     period = 1 / freq
#     t = np.linspace(0, period, num_samples, endpoint=False)
#     b_wave = amplitudes * np.sin(2 * np.pi * freq * t)
#     p, h = mdl(b_wave, 400e3, temp)
#     power_losses_400kHz.append(p / 1000)
#     amplitudes_400kHz.append(amplitudes * 1000)
#
# plt.plot(amplitudes_100kHz, power_losses_200kHz, linestyle='-', color='r', label='Calc-100kHz')
# plt.plot(amplitudes_200kHz, power_losses_200kHz, linestyle='-', color='b', label='Calc-200kHz')
# plt.plot(amplitudes_400kHz, power_losses_200kHz, linestyle='-', color='g', label='Calc-400kHz')
#
# # Read the CSV file
# csv_file_path_100kHz = 'C:/Users/vijay/Desktop/UPB/Thesis/3C95_100kHz.csv'
# csv_file_path_200kHz = 'C:/Users/vijay/Desktop/UPB/Thesis/3C95_200kHz.csv'
# csv_file_path_400kHz = 'C:/Users/vijay/Desktop/UPB/Thesis/3C95_400kHz.csv'
# csv_data_100kHz = pd.read_csv(csv_file_path_100kHz)
# csv_data_200kHz = pd.read_csv(csv_file_path_200kHz)
# csv_data_400kHz = pd.read_csv(csv_file_path_400kHz)
#
# # Plot CSV data
# plt.plot(csv_data_100kHz['B'], csv_data_100kHz['P'], linestyle='--', color='r', label='100kHz')
# plt.plot(csv_data_200kHz['B'], csv_data_200kHz['P'], linestyle='--', color='b', label='200kHz')
# plt.plot(csv_data_400kHz['B'], csv_data_400kHz['P'], linestyle='--', color='g', label='400kHz')
#
# plt.title('Power Loss vs Magnetic Flux Density Amplitude')
# plt.xlabel('Magnetic Flux Density Amplitude (mT)')
# plt.ylabel('Power Loss (kW/m³)')
# plt.legend()
# plt.grid(True)
# plt.show()

#==========================================================
# amplitude = 50e-3
# num_samples = 1024
# period = 1 / freq
# t = np.linspace(0, period, num_samples, endpoint=False)
# b_wave2 = amplitude * np.sin(2 * np.pi * freq * t)
# # get power loss in W/m³ and estimated H wave in A/m
# p, h = mdl(b_wave2, freq, temp)
# print(f'p:{p/1000} kW/m³')
#==============================================================

mdl = mh.loss.LossModel(material="3C95", team="paderborn")
freq = 200e3
temp = 100

df = pd.read_csv('Current waveform.csv')
time = df['# t']
i_Lc1 = df['i_Lc1']
total_points = len(i_Lc1)
step_size = total_points // 1024
i_Lc1_sampled = i_Lc1[::step_size][:1024]
time_sampled = time[::step_size][:1024]
i_Lc1_sampled = np.array(i_Lc1_sampled)
time_sampled = np.array(time_sampled)
print(f'I_max:{np.nanmax(i_Lc1_sampled)}')

mu_o = 4e-7 * np.pi
mu_r = 3000
n = 40
l_gap = 1e-3

core_database = fmt.core_database()
pq1611 = core_database["PQ 20/16"]

core_inner_diameter = pq1611['core_inner_diameter']
core_height_upper = pq1611['core_h'] / 2
core_height_lower = pq1611['core_h'] / 2
window_w = pq1611['window_w']
core_round_height = pq1611['window_h']
core_top_bot_height = core_inner_diameter / 4

r_gap = fr.r_air_gap_round_round(l_gap, core_inner_diameter, core_height_upper, core_height_lower)
r_top = fr.r_core_top_bot_radiant(core_inner_diameter, window_w, mu_r, core_top_bot_height)
r_round = fr.r_core_round(core_inner_diameter, core_round_height, mu_r)
r_core = 2 * r_top + 2 * r_round
r_total = r_core + r_gap
core_area = (core_inner_diameter / 2) ** 2 * np.pi

b_wave1 = n * i_Lc1_sampled / (r_total * core_area)
print(f'B_max:{np.nanmax(b_wave1)}')

# get power loss in W/m³ and estimated H wave in A/m
p, h = mdl(b_wave1, freq, temp)
print(f'p:{p/1000} kW/m³')

# Plotting the data
plt.figure(figsize=(12, 6))
# Current vs Time plot
plt.subplot(2, 1, 1)
plt.plot(time_sampled, i_Lc1_sampled, label='i_Lc1 (Current)')
plt.xlabel('Time (s)')
plt.ylabel('Current (A)')
plt.title('Current vs Time')
plt.legend()
plt.grid(True)

# B wave vs Time plot
plt.subplot(2, 1, 2)
plt.plot(time_sampled, b_wave1, label='b_wave (Magnetic Flux Density)', color='r')
plt.xlabel('Time (s)')
plt.ylabel('Magnetic Flux Density (T)')
plt.title('Magnetic Flux Density vs Time')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
