import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import magnethub as mh
import femmt as fmt
import femmt.functions_reluctance as fr
from pandas.plotting import table
from scipy.integrate import quad

# core_height = 11.6e-3 + 7e-3 / 2
# winding_height = 6.7e-3
# core_width = 14.4e-3
# winding_width = 3.7e-3
# inner_leg_width = 7e-3 / 2
#
# air_gap_volume = np.pi * inner_leg_width ** 2 * 1e-3
#
# x = np.pi * (core_width ** 2 * core_height - (inner_leg_width + winding_width) ** 2 * winding_height
#              + inner_leg_width ** 2 * winding_height) - air_gap_volume

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
# #=======================================================================================================================
# def calculate_litz_areas():
#     """
#     Calculate the cross-sectional areas of all Litz wires in the database.
#
#     :return: A dictionary where keys are Litz wire identifiers and values are their cross-sectional areas in square meters.
#     :rtype: Dict[str, float]
#     """
#     litz_dict = fmt.litz_database()
#     areas = {}
#
#     for key, data in litz_dict.items():
#         strand_numbers = data["strands_numbers"]
#         strand_radius = data["strand_radii"]
#         conductor_radius = data["conductor_radii"]
#
#         strand_area = np.pi * (strand_radius ** 2)
#         total_area = strand_numbers * strand_area
#
#         areas[key] = {
#             "strands": strand_numbers,
#             "strand diameter / mm": round(2 * strand_radius * 1000, 3),  # Diameter in mm
#             "conductor diameter / mm": round(2 * (conductor_radius * 1000), 3),  # Assuming same as stranddia for this example
#             "crossection area / mm^2": round(total_area, 8)
#         }
#
#     return areas
#
# # Calculate areas
# litz_data = calculate_litz_areas()
#
# # Create a DataFrame from the dictionary
# df = pd.DataFrame.from_dict(litz_data, orient='index')
#
# # Sort DataFrame by 'area' in ascending order
# df = df.sort_values(by='crossection area / mm^2', ascending=True)
#
# # Apply Times New Roman font globally
# plt.rcParams.update({'font.family': 'serif', 'font.serif': 'Times New Roman'})
#
# # Save DataFrame to an image
# fig, ax = plt.subplots(figsize=(10, 4))  # Adjust the size as needed
# ax.xaxis.set_visible(False)
# ax.yaxis.set_visible(False)
# ax.set_frame_on(False)
# tbl = table(ax, df, loc='center', cellLoc='center', colWidths=[0.2]*len(df.columns))
# tbl.auto_set_font_size(False)
# tbl.set_fontsize(12)
# tbl.scale(1.2, 1.2)  # Adjust the scale as needed
#
# # Save the table as an image
# # plt.savefig('litz_table.png', bbox_inches='tight', pad_inches=0.1)
# plt.show()
# #=======================================================================================================================

# def calculate_litz_areas():
#     """
#     Calculate the cross-sectional areas of all Litz wires in the database.
#
#     :return: A dictionary where keys are Litz wire identifiers and values are their cross-sectional areas in square meters.
#     :rtype: Dict[str, float]
#     """
#     litz_dict = fmt.litz_database()
#     areas = {}
#
#     for key, data in litz_dict.items():
#         strand_numbers = data["strands_numbers"]
#         strand_radius = data["strand_radii"]
#
#         strand_area = np.pi * (strand_radius ** 2)
#         total_area = strand_numbers * strand_area
#
#         areas[key] = total_area
#
#     return areas
#
#
# def filter_and_print_small_areas(threshold: float):
#     """
#     Filter and print Litz wires with cross-sectional areas less than the given threshold.
#
#     :param threshold: The area threshold in square meters.
#     """
#     areas = calculate_litz_areas()
#
#     for key, area in areas.items():
#         if threshold > area > 9.5e-8:
#             print(f"copper cross-sectional area of the Litz wire '{key}' is {area:.6e} m2.")
#
# threshold = 1.5e-6
# filter_and_print_small_areas(threshold)

##======================================================================================================================
def cos_power(x):
    return np.abs(np.cos(x)) ** 1.5

k = 1.36148085
a = 1.49789876
b = 2.87767503
D = 0.50
integral_value, error = quad(cos_power, 0, 2 * np.pi)
g = 2 * (D ** (1 - a) + (1 - D) ** (1 - a)) / (np.pi ** (a - 1) * integral_value)
# print(f'g(a:1.50, D:0.25) = {2 * (0.25 ** (1 - 1.5) + (1 - 0.25) ** (1 - 1.5)) / (np.pi ** (1.5 - 1) * integral_value):.3f}')
print(f'g(a:{a}, D:{D:.2f}) = {g:.3f}')

mdl = mh.loss.LossModel(material="3C95", team="paderborn")
freq = 200e3
temp = 100
num_samples = 1024

df = pd.read_csv('currents_shifted.csv')
time = df['# t']
i_Lc1 = df['i_Lc1']
total_points = len(i_Lc1)
# print(total_points)
step_size = total_points // num_samples
i_Lc1_sampled = i_Lc1[::step_size][:num_samples]
time_sampled = time[::step_size][:num_samples]
i_Lc1_sampled = np.array(i_Lc1_sampled)
time_sampled = np.array(time_sampled)
# print(f'I_max:{np.nanmax(i_Lc1_sampled)}')

mu_o = 4e-7 * np.pi
mu_r = 3000
n = 50
l_gap = 0.001269

core_database = fmt.core_database()
core = core_database["PQ 35/35"]

core_inner_diameter = core['core_inner_diameter']
core_height_upper = 0.024041666666666666 / 2 #core['core_h'] / 2
core_height_lower = 0.024041666666666666 / 2 #core['core_h'] / 2
window_w = core['window_w']
core_round_height = 0.016866666666666665 #pq1611['window_h']
core_top_bot_height = core_inner_diameter / 4

r_gap = fr.r_air_gap_round_round(l_gap, core_inner_diameter, core_height_upper, core_height_lower)
r_top = fr.r_core_top_bot_radiant(core_inner_diameter, window_w, mu_r, core_top_bot_height)
r_round = fr.r_core_round(core_inner_diameter, core_round_height, mu_r)
r_core = 0.5 * r_top + 2 * r_round
r_total = r_core + r_gap
# print(f'r_total: {r_total:.3f}')
core_area = (core_inner_diameter / 2) ** 2 * np.pi

L = n**2/r_total
# print(f'Inductance:{L}')

period = 1 / freq
t = np.linspace(0, period, num_samples, endpoint=False)
b_wave_sine = n * (np.nanmax(i_Lc1_sampled) * np.sin(2 * np.pi * freq * t - np.pi / 2)) / (r_total * core_area)

# get power loss in W/m³ and estimated H wave in A/m
p_sine, _ = mdl(b_wave_sine, freq, temp)

p_sine_se = k * (freq**a) * (np.nanmax(b_wave_sine)**b)

b_wave_triangle = n * i_Lc1_sampled / (r_total * core_area)
p_triangle, _ = mdl(b_wave_triangle, freq, temp)

p_triangle_se = p_sine_se * g

print(f'Power Density - sine (using MagNet) :   {p_sine/1000:.2f} kW/m³, Power loss: {p_sine*2.3223742062601302e-05:.2f} W')
print(f'Power Density - sine (using Steinmetz): {p_sine_se/1000:.2f} kW/m³, Power loss: {p_sine_se*2.3223742062601302e-05:.2f} W')
print(f'error: {abs(p_sine-p_sine_se) * 100/p_sine_se} %\n')
print(f'Power Density - tri. (using MagNet) :   {p_triangle/1000:.2f} kW/m³, Power loss: {p_triangle*2.3223742062601302e-05:.2f} W')
print(f'Power Density - tri. (using Steinmetz): {p_triangle_se/1000:.2f} kW/m³, Power loss: {p_triangle_se*2.3223742062601302e-05:.2f} W')

# print(f'traingular power_density should be = {g*p_sine_se/1000:.2f} kW/m³ , but Calc. traingular power_density is = {p_triangle/1000:.2f} kW/m³')
print(f'Calc. g(a, D) from p_triangle(MagNet)/p_sine(Steinmetz): {p_triangle/p_sine_se:.3f}')


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
plt.plot(time_sampled, b_wave_triangle, label='b_wave_triangle', color='r')
plt.plot(time_sampled, b_wave_sine, label='b_wave_sine', color='b')
plt.xlabel('Time (s)')
plt.ylabel('Magnetic Flux Density (T)')
plt.title('Magnetic Flux Density vs Time')
plt.legend()
plt.grid(True)

plt.tight_layout()
# plt.show()

##======================================================================================================================

# mu_o = 4e-7 * np.pi
# mu_r = 3000
# n_range = np.arange(1, 101)
# l_gap_range = np.linspace(0.1e-3, 2e-3, 100)
#
# core_database = fmt.core_database()
# core = core_database["PQ 20/20"]
#
# core_inner_diameter = core['core_inner_diameter']
# core_height_upper = core['core_h'] / 2
# core_height_lower = core['core_h'] / 2
# window_w = core['window_w']
# core_round_height = core['window_h']
# core_top_bot_height = core_inner_diameter / 4
#
# def calculate_r_total(l_gap):
#     r_gap = fr.r_air_gap_round_round(l_gap, core_inner_diameter, core_height_upper, core_height_lower)
#     r_top = fr.r_core_top_bot_radiant(core_inner_diameter, window_w, mu_r, core_top_bot_height)
#     r_round = fr.r_core_round(core_inner_diameter, core_round_height, mu_r)
#     r_core = 2 * r_top + 2 * r_round
#     r_total = r_core + r_gap
#     return r_total
#
# def find_turns_and_gap(target_inductance):
#     results = []
#     for n in n_range:
#         for l_gap in l_gap_range:
#             r_total = calculate_r_total(l_gap)
#             L = n**2 / r_total
#             if np.isclose(L, target_inductance, rtol=1e-2):  # 1% relative tolerance
#                 results.append((n, r_total, l_gap, L))
#     return results
#
# target_inductance = 680e-6
# results = find_turns_and_gap(target_inductance)
#
# for result in results:
#     print(f"Turns: {result[0]}, r_total: {result[1]:.0f} Air Gap: {result[2]:.4f} m, Inductance: {result[3]:.2e} H")