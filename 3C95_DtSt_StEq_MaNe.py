# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib import rcParams
# import magnethub as mh
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
# def calculate_power_loss_using_model(B_values, freq):
#     period = 1 / freq
#     t = np.linspace(0, period, num_samples, endpoint=False)
#     power_losses = []
#
#     for B in B_values:
#         b_wave = 1e-3 * B * np.sin(2 * np.pi * freq * t)
#         p, _ = mdl(b_wave, freq, temp)
#         power_losses.append(p / 1000)
#     return power_losses
#
#
# for label, file_path in csv_file_paths.items():
#     freq = int(label[:-3]) * 1e3
#     csv_data = pd.read_csv(file_path)
#     B_values = csv_data['B'].values
#     amplitudes_dict[freq] = B_values
#     all_power_losses[freq] = calculate_power_loss_using_model(B_values, freq)
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

#==================================== Datasheet-Steinmetz-mdoel for all freq
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib import rcParams
# import magnethub as mh
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
# steinmetz_power_losses = {freq: [] for freq in frequencies}
#
# # Steinmetz coefficients
# k = 1.36148085
# a = 1.49789876
# b = 2.87767503
#
# # Function to calculate power loss using the Steinmetz equation
# def calculate_power_loss_steinmetz(B_values, freq):
#     power_losses = k * (freq**a) * ((B_values/1000)**b) / 1000
#     return power_losses
#
# # Function to calculate power loss for given B values and frequency using the model
# def calculate_power_loss_using_model(B_values, freq):
#     period = 1 / freq
#     t = np.linspace(0, period, num_samples, endpoint=False)
#     power_losses = []
#
#     for B in B_values:
#         b_wave = 1e-3 * B * np.sin(2 * np.pi * freq * t)
#         p, _ = mdl(b_wave, freq, temp)
#         power_losses.append(p / 1000)
#     return power_losses
#
# # Read data and calculate losses
# for label, file_path in csv_file_paths.items():
#     freq = int(label[:-3]) * 1e3
#     csv_data = pd.read_csv(file_path)
#     B_values = csv_data['B'].values
#     amplitudes_dict[freq] = B_values
#     all_power_losses[freq] = calculate_power_loss_using_model(B_values, freq)
#     steinmetz_power_losses[freq] = calculate_power_loss_steinmetz(B_values, freq)
#
# print(steinmetz_power_losses)
# print(all_power_losses)
# # Plot the data
# plt.figure()
#
# for freq, color, label in zip(frequencies, colors, labels):
#     plt.plot(amplitudes_dict[freq], all_power_losses[freq], linestyle='-', marker='none', color=color, label=f'Model {label}')
#     plt.plot(amplitudes_dict[freq], steinmetz_power_losses[freq], linestyle='-', marker='x', color=color, label=f'Steinmetz {label}')
#     plt.text(amplitudes_dict[freq][-1], all_power_losses[freq][-1], label, color=color)
#
# # Plot the original data points
# for label, color in zip(labels, colors):
#     csv_data = pd.read_csv(csv_file_paths[label])
#     plt.plot(csv_data['B'], csv_data['P'], linestyle='--', marker='none', color=color, label=f'Datasheet {label}')
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
# textstr = (' - Calc. power losses using model\n'
#            '-- Datasheet power losses\n'
#            'x Steinmetz power losses')
#
# props = dict(boxstyle='round', facecolor='white', alpha=0.5)
# plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
#          verticalalignment='top', bbox=props)
#
# plt.title('Power Loss vs Flux Density')
# plt.xlabel('Flux Density (mT)')
# plt.ylabel('Power Loss (kW/m³)')
# plt.legend()
# plt.show()


#============== for 200kHz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import magnethub as mh

# Setting font to Times New Roman and font size to 14
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 14

# Initialize the loss model
mdl = mh.loss.LossModel(material="3C95", team="paderborn")
temp = 100

# Define constants and parameters
mu_o = 4e-7 * np.pi
mu_r = 3000
num_samples = 1024

# CSV file paths
csv_file_paths = {
    '200kHz': 'C:/Users/vijay/Desktop/UPB/Thesis/3C95_200kHz.csv',
}

# Frequencies and labels
frequencies = [200e3]
colors = ['orange']
labels = ['200kHz']

# Initialize dictionaries to store power loss and amplitudes for different frequencies
all_power_losses = {freq: [] for freq in frequencies}
amplitudes_dict = {freq: [] for freq in frequencies}
steinmetz_power_losses = {freq: [] for freq in frequencies}

# Steinmetz coefficients
k = 1.36148085
a = 1.49789876
b = 2.87767503

# Function to calculate power loss using the Steinmetz equation
def calculate_power_loss_steinmetz(B_values, freq):
    power_losses = k * (freq**a) * ((B_values/1000)**b) / 1000
    return power_losses

# Function to calculate power loss for given B values and frequency using the model
def calculate_power_loss_using_model(B_values, freq):
    period = 1 / freq
    t = np.linspace(0, period, num_samples, endpoint=False)
    power_losses = []

    for B in B_values:
        b_wave = 1e-3 * B * np.sin(2 * np.pi * freq * t)
        p, _ = mdl(b_wave, freq, temp)
        power_losses.append(p / 1000)
    return power_losses

# Read data and calculate losses
for label, file_path in csv_file_paths.items():
    freq = int(label[:-3]) * 1e3
    csv_data = pd.read_csv(file_path)
    B_values = csv_data['B'].values
    amplitudes_dict[freq] = B_values
    all_power_losses[freq] = calculate_power_loss_using_model(B_values, freq)
    steinmetz_power_losses[freq] = calculate_power_loss_steinmetz(B_values, freq)

# Plot the data
plt.figure()

for freq, color, label in zip(frequencies, colors, labels):
    plt.plot(amplitudes_dict[freq], all_power_losses[freq], linestyle='solid', marker='none', color='b', label='MagNet')
    plt.plot(amplitudes_dict[freq], steinmetz_power_losses[freq], linestyle='dotted', marker='o', color='r', label='Steinmetz')
    # plt.text(amplitudes_dict[freq][-1], all_power_losses[freq][-1], label, color=color)

# Plot the original data points
for label, color in zip(labels, colors):
    csv_data = pd.read_csv(csv_file_paths[label])
    plt.plot(csv_data['B'], csv_data['P'], linestyle='--', marker='x', color='g', label='Datasheet')

# Set the axis scales to logarithmic
plt.xscale('log')
plt.yscale('log')

# Set the axis limits
plt.xlim(40, 250)
plt.ylim(10, 2000)

plt.grid(True, which="both", ls="--", linewidth=0.5)

# Add a text box at the top left corner
# textstr = (' - MagNet\n'
#            '-x- Datasheet\n'
#            '... Steinmetz')

props = dict(boxstyle='round', facecolor='white', alpha=0.5)
# plt.text(0.05, 0.95, transform=plt.gca().transAxes, fontsize=12,
#          verticalalignment='top', bbox=props)

plt.title('Power Density Loss vs Peak Flux Density @ 200kHz')
plt.xlabel('Peak Flux Density / mT')
plt.ylabel('Power Density Loss / kW/m³')
plt.legend()
plt.show()
