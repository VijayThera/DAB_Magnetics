import numpy as np
import magnethub as mh
import pandas as pd
import matplotlib.pyplot as plt

# femmt libraries
import femmt as fmt

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

core_database = fmt.core_database()
pq1611 = core_database["PQ 16/11.6"]
pq3535 = core_database["PQ 35/35"]

min_core = pq1611
max_core = pq3535

mu_o = 4e-7 * np.pi
mu_r = 3000

no_of_turns = np.arange(10, 51)  # Array of turns

# Create meshes for eff_mag_length and air_gap_length
eff_mag_length = np.linspace(pq1611['Eff_mag_path_length'], pq3535['Eff_mag_path_length'], 7)
air_gap_length = np.linspace(0.001, 0.0015, 15)

# Initialize an empty list to store b_waves
b_waves = []

# Iterate over all combinations of no_of_turns, eff_mag_length, and air_gap_length
for turns in no_of_turns:
    for eff_mag in eff_mag_length:
        for air_gap in air_gap_length:
            total_length = eff_mag + air_gap  # Calculate total length for this combination
            # print(f'turns: {turns}, total_length: {total_length}')  # Print the values of turns and total_length
            b_wave = 10e-3 * mu_o * mu_r * turns * i_Lc1_sampled / total_length
            b_waves.append(b_wave)

# Convert b_waves_list to a NumPy array
b_waves = np.array(b_waves)
print(f'B_waves:{b_waves[0]}')

# get power loss in W/m³ and estimated H wave in A/m
p, h = mdl(b_waves, freq, temp)
print(f'p:{p / 1000} kW/m³')
print(np.size(p))

# # Plotting the data
# plt.figure(figsize=(12, 6))
# # Current vs Time plot
# plt.subplot(2, 1, 1)
# plt.plot(time_sampled, i_Lc1_sampled, label='i_Lc1 (Current)')
# plt.xlabel('Time (s)')
# plt.ylabel('Current (A)')
# plt.title('Current vs Time')
# plt.legend()
# plt.grid(True)
#
# # B wave vs Time plot
# plt.subplot(2, 1, 2)
# plt.plot(time_sampled, b_waves[0], label='b_wave (Magnetic Flux Density)', color='r')
# plt.xlabel('Time (s)')
# plt.ylabel('Magnetic Flux Density (T)')
# plt.title('Magnetic Flux Density vs Time')
# plt.legend()
# plt.grid(True)
#
# plt.tight_layout()
# plt.show()
