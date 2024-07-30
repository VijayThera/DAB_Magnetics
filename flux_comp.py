import os
import json
import csv
import numpy as np
from typing import Dict, List, Tuple


def extract_flux_from_json(working_directory: str) -> Dict[int, float]:
    """
    Extract flux values from JSON files in the given working directory.

    param working_directory: Sets the working directory
    :type working_directory: str
    """
    flux_values = {}

    design_files_directory = os.path.join(working_directory, 'fem_simulation_results')
    file_names = [f for f in os.listdir(design_files_directory) if
                  os.path.isfile(os.path.join(design_files_directory, f))]

    for name in file_names:
        file_path = os.path.join(design_files_directory, name)
        with open(file_path, 'r') as f:
            data = json.load(f)
            sweeps = data.get('single_sweeps', [])

            if sweeps:
                winding1 = sweeps[0].get('winding1', {})
                flux = winding1.get('flux', [0, 0])[0]  # Extract first value of the first flux array

            case_number = int(name.removesuffix('.json').split('_')[-1])
            flux_values[case_number] = flux

    return flux_values


def read_flux_from_csv(csv_path: str) -> Dict[int, float]:
    """
    Read 'Case_no.' and 'total_flux_max' from a CSV file.

    param csv_path: Path to the CSV file
    :type csv_path: str
    """
    flux_dict = {}
    with open(csv_path, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            case_no = int(float(row['Case_no.']))  # Convert to integer case number
            total_flux_max = float(row['total_flux_max'])
            flux_dict[case_no] = total_flux_max
    return flux_dict


def compare_flux_and_calculate_error(json_flux: Dict[int, float], csv_flux: Dict[int, float]) -> float:
    """
    Compare flux values from JSON and CSV and calculate the average error.

    param json_flux: Flux values extracted from JSON files
    :type json_flux: Dict[int, float]
    param csv_flux: Flux values extracted from the CSV file
    :type csv_flux: Dict[int, float]
    """
    errors = []
    for case_no, json_flux_value in json_flux.items():
        if case_no in csv_flux:
            csv_flux_value = csv_flux[case_no]
            # print(json_flux_value)
            # print('===============================================\n\n')
            # print(csv_flux_value)
            error = abs(json_flux_value - csv_flux_value) / csv_flux_value
            errors.append(error)

    average_error = np.mean(errors) if errors else 0.0
    return average_error


# Example usage:
working_directory = f'C:/Users/vijay/Desktop/UPB/Thesis/0_VM_results/inductor_optimization_07-29__15-37'
csv_path = os.path.join(working_directory, 'data_matrix_fem.csv')

# Extract flux values from JSON
json_flux = extract_flux_from_json(working_directory)

# Read flux values from CSV
csv_flux = read_flux_from_csv(csv_path)

# Compare and calculate the average error
average_error = compare_flux_and_calculate_error(json_flux, csv_flux)

# Print the average error
print(f"Average Error: {average_error * 100:.2f}%")