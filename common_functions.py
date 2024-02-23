from __future__ import annotations
import numpy as np
import datetime
import os

stats_file_names = ['SGD_stats.txt', 'SGD_with_momentum_stats.txt', 'NAG_stats.txt']

asdf = 999999


def print_asdf():
    print(asdf)


def sigmoid(z: np.ndarray) -> np.ndarray:
    # global asdf
    # x = min(z)
    # asdf = min(asdf, x)
    # for i in range(len(z)):
    #     if z[i] < -512.0:
    #         print("overflow :D")
    #         z[i] = -512.0
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(z: np.ndarray) -> np.ndarray:
    return sigmoid(z) * (1.0 - sigmoid(z))


def cost_function(output_activations: np.ndarray, desired_activations: np.ndarray) -> np.ndarray:
    return 1 / 2 * (output_activations - desired_activations) ** 2


def cost_function_derivative(output_activations: np.ndarray, desired_activations: np.ndarray) -> np.ndarray:
    return output_activations - desired_activations


def get_folder_name_for_this_run():
    """GPT stuff"""
    # Get current date and time
    now = datetime.datetime.now()

    # Format the date string
    date_string = now.strftime("%a_%d_%B_%H_%M_%S").lower()

    return os.path.join('/home/jiri/PycharmProjects/NeuralNetwork/runs', date_string)


def create_folder_if_it_does_not_exits(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def get_runtime_nicely_formatted(start_time, end_time, action: str):
    """GPT stuff"""
    runtime = end_time - start_time
    minutes, seconds = divmod(runtime, 60)
    hours, minutes = divmod(minutes, 60)

    return f"Runtime of {action}: {hours:02.0f}:{minutes:02.0f}:{seconds:02.0f}"


def print_runtime_and_write_to_time_log(time_log: str, start_time, end_time, action: str):
    runtime = get_runtime_nicely_formatted(start_time, end_time, action)
    print(runtime)
    return time_log + runtime + '\n'


def save_string_to_file(string: str, file_path: str):
    """GPT stuff"""
    with open(file_path, 'w') as file:
        file.write(string)


def print_and_save_time_log(time_log: str, path: str):
    print('\n' + 37 * '-' + '\n')
    print(time_log)
    save_string_to_file(time_log, path)


def get_last_run_folder(parent_dir: str):
    """GPT stuff"""
    subdirs = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]

    # Sort the subdirectories by creation time in descending order
    subdirs.sort(key=lambda x: os.path.getctime(os.path.join(parent_dir, x)), reverse=True)

    # Select the youngest subdirectory (the first one in the sorted list)
    last_dir = os.path.join(parent_dir, subdirs[0])

    return last_dir
