from common_functions import get_last_run_folder, stats_file_names
from plotly.subplots import make_subplots
from matplotlib import pyplot as plt
import plotly.graph_objs as go
import json
import os


"""entire file is GPT stuff tbh"""

def plot_validation_vs_testing(pairs):
    n = len(pairs)

    for i in range(n):
        fig, ax = plt.subplots()
        ax.plot(pairs[i]['validation'], label='Validation data')
        ax.plot(pairs[i]['testing'], label='Testing data')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Percentile')
        ax.set_title(pairs[i]['method'])
        ax.legend()
        plt.show()


if __name__ == '__main__':
    run_folder_path = input('Enter run folder path:')
    if run_folder_path == '':
        run_folder_path = get_last_run_folder('/home/jiri/PycharmProjects/NeuralNetwork/runs')
    percentiles = []
    for file_name in stats_file_names:
        with open(os.path.join(run_folder_path, file_name), 'r') as infile:
            data = json.load(infile)
            data['method'] = file_name[:-9].replace('_', ' ')
            percentiles.append(data)
    plot_validation_vs_testing(percentiles)
