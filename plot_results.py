from common_functions import get_last_run_folder, stats_file_names
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import json
import os

"""entire file GPT stuff"""


def plot_validation_and_testing(data_list):
    """GPT stuff"""
    n = len(data_list)
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Validation Data', 'Testing Data'))
    for i, data_dict in enumerate(data_list):
        validation_data = data_dict['validation']
        testing_data = data_dict['testing']
        method = data_dict.get('method', f'Data {i + 1}')
        fig.add_trace(go.Scatter(x=list(range(1, len(validation_data) + 1)), y=validation_data, mode='lines',
                                 name=f'{method} - V'), row=1, col=1)
        fig.add_trace(
            go.Scatter(x=list(range(1, len(testing_data) + 1)), y=testing_data, mode='lines', name=f'{method} - T'),
            row=1, col=2)
    fig.update_xaxes(title_text='Epoch', row=1, col=1)
    fig.update_yaxes(title_text='Percentile', row=1, col=1)
    fig.update_xaxes(title_text='Epoch', row=1, col=2)
    fig.update_yaxes(title_text='Percentile', row=1, col=2)
    fig.show()


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
    plot_validation_and_testing(percentiles)
