from NeuralNetwork import NeuralNetwork
from common_functions import print_asdf
import matplotlib.pyplot as plt
from random import randint
import numpy as np
import json
import os


class OilPumpField:
    def __init__(self, func, matrix_sizes, coordinates_boundaries, number_of_pumps, copy_field=None, label="Empty"):
        self.func = func
        self.matrix_sizes = matrix_sizes
        self.coordinates_boundaries = coordinates_boundaries
        self.number_of_pumps = number_of_pumps
        self.label = label
        self.img_id = 0

        self.is_field_oil_pump = np.zeros(self.matrix_sizes, int)
        self.field_coordinates = np.empty(self.matrix_sizes, tuple)
        self.unit_circle_field_coordinates = np.empty(self.matrix_sizes, tuple)
        self.list_of_pumps_entries = list()
        self.neural_network_answer_is_field_empty = np.zeros(self.matrix_sizes, int)
        self.neural_network_has_answered_at_least_once = False
        self.precalculate_coordinates()

        if copy_field is None:
            self.choose_oil_pumps()
        else:
            self.copy_oil_pumps(copy_field)

    def save_current_field(self, output_dir_name: str):
        bx, by = self.coordinates_boundaries
        mh, mw = self.matrix_sizes
        x_linspace = np.linspace(-bx, bx, mw)

        plt.figure(figsize=(8, 6))
        plt.plot(x_linspace, self.func(x_linspace), color='black', linestyle='-')

        for i in range(mh):
            for j in range(mw):
                x, y = self.field_coordinates[i, j]
                marker = '+'
                color = '#CCCCCC'

                if self.is_field_oil_pump[i, j]:
                    color = 'red' if self.is_oil_pump_empty(i, j) else 'blue'
                    marker = 'o'
                elif self.neural_network_has_answered_at_least_once:
                    color = 'red' if self.neural_network_answer_is_field_empty[i, j] else 'blue'

                plt.scatter(x, y, c=color, marker=marker, s=100)

        # plt.xticks([])
        # plt.yticks([])

        plt.savefig(os.path.join(output_dir_name, f'{self.label}-{self.img_id}.eps'), format='eps')
        self.img_id += 1

    def get_coordinates_by_indices(self, i, j):
        bx, by = self.coordinates_boundaries
        mh, mw = self.matrix_sizes
        dx, dy = 2 * bx / (mw - 1), 2 * by / (mh - 1)
        x, y = -bx + j * dx, by - i * dy
        return x, y

    def precalculate_coordinates(self):
        for i, row in enumerate(self.field_coordinates):
            for j, e in enumerate(row):
                self.field_coordinates[i, j] = self.get_coordinates_by_indices(i, j)
                self.unit_circle_field_coordinates[i, j] = self.push_indices_into_unit_circle(i, j)

    def choose_oil_pumps(self):
        c = 0
        while c < self.number_of_pumps:
            mh, mw = self.matrix_sizes
            i, j = randint(0, mh - 1), randint(0, mw - 1)
            if self.is_field_oil_pump[i, j] == 1:
                continue
            self.is_field_oil_pump[i, j] = 1
            self.list_of_pumps_entries.append((i, j))
            c += 1

    def copy_oil_pumps(self, cf):
        self.is_field_oil_pump = cf.is_field_oil_pump
        self.list_of_pumps_entries = cf.list_of_pumps_entries

    def save_field_settings(self, settings_folder_path):
        with open(os.path.join(settings_folder_path, "pump_entries.json"), 'w') as file:
            json.dump(self.list_of_pumps_entries, file)
        np.save(os.path.join(settings_folder_path, "is_field_oil_pump.npy"), self.is_field_oil_pump)

    def load_field_settings(self, settings_folder_path):
        with open(os.path.join(settings_folder_path, "pump_entries.json"), 'r') as file:
            self.list_of_pumps_entries = json.load(file)
        self.is_field_oil_pump = np.load(os.path.join(settings_folder_path, "is_field_oil_pump.npy"))

    def push_indices_into_unit_circle(self, i, j):
        bx, by = self.coordinates_boundaries

        x, y = self.field_coordinates[i, j]
        x /= bx
        y /= by

        return x, y

    def is_oil_pump_empty(self, i, j):
        x, y = self.field_coordinates[i, j]
        return y <= self.func(x)

    def update_neural_network_answers(self, nn):
        mh, mw = self.matrix_sizes
        for i in range(mh):
            for j in range(mw):
                x, y = self.unit_circle_field_coordinates[i, j]
                nn_input = np.array([[x], [y]])
                full, empty = nn.feedforward(nn_input)
                self.neural_network_answer_is_field_empty[i, j] = 1 if empty > full else 0

        self.neural_network_has_answered_at_least_once = True

    def get_training_data(self):
        inputs = []
        results = []

        for i, j in self.list_of_pumps_entries:
            x, y = self.unit_circle_field_coordinates[i, j]
            inputs.append(np.array([[x], [y]]))

            result = np.array([[0], [1]]) if self.is_oil_pump_empty(i, j) else np.array([[1], [0]])
            results.append(result)

        return list(zip(inputs, results))


if __name__ == '__main__':
    # output_dirname = input()
    output_dirname = "/home/jiri/PycharmProjects/NeuralNetwork/oil_pumps_example_plots/temp"

    net_structure = [2, 8, 2]
    net = NeuralNetwork(net_structure)
    net_params_file_name = os.path.join(output_dirname, "random_net_params.json")
    net.init_from_json(net_params_file_name)

    abstract_field = OilPumpField(lambda x: 2 - x ** 2 / 9,
                                  (21, 27), (6, 3), 75)
    abstract_field.load_field_settings("/home/jiri/PycharmProjects/NeuralNetwork/oil_pumps_example_plots/temp/oil_pumps_field_settings")
    abstract_field.save_current_field(output_dirname)

    training_data = abstract_field.get_training_data()
    print(net.get_total_loss_on_dataset_on_current_parameters(training_data))

    # stats = {}

    af = abstract_field
    # n_epochs = 2000
    tol = 0.005

    temp_opf = OilPumpField(af.func, af.matrix_sizes, af.coordinates_boundaries, af.number_of_pumps, af, "NM")

    temp_net = NeuralNetwork(net_structure)
    temp_net.init_from_json(net_params_file_name)
    temp_net.train_using_nelder_mead(training_data, 100000, 75, 1)

    temp_opf.update_neural_network_answers(temp_net)
    temp_opf.save_current_field(output_dirname)

    # for eta in [10, 5, 2, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]:
    #     print(f'Starting training with eta = {eta}.')
    #     temp_label = f'GD-{eta}'
    #     temp_opf = OilPumpField(af.func, af.matrix_sizes, af.coordinates_boundaries, af.number_of_pumps, af, temp_label)
    #     temp_net = NeuralNetwork(net_structure)
    #     temp_net.init_from_json(net_params_file_name)
    #     it = 0
    #     while it < n_epochs:
    #         it += 1
    #         temp_net.train_using_stochastic_gradient_descent(training_data, 1, 75, eta)
    #         if temp_net.get_total_loss_on_dataset_on_current_parameters(training_data) < tol:
    #             break
    #         if it == 300 and temp_net.get_total_loss_on_dataset_on_current_parameters(training_data) > 0.1:
    #             it = float('inf')
    #             break
    #     stats[temp_label] = {"net": temp_net, "opf": temp_opf, "n_steps": it}
    #
    #     temp_label = f'GDM-{eta}'
    #     temp_opf = OilPumpField(af.func, af.matrix_sizes, af.coordinates_boundaries, af.number_of_pumps, af, temp_label)
    #     temp_net = NeuralNetwork(net_structure)
    #     temp_net.init_from_json(net_params_file_name)
    #     it = 0
    #     while it < n_epochs:
    #         it += 1
    #         temp_net.train_using_stochastic_gradient_descent_with_momentum(training_data, 1, 1, eta, 0.9)
    #         if temp_net.get_total_loss_on_dataset_on_current_parameters(training_data) < tol:
    #             break
    #         if it == 300 and temp_net.get_total_loss_on_dataset_on_current_parameters(training_data) > 0.1:
    #             it = float('inf')
    #             break
    #     stats[temp_label] = {"net": temp_net, "opf": temp_opf, "n_steps": it}
    #
    #     temp_label = f'NAG-{eta}'
    #     temp_opf = OilPumpField(af.func, af.matrix_sizes, af.coordinates_boundaries, af.number_of_pumps, af, temp_label)
    #     temp_net = NeuralNetwork(net_structure)
    #     temp_net.init_from_json(net_params_file_name)
    #     it = 0
    #     while it < n_epochs:
    #         it += 1
    #         temp_net.train_using_nesterov(training_data, 1, 1, eta, 0.9)
    #         if temp_net.get_total_loss_on_dataset_on_current_parameters(training_data) < tol:
    #             break
    #         if it == 300 and temp_net.get_total_loss_on_dataset_on_current_parameters(training_data) > 0.1:
    #             it = float('inf')
    #             break
    #     stats[temp_label] = {"net": temp_net, "opf": temp_opf, "n_steps": it}
    #
    #     temp_label = f'AdaGrad-{eta}'
    #     temp_opf = OilPumpField(af.func, af.matrix_sizes, af.coordinates_boundaries, af.number_of_pumps, af, temp_label)
    #     temp_net = NeuralNetwork(net_structure)
    #     temp_net.init_from_json(net_params_file_name)
    #     it = 0
    #     while it < n_epochs:
    #         it += 1
    #         temp_net.train_using_adagrad(training_data, 1, 1, eta)
    #         if temp_net.get_total_loss_on_dataset_on_current_parameters(training_data) < tol:
    #             break
    #         if it == 300 and temp_net.get_total_loss_on_dataset_on_current_parameters(training_data) > 0.1:
    #             it = float('inf')
    #             break
    #     stats[temp_label] = {"net": temp_net, "opf": temp_opf, "n_steps": it}
    #
    #     temp_label = f'Adam-{eta}'
    #     temp_opf = OilPumpField(af.func, af.matrix_sizes, af.coordinates_boundaries, af.number_of_pumps, af, temp_label)
    #     temp_net = NeuralNetwork(net_structure)
    #     temp_net.init_from_json(net_params_file_name)
    #     it = 0
    #     while it < n_epochs:
    #         it += 1
    #         temp_net.train_using_adam(training_data, 1, 1, eta, 0.9, 0.999)
    #         if temp_net.get_total_loss_on_dataset_on_current_parameters(training_data) < tol:
    #             break
    #         if it == 300 and temp_net.get_total_loss_on_dataset_on_current_parameters(training_data) > 0.1:
    #             it = float('inf')
    #             break
    #     stats[temp_label] = {"net": temp_net, "opf": temp_opf, "n_steps": it}
    #
    # for label, stat in stats.items():
    #     stat["final_loss"] = stat["net"].get_total_loss_on_dataset_on_current_parameters(training_data)
    #     stat["label"] = label
    #
    # results = list(stats.values())
    # results.sort(key=lambda k: k["n_steps"])
    #
    # for i, res in enumerate(results):
    #     print(f'{res["label"]} - {res["n_steps"]} - {res["final_loss"]}')
    #     if i < 5:
    #         res["opf"].update_neural_network_answers(res["net"])
    #         res["opf"].save_current_field(output_dirname)

    # for it in range(50):
    #     #     # net.train_using_nelder_mead(training_data, 50, 50, 50)
    #     net.train_using_stochastic_gradient_descent(training_data, 1, 1, 1.0)
    #     print(net.get_total_loss_on_dataset_on_current_parameters(training_data))
    #     if it % 10 == 9:
    #         opf.update_neural_network_answers(net)
    #         opf.save_current_field(output_dirname)
    #
    # opf.save_current_field(output_dirname)
