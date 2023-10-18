from NeuralNetwork import NeuralNetwork
from common_functions import print_asdf
import matplotlib.pyplot as plt
from random import randint
import numpy as np
import os

UID = 0


class OilPumpField:
    def __init__(self, func, matrix_sizes, coordinates_boundaries, number_of_pumps):
        self.func = func
        self.matrix_sizes = matrix_sizes
        self.coordinates_boundaries = coordinates_boundaries
        self.number_of_pumps = number_of_pumps

        self.is_field_oil_pump = np.zeros(self.matrix_sizes, int)
        self.field_coordinates = np.empty(self.matrix_sizes, tuple)
        self.unit_circle_field_coordinates = np.empty(self.matrix_sizes, tuple)
        self.list_of_pumps_entries = list()
        self.neural_network_answer_is_field_empty = np.zeros(self.matrix_sizes, int)
        self.neural_network_has_answered_at_least_once = False
        self.precalculate_coordinates()
        self.choose_oil_pumps()

    def save_current_field(self, output_dir_name: str):
        global UID

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

        plt.title('Oil pumps field')
        plt.savefig(os.path.join(output_dir_name, f'{UID}.jpg'))
        UID += 1

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
    output_dirname = input()
    # output_dirname = "/home/jiri/PycharmProjects/NeuralNetwork/oil_pumps_example_plots"

    net = NeuralNetwork([2, 8, 2])

    opf = OilPumpField(lambda x: 4 * x ** 4 - 0.15 * x ** 3 - 18.45 * x ** 2 + 0.4 * x + 10,
                       (21, 27), (2, 11), 150)

    opf.save_current_field(output_dirname)

    opf.update_neural_network_answers(net)

    opf.save_current_field(output_dirname)

    training_data = opf.get_training_data()
    print(net.get_total_loss_on_dataset_on_current_parameters(training_data))

    for _ in range(50):
        # net.train_using_nelder_mead(training_data, 50, 50, 50)
        net.train_using_stochastic_gradient_descent(training_data, 1, 1, 1.0)
        print(net.get_total_loss_on_dataset_on_current_parameters(training_data))
        opf.update_neural_network_answers(net)
        opf.save_current_field(output_dirname)

    opf.save_current_field(output_dirname)

