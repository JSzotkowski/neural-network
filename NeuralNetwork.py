from __future__ import annotations
from common_functions import sigmoid, sigmoid_derivative, cost_function, cost_function_derivative
from NelderMeadHelper import NelderMeadHelper
from random import shuffle
from time import time
from math import inf
import numpy as np
import json


class NeuralNetwork:
    def __init__(self, structure: list):
        self.n_layers = len(structure)
        self.structure = structure
        self.biases = [np.random.randn(n, 1) for n in self.structure[1:]]
        self.weights = [np.random.randn(n, m) for m, n in zip(self.structure[:-1], self.structure[1:])]

        # initialization of previous gradients for methods that need them
        # = SGD with momentum, Adagrad, Adam
        self.previous_update_w = [np.zeros(w.shape) for w in self.weights]
        self.previous_update_b = [np.zeros(b.shape) for b in self.biases]
        self.gradient_squares_sum_w = [np.zeros(w.shape) for w in self.weights]
        self.gradient_squares_sum_b = [np.zeros(b.shape) for b in self.biases]
        self.previous_momentum_w = [np.zeros(w.shape) for w in self.weights]
        self.previous_momentum_b = [np.zeros(b.shape) for b in self.biases]

        self.validation_percentiles = []
        self.testing_percentiles = []

    def two_dimensional_move_to(self, x, y):
        self.weights[1][9][14] = x
        self.weights[1][4][14] = y

    def two_dimensional_move_by(self, x, y):
        self.weights[1][9][14] += x
        self.weights[1][4][14] += y

    def two_dimensional_get_cords(self):
        return self.weights[1][9][14], self.weights[1][4][14]

    def get_total_loss_on_dataset_on_current_parameters(self, dataset):
        rs = 0
        for input_data, desired_output in dataset:
            output_activations = self.feedforward(input_data)
            rs += cost_function(output_activations, desired_output)
        rs /= len(dataset)
        rs = sum(rs) / len(rs)
        return rs[0]

    def set_parameters_of_neural_network_to_certain_values(self, nn_parameters):
        """this is a helper function for implementing Nelder Mead...
        nn_parameters should be ndarray of size (n weights + n biases, 1)
        where there are biases and then weights (concrete order can be found in NM implementation)"""
        ix = 0
        for i, br in enumerate(self.biases):
            for j, b in enumerate(br):
                self.biases[i][j] = nn_parameters[ix]
                ix += 1
        for i, wr in enumerate(self.weights):
            for j, wrr in enumerate(wr):
                for k, w in enumerate(wrr):
                    self.weights[i][j, k] = nn_parameters[ix]
                    ix += 1

    def get_total_loss_on_dataset_on_certain_parameters(self, dataset, nn_parameters):
        """this is a helper function for implementing Nelder Mead...
        nn_parameters should be ndarray of size (n weights + n biases, 1)
        where there are biases and then weights (concrete order can be found in NM implementation)"""
        self.set_parameters_of_neural_network_to_certain_values(nn_parameters)

        return self.get_total_loss_on_dataset_on_current_parameters(dataset)

    def init_from_json(self, file_name: str):
        with open(file_name, 'r') as infile:
            data = json.load(infile)
        self.weights = [np.array(w) for w in data['weights']]
        self.biases = [np.array(b) for b in data['biases']]

    def save_to_file(self, file_name: str):
        json_string = json.dumps({
            'weights': [w.tolist() for w in self.weights],
            'biases': [b.tolist() for b in self.biases]
        })
        with open(file_name, 'w') as outfile:
            outfile.write(json_string)

    def get_training_stats(self):
        return {
            'validation': self.validation_percentiles,
            'testing': self.testing_percentiles
        }

    def save_training_stats(self, file_name: str):
        json_string = json.dumps(self.get_training_stats())
        with open(file_name, 'w') as outfile:
            outfile.write(json_string)

    def feedforward(self, a: np.ndarray) -> np.ndarray:
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def train_using_nelder_mead(self, training_data: list, n_epochs: int, mini_batch_size: int, steps_per_epoch: int,
                                test_data: list = None,
                                validation_data: list = None) -> None:
        """:param steps_per_epoch: each epoch there will be this many steps of NM... batches will probably repeat
        :param training_data: list of tuples (training_input, desired_output)
        :param n_epochs:
        :param mini_batch_size:
        :param test_data: if test_data is provided partial progress will be printed
        :param validation_data: if validation_data is provided partial progress will be printed - the network also
        trains on these data
        :return: None"""
        n_test_data = len(test_data) if test_data else 0
        n_validation_data = len(test_data) if test_data else 0
        n = len(training_data)

        alpha, gamma, ro, sigma = 1, 2, 1 / 2, 1 / 2
        nn_parameters = []
        for br in self.biases:
            for b in br:
                nn_parameters.append(b[0])
        for wr in self.weights:
            for wrr in wr:
                for w in wrr:
                    nn_parameters.append(w)
        nn_parameters_size = len(nn_parameters)
        centroid = np.asarray(nn_parameters).reshape((nn_parameters_size, 1))
        simplex = []
        NelderMeadHelper.get_initial_simplex(simplex, centroid, nn_parameters_size, 1.0)

        for i in range(n_epochs):
            j = 0
            while j < steps_per_epoch:
                shuffle(training_data)
                mini_batches = [training_data[k: k + mini_batch_size] for k in range(0, n, mini_batch_size)]

                for mini_batch in mini_batches:
                    def func(k):
                        return self.get_total_loss_on_dataset_on_certain_parameters(mini_batch, k)

                    simplex.sort(key=lambda k: func(k))
                    centroid = NelderMeadHelper.get_centroid_of_a_simplex(simplex[:-1])

                    reflected = NelderMeadHelper.get_reflected_point(centroid, simplex[-1], alpha)

                    f1, fr, fn = func(simplex[0]), func(reflected), func(simplex[-2])
                    if f1 <= fr < fn:
                        NelderMeadHelper.replace_worst_point_in_simplex(simplex, reflected)
                        j += 1
                        continue
                    if fr < f1:
                        expanded = NelderMeadHelper.get_expanded_point(centroid, reflected, gamma)
                        fe = func(expanded)
                        if fe < fr:
                            simplex.append(expanded)
                            j += 1
                            continue
                        NelderMeadHelper.replace_worst_point_in_simplex(simplex, reflected)
                        j += 1
                        continue
                    fn1 = func(simplex[-1])
                    if fr < fn1:
                        contracted = NelderMeadHelper.get_contracted_point_on_the_outside(centroid, reflected, ro)
                        fc = func(contracted)
                        if fc < fr:
                            NelderMeadHelper.replace_worst_point_in_simplex(simplex, contracted)
                            j += 1
                            continue
                    contracted = NelderMeadHelper.get_contracted_point_on_the_inside(centroid, reflected, ro)
                    fc = func(contracted)
                    if fc < fn1:
                        NelderMeadHelper.replace_worst_point_in_simplex(simplex, contracted)
                        j += 1
                        continue

                    NelderMeadHelper.replace_all_points_except_the_best(simplex, sigma)
                    j += 1
                    if j >= steps_per_epoch:
                        break
            nn_parameters = list(centroid)
            self.set_parameters_of_neural_network_to_certain_values(nn_parameters)
            print(self.get_total_loss_on_dataset_on_current_parameters(training_data))
            self.print_and_write_down_epoch_stats(i, n_test_data, n_validation_data, test_data, validation_data)

    def train_using_stochastic_gradient_descent(self, training_data: list, n_epochs: int, mini_batch_size: int,
                                                eta: float,
                                                test_data: list = None,
                                                validation_data: list = None,
                                                steps_times: list[float] = [],
                                                time_limit: float = inf
                                                ) -> None:

        """:param training_data: list of tuples (training_input, desired_output)
        :param n_epochs:
        :param mini_batch_size:
        :param eta: a learning rate
        :param test_data: if test_data is provided partial progress will be printed
        :param validation_data: if validation_data is provided partial progress will be printed - the network also
        trains on these data
        :param steps_times: on index i there is how much time it took to make (i+1)-th step of GD
        :param time_limit: time limit in seconds for calculation time (evaluation on test data excluded!)
        :return: None"""
        n_test_data = len(test_data) if test_data else 0
        n_validation_data = len(test_data) if test_data else 0
        n = len(training_data)

        total_time = 0
        self.validation_percentiles = []
        self.testing_percentiles = []

        for i in range(n_epochs):
            shuffle(training_data)
            mini_batches = [training_data[k: k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                step_start_time = time()
                self.train_using_stochastic_gradient_descent_on_a_single_mini_batch(mini_batch, eta)
                step_end_time = time()

                step_time = step_end_time - step_start_time
                steps_times.append(step_time)
                total_time += step_time

                if total_time >= time_limit:
                    break
            self.print_and_write_down_epoch_stats(i, n_test_data, n_validation_data, test_data, validation_data)
            print(f'Total time for batch size of {mini_batch_size} is {total_time} s.')
            if total_time >= time_limit:
                break

    def train_using_stochastic_gradient_descent_on_a_single_mini_batch(self, mini_batch: list, eta: float) -> None:
        """applies one step of stochastic gradient descent
        mini_batch is a list of tuples (training_input, desired_output)
        eta div k is a learning rate"""
        k = len(mini_batch)
        gradient_w = [np.zeros(w.shape) for w in self.weights]
        gradient_b = [np.zeros(b.shape) for b in self.biases]
        for training_input, desired_output in mini_batch:
            delta_nabla_w, delta_nabla_b = self.backpropagation(training_input, desired_output)
            gradient_w = [nw + dnw for nw, dnw in zip(gradient_w, delta_nabla_w)]
            gradient_b = [nb + dnb for nb, dnb in zip(gradient_b, delta_nabla_b)]
        # self.weights = [w - (eta div k / k) * nw for w, nw in zip(self.weights, gradient_w)]
        # self.biases = [b - (eta div k / k) * nb for b, nb in zip(self.biases, gradient_b)]
        self.weights = [w - eta * nw for w, nw in zip(self.weights, gradient_w)]
        self.biases = [b - eta * nb for b, nb in zip(self.biases, gradient_b)]

    def train_using_stochastic_gradient_descent_with_momentum(self, training_data: list, n_epochs: int,
                                                              mini_batch_size: int,
                                                              eta: float, mtm: float,
                                                              test_data: list = None,
                                                              validation_data: list = None) -> None:
        """
        :param training_data: list of tuples (training_input, desired_output)
        :param n_epochs:
        :param mini_batch_size:
        :param eta: a learning rate
        :param mtm: momentum factor - Many experiments have empirically verified the most appropriate
        setting for the momentum factor is 0.9 (sun.pdf, p. 6)... update is not only current gradient, but
        we additionally add mtm * previous update to that
        :param test_data: if test_data is provided partial progress will be printed
        :param validation_data: if validation_data is provided partial progress will be printed - the network also
        trains on these data
        :return: None
        """
        n_test_data = len(test_data) if test_data else 0
        n_validation_data = len(test_data) if test_data else 0
        n = len(training_data)

        self.previous_update_w = [np.zeros(w.shape) for w in self.weights]
        self.previous_update_b = [np.zeros(b.shape) for b in self.biases]

        for i in range(n_epochs):
            shuffle(training_data)
            mini_batches = [training_data[k: k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.train_using_stochastic_gradient_descent_with_momentum_on_a_single_mini_batch(mini_batch, eta, mtm)
            self.print_and_write_down_epoch_stats(i, n_test_data, n_validation_data, test_data, validation_data)

    def train_using_stochastic_gradient_descent_with_momentum_on_a_single_mini_batch(self, mini_batch: list,
                                                                                     eta: float, mtm: float) -> None:
        """applies one step of stochastic gradient descent with momentum
        :param mini_batch: is a list of tuples (training_input, desired_output)
        :param eta: is a learning rate
        :param mtm: momentum factor - Many experiments have empirically verified the most appropriate
        setting for the momentum factor is 0.9 (sun.pdf, p. 6)... update is not only current gradient, but
        we additionally add mtm * previous update to that"""
        k = len(mini_batch)
        gradient_w = [np.zeros(w.shape) for w in self.weights]
        gradient_b = [np.zeros(b.shape) for b in self.biases]

        # calculate gradient
        for training_input, desired_output in mini_batch:
            delta_nabla_w, delta_nabla_b = self.backpropagation(training_input, desired_output)
            gradient_w = [nw + dnw for nw, dnw in zip(gradient_w, delta_nabla_w)]
            gradient_b = [nb + dnb for nb, dnb in zip(gradient_b, delta_nabla_b)]

        # eta *= (1 - mtm)

        # calculate update
        # update_w = [- eta * nw + mtm * pnw for nw, pnw in zip(gradient_w, self.previous_update_w)]
        # update_b = [- eta * nb + mtm * pnb for nb, pnb in zip(gradient_b, self.previous_update_b)]

        # calculate update like we did on simpler functions
        update_w = [- (1 - mtm) * eta * nw + mtm * pnw for nw, pnw in zip(gradient_w, self.previous_update_w)]
        update_b = [- (1 - mtm) * eta * nb + mtm * pnb for nb, pnb in zip(gradient_b, self.previous_update_b)]

        # update
        self.weights = [w + uw for w, uw in zip(self.weights, update_w)]
        self.biases = [b + ub for b, ub in zip(self.biases, update_b)]

        # remember update
        self.previous_update_w = update_w
        self.previous_update_b = update_b

    def train_using_nesterov(self, training_data: list, n_epochs: int,
                             mini_batch_size: int,
                             eta: float, mtm: float,
                             test_data: list = None,
                             validation_data: list = None) -> None:
        """
        :param training_data: list of tuples (training_input, desired_output)
        :param n_epochs:
        :param mini_batch_size:
        :param eta: a learning rate
        :param mtm: momentum factor - Many experiments have empirically verified the most appropriate
        setting for the momentum factor is 0.9 (sun.pdf, p. 6)... update is not only current gradient, but
        we additionally add mtm * previous update to that
        :param test_data: if test_data is provided partial progress will be printed
        :param validation_data: if validation_data is provided partial progress will be printed - the network also
        trains on these data
        :return: None
        """
        n_test_data = len(test_data) if test_data else 0
        n_validation_data = len(test_data) if test_data else 0
        n = len(training_data)

        self.previous_update_w = [np.zeros(w.shape) for w in self.weights]
        self.previous_update_b = [np.zeros(b.shape) for b in self.biases]

        for i in range(n_epochs):
            shuffle(training_data)
            mini_batches = [training_data[k: k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.train_using_nesterov_on_a_single_mini_batch(mini_batch, eta, mtm)
            self.print_and_write_down_epoch_stats(i, n_test_data, n_validation_data, test_data, validation_data)

    def train_using_nesterov_on_a_single_mini_batch(self, mini_batch: list,
                                                    eta: float, mtm: float) -> None:
        """applies one step of stochastic gradient descent with momentum
        :param mini_batch: is a list of tuples (training_input, desired_output)
        :param eta: is a learning rate
        :param mtm: momentum factor - Many experiments have empirically verified the most appropriate
        setting for the momentum factor is 0.9 (sun.pdf, p. 6)... update is not only current gradient, but
        we additionally add mtm * previous update to that"""
        k = len(mini_batch)
        gradient_w = [np.zeros(w.shape) for w in self.weights]
        gradient_b = [np.zeros(b.shape) for b in self.biases]

        # move to "future position" (determined by past updates)
        self.weights = [w + mtm * pnw for w, pnw in zip(self.weights, self.previous_update_w)]
        self.biases = [b + mtm * pnb for b, pnb in zip(self.biases, self.previous_update_b)]

        # calculate gradient
        for training_input, desired_output in mini_batch:
            delta_nabla_w, delta_nabla_b = self.backpropagation(training_input, desired_output)
            gradient_w = [nw + dnw for nw, dnw in zip(gradient_w, delta_nabla_w)]
            gradient_b = [nb + dnb for nb, dnb in zip(gradient_b, delta_nabla_b)]

        # move back to "current position"
        self.weights = [w - mtm * pnw for w, pnw in zip(self.weights, self.previous_update_w)]
        self.biases = [b - mtm * pnb for b, pnb in zip(self.biases, self.previous_update_b)]

        eta *= (1 - mtm)

        # calculate update
        update_w = [- eta * nw + mtm * pnw for nw, pnw in zip(gradient_w, self.previous_update_w)]
        update_b = [- eta * nb + mtm * pnb for nb, pnb in zip(gradient_b, self.previous_update_b)]

        # calculate update like we did on simpler functions
        # update_w = [- (eta / k) * nw + mtm * pnw for nw, pnw in zip(gradient_w, self.previous_update_w)]
        # update_b = [- (eta / k) * nb + mtm * pnb for nb, pnb in zip(gradient_b, self.previous_update_b)]

        # update
        self.weights = [w + uw for w, uw in zip(self.weights, update_w)]
        self.biases = [b + ub for b, ub in zip(self.biases, update_b)]

        # remember update
        self.previous_update_w = update_w
        self.previous_update_b = update_b

    def train_using_adagrad(self, training_data: list, n_epochs: int,
                            mini_batch_size: int,
                            eta: float,
                            test_data: list = None,
                            validation_data: list = None) -> None:
        """
                :param training_data: list of tuples (training_input, desired_output)
                :param n_epochs:
                :param mini_batch_size:
                :param eta: a learning rate
                setting for the momentum factor is 0.9 (sun.pdf, p. 6)... update is not only current gradient, but
                we additionally add mtm * previous update to that
                :param test_data: if test_data is provided partial progress will be printed
                :param validation_data: if validation_data is provided partial progress will be printed - the network also
                trains on these data
                :return: None
                """
        n_test_data = len(test_data) if test_data else 0
        n_validation_data = len(test_data) if test_data else 0
        n = len(training_data)

        self.previous_update_w = [np.zeros(w.shape) for w in self.weights]
        self.previous_update_b = [np.zeros(b.shape) for b in self.biases]
        self.gradient_squares_sum_w = [np.zeros(w.shape) for w in self.weights]
        self.gradient_squares_sum_b = [np.zeros(b.shape) for b in self.biases]

        for i in range(n_epochs):
            shuffle(training_data)
            mini_batches = [training_data[k: k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.train_using_adagrad_on_a_single_mini_batch(mini_batch, eta)
            self.print_and_write_down_epoch_stats(i, n_test_data, n_validation_data, test_data, validation_data)

    def train_using_adagrad_on_a_single_mini_batch(self, mini_batch: list,
                                                   eta: float) -> None:
        """applies one step of stochastic gradient descent with momentum
        :param mini_batch: is a list of tuples (training_input, desired_output)
        :param eta: is a learning rate
        """
        k = len(mini_batch)
        eps = 1
        gradient_w = [np.zeros(w.shape) for w in self.weights]
        gradient_b = [np.zeros(b.shape) for b in self.biases]

        # calculate gradient
        for training_input, desired_output in mini_batch:
            delta_nabla_w, delta_nabla_b = self.backpropagation(training_input, desired_output)
            gradient_w = [nw + dnw for nw, dnw in zip(gradient_w, delta_nabla_w)]
            gradient_b = [nb + dnb for nb, dnb in zip(gradient_b, delta_nabla_b)]

        self.gradient_squares_sum_w = [pss + nw ** 2 for pss, nw in zip(self.gradient_squares_sum_w, gradient_w)]
        self.gradient_squares_sum_b = [pss + nb ** 2 for pss, nb in zip(self.gradient_squares_sum_b, gradient_b)]

        vw = [(eps + gss) * 0.5 for gss in self.gradient_squares_sum_w]
        vb = [(eps + gss) * 0.5 for gss in self.gradient_squares_sum_b]

        # calculate update
        update_w = [- eta * nw / v for nw, v in zip(gradient_w, vw)]
        update_b = [- eta * nb / v for nb, v in zip(gradient_b, vb)]
        # update
        self.weights = [w + uw for w, uw in zip(self.weights, update_w)]
        self.biases = [b + ub for b, ub in zip(self.biases, update_b)]

    def train_using_adam(self, training_data: list, n_epochs: int,
                         mini_batch_size: int,
                         eta: float, beta1: float, beta2: float,
                         test_data: list = None,
                         validation_data: list = None) -> None:
        """
        :param training_data: list of tuples (training_input, desired_output)
        :param n_epochs:
        :param mini_batch_size:
        :param eta: a learning rate
        :param beta1: momentum factor - coefficient for exponentially decaying average
        :param beta2: another exponential decay rate - for decreasing gradient step
        :param test_data: if test_data is provided partial progress will be printed
        :param validation_data: if validation_data is provided partial progress will be printed - the network also
        trains on these data
        :return: None
        """
        n_test_data = len(test_data) if test_data else 0
        n_validation_data = len(test_data) if test_data else 0
        n = len(training_data)

        self.previous_momentum_w = [np.zeros(w.shape) for w in self.weights]
        self.previous_momentum_b = [np.zeros(b.shape) for b in self.biases]
        self.gradient_squares_sum_w = [np.zeros(w.shape) for w in self.weights]
        self.gradient_squares_sum_b = [np.zeros(b.shape) for b in self.biases]

        for i in range(n_epochs):
            shuffle(training_data)
            mini_batches = [training_data[k: k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.train_using_adam_on_a_single_mini_batch(mini_batch, eta, beta1, beta2)
            self.print_and_write_down_epoch_stats(i, n_test_data, n_validation_data, test_data, validation_data)

    def train_using_adam_on_a_single_mini_batch(self, mini_batch: list,
                                                eta: float, beta1: float, beta2: float) -> None:
        """applies one step of stochastic gradient descent with momentum
        :param mini_batch: is a list of tuples (training_input, desired_output)
        :param eta: is a learning rate
        :param beta1: momentum factor - coefficient for exponentially decaying average
        :param beta2: another exponential decay rate - for decreasing gradient step
        """
        k = len(mini_batch)
        eps = 0.1
        gradient_w = [np.zeros(w.shape) for w in self.weights]
        gradient_b = [np.zeros(b.shape) for b in self.biases]
        eta *= (1 - beta1)

        # calculate gradient
        for training_input, desired_output in mini_batch:
            delta_nabla_w, delta_nabla_b = self.backpropagation(training_input, desired_output)
            gradient_w = [nw + dnw for nw, dnw in zip(gradient_w, delta_nabla_w)]
            gradient_b = [nb + dnb for nb, dnb in zip(gradient_b, delta_nabla_b)]

        self.previous_momentum_w = [(1 - beta1) * nw + beta1 * mw for nw, mw in
                                    zip(self.previous_momentum_w, gradient_w)]
        self.previous_momentum_b = [(1 - beta1) * nb + beta1 * mb for nb, mb in
                                    zip(self.previous_momentum_b, gradient_b)]

        self.gradient_squares_sum_w = [((1 - beta2) * (nw ** 2) + beta2 * pss) ** 0.5 for pss, nw in
                                       zip(self.gradient_squares_sum_w, gradient_w)]
        self.gradient_squares_sum_b = [((1 - beta2) * (nb ** 2) + beta2 * pss) ** 0.5 for pss, nb in
                                       zip(self.gradient_squares_sum_b, gradient_b)]

        # calculate update
        update_w = [- eta * mw / (vw + eps) * ((1 - beta2) ** 0.5) / (1 - beta1) for mw, vw in zip(
            self.previous_momentum_w, self.gradient_squares_sum_w)]
        update_b = [- eta * mb / (vb + eps) * ((1 - beta2) ** 0.5) / (1 - beta1) for mb, vb in zip(
            self.previous_momentum_b, self.gradient_squares_sum_b)]
        # update
        self.weights = [w + uw for w, uw in zip(self.weights, update_w)]
        self.biases = [b + ub for b, ub in zip(self.biases, update_b)]

    def print_and_write_down_epoch_stats(self, i, n_test_data, n_validation_data, test_data, validation_data):
        if validation_data and test_data:
            n_correct_results_on_validation_data = self.evaluate(validation_data)
            percentile_validation = n_correct_results_on_validation_data / n_validation_data * 100
            n_correct_results_on_testing_data = self.evaluate(test_data)
            percentile_testing = n_correct_results_on_testing_data / n_test_data * 100
            print(
                f'Epoch {i + 1} | Validation {n_correct_results_on_validation_data}/{n_validation_data} ~ {percentile_validation}% | Testing {n_correct_results_on_testing_data}/{n_test_data} ~ {percentile_testing}%'
            )
            self.testing_percentiles.append(percentile_testing)
            self.validation_percentiles.append(percentile_validation)
        else:
            print(f'Training epoch {i + 1} completed.')
            pass

    def backpropagation(self, training_input: np.ndarray, desired_output: np.ndarray) -> tuple:
        """"""
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        activation = training_input
        activations = [training_input]
        list_of_z_vectors = []

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            list_of_z_vectors.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # partial derivatives of cost function with respect to activations from the last layer
        delta = cost_function_derivative(activations[-1], desired_output) * sigmoid_derivative(list_of_z_vectors[-1])
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        nabla_b[-1] = delta

        for i in range(2, self.n_layers):
            z = list_of_z_vectors[-i]
            sd = sigmoid_derivative(z)
            delta = np.dot(self.weights[-i + 1].transpose(), delta) * sd
            nabla_w[-i] = np.dot(delta, activations[-i - 1].transpose())
            nabla_b[-i] = delta
        return nabla_w, nabla_b

    def evaluate(self, data: list) -> int:
        """returns number of correctly evaluated data """
        results = [(np.argmax(self.feedforward(x)), y) for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

    def __str__(self):
        rs = 'Weights:\n'
        for i, w in enumerate(self.weights):
            rs += f'Layer {i + 1}:\n'
            rs += str(w) + '\n'
        rs += 'Biases:\n'
        for i, b in enumerate(self.biases):
            rs += f'Layer {i + 1}:\n'
            rs += str(b) + '\n'
        return rs
