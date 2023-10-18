from common_functions import get_folder_name_for_this_run, create_folder_if_it_does_not_exits, \
    print_runtime_and_write_to_time_log, print_and_save_time_log
from NeuralNetwork import NeuralNetwork
from time import time
import mnist_loader
import os

if __name__ == '__main__':
    epochs = 30
    batch_size = 10
    network_layout = [784, 30, 10]

    time_log = ""

    # Load mnist database
    start_time = time()
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    run_folder_name = get_folder_name_for_this_run()
    create_folder_if_it_does_not_exits(run_folder_name)
    end_time = time()
    time_log = print_runtime_and_write_to_time_log(time_log, start_time, end_time, 'mnist database initialization')

    # SGD
    # start_time = time()
    net = NeuralNetwork(network_layout)
    net.save_to_file(os.path.join(run_folder_name, 'randomly_initialized_network.txt'))
    # net.train_using_stochastic_gradient_descent(training_data, epochs, batch_size, 0.3, test_data, validation_data)
    # net.save_to_file(os.path.join(run_folder_name, 'trained_network_SGD.txt'))
    # # net.save_training_stats(os.path.join(run_folder_name, 'SGD_stats.txt'))
    # end_time = time()
    # time_log = print_runtime_and_write_to_time_log(time_log, start_time, end_time, 'SGD')

    # SGD with momentum
    start_time = time()
    net = NeuralNetwork(network_layout)
    net.init_from_json(os.path.join(run_folder_name, 'randomly_initialized_network.txt'))
    # net.train_using_stochastic_gradient_descent_with_momentum(training_data, epochs, batch_size, 0.075, 0.75, test_data,
    #                                                           validation_data)
    net.train_using_stochastic_gradient_descent_with_momentum(training_data, epochs, batch_size, 0.3, 0.8, test_data,
                                                              validation_data)
    net.save_to_file(os.path.join(run_folder_name, 'trained_network_SGD_with_momentum.txt'))
    net.save_stats(os.path.join(run_folder_name, 'SGD_with_momentum_stats.txt'))
    # end_time = time()
    time_log = print_runtime_and_write_to_time_log(time_log, start_time, end_time, 'SGD with momentum')
    #
    # # NAG - Nesterov accelerated gradient descent
    # start_time = time()
    # net = NeuralNetwork(network_layout)
    # net.init_from_json(os.path.join(run_folder_name, 'randomly_initialized_network.txt'))
    # net.train_using_nesterov(training_data, epochs, batch_size, 0.008, 0.99, test_data, validation_data)
    # net.save_to_file(os.path.join(run_folder_name, 'trained_network_NAG.txt'))
    # net.save_stats(os.path.join(run_folder_name, 'NAG_stats.txt'))
    # end_time = time()
    # time_log = print_runtime_and_write_to_time_log(time_log, start_time, end_time, 'NAG')

    print_and_save_time_log(time_log, os.path.join(run_folder_name, 'time_log.txt'))
