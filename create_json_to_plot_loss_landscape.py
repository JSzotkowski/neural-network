import mnist_loader
import numpy
import json
from NeuralNetwork import NeuralNetwork
from time import time

if __name__ == '__main__':
    net = NeuralNetwork([784, 30, 10])
    net.init_from_json('/home/jiri/PycharmProjects/NeuralNetwork/runs/mon_02_october_09_35_26/trained_network_SGD.txt')

    # somehow it could be related to matrix/vector norms... mby try it
    # print(numpy.linalg.norm(net.weights[0][0]))
    # print(numpy.linalg.norm(net.weights[1][0]))

    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    test_data = mnist_loader.vectorize_results_of_test_data(test_data)
    validation_data = mnist_loader.vectorize_results_of_test_data(validation_data)

    print(net.get_total_loss_on_dataset_on_current_parameters(test_data))
    print(net.get_total_loss_on_dataset_on_current_parameters(validation_data))

    start = time()
    evaluations = []
    length = 50
    net.two_dimensional_move_to(-4.1, -2.47)
    # net.two_dimensional_move_to(0, 0)
    net.two_dimensional_move_by(- length / 2, - length / 2)
    iter = 50
    delta = length / iter
    for i in range(iter):
        evaluations.append([])
        for j in range(iter):
            print(i * iter + j + 1)
            net.two_dimensional_move_by(delta, 0)
            evaluations[i].append(
                (*net.two_dimensional_get_cords(), net.get_total_loss_on_dataset_on_current_parameters(validation_data)))
        net.two_dimensional_move_by(-length, delta)
    end = time()

    # print(f'Total duration: {end - start}')
    # with open('/home/jiri/PycharmProjects/NeuralNetwork/terrain_to_visualize.json', 'w') as outfile:
    #     json.dump(evaluations, outfile, indent=4)

    # times = []
    #
    # for _ in range(300):
    #     net = NeuralNetwork([784, 30, 10])
    #     start = time()
    #     print(f'Total loss on a test_data dataset: {net.get_total_loss_on_dataset_on_current_parameters(test_data)}')
    #     end = time()
    #     times.append(end - start)
    #
    # print('=' * 30)
    # print(f'Loss evaluations stats:')
    # print(f'Fastest: {min(times)}')
    # print(f'Slowest: {max(times)}')
    # print(f'Average: {sum(times) / len(times)}')
    # print(f'Median: {list(sorted(times))[len(times) // 2]}')
