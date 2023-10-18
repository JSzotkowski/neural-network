from NeuralNetwork import NeuralNetwork
from time import time
import mnist_loader
import json

if __name__ == '__main__':
    MAX_EPOCHS = 30

    net = NeuralNetwork([784, 30, 10])

    # Load mnist database
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    stats = dict()

    for batch_s in [1, 5, 10, 15, 30, 50, 100, 1000, 20000, 60000]:
        net = NeuralNetwork([784, 30, 10])
        net.init_from_json(
            '/home/jiri/PycharmProjects/NeuralNetwork/runs/mon_02_october_09_35_26/randomly_initialized_network.txt'
        )

        steps_times = []

        net.train_using_stochastic_gradient_descent(training_data, MAX_EPOCHS, batch_s, 0.3, test_data, validation_data, steps_times=steps_times, time_limit=60)
        net_stats = net.get_training_stats()

        stats[batch_s] = {
            "steps_times_stats": {
                "max": max(steps_times),
                "min": min(steps_times),
                "average": sum(steps_times) / len(steps_times)
            },
            "validation_percentiles": net_stats["validation"],
            "testing_percentiles": net_stats["testing"],
            "total_number_of_steps": len(steps_times),
        }

    with open(f'/home/jiri/PycharmProjects/NeuralNetwork/standard_vs_stoch/{time()}.json', 'w') as outfile:
        json.dump(stats, outfile, indent=4)

