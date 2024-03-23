from NeuralNetwork import NeuralNetwork
from json import load, dump
from time import time
import mnist_loader


def get_empty_testing_dict():
    ans = dict()

    # for bs in [1, 2, 4, 8, 16, 32, 64, 128, 512, 2048, 16384, 60000]:
    for bs in [2048]:
        ans[bs] = {
            "batch_size": bs,
            "is_testing_done": False,
            "final_testing_data_accuracy": None,
            "final_validation_data_accuracy": None,
            "testing_data_accuracy_list": [],
            "validation_data_accuracy_list": [],
            "total_time_training": None
        }

    return ans


def reset(run_number):
    testing_dict = get_empty_testing_dict()

    with open(f"/home/jiri/PycharmProjects/NeuralNetwork/final_testing_batch_sizes/data{run_number}.json",
              "w") as outfile:
        dump(testing_dict, outfile, indent=4)

    net = NeuralNetwork(network_layout)
    net.save_to_file(
        f"/home/jiri/PycharmProjects/NeuralNetwork/final_testing_batch_sizes/randomly_initialized_network{run_number}.txt"
    )


def test(run_number):
    with open(f"/home/jiri/PycharmProjects/NeuralNetwork/final_testing_batch_sizes/data{run_number}.json",
              "r") as infile:
        testing_dict = load(infile)

    for bs, out_dict in testing_dict.items():
        if out_dict["is_testing_done"] is True:
            continue

        batch_size = out_dict["batch_size"]
        net = NeuralNetwork(network_layout)
        net.init_from_json(
            f"/home/jiri/PycharmProjects/NeuralNetwork/final_testing_batch_sizes/randomly_initialized_network{run_number}.txt"
        )
        start_time_of_training = time()
        net.train_using_stochastic_gradient_descent(training_data, epochs, batch_size, 0.3, test_data, validation_data)
        end_time_of_training = time()

        training_duration = end_time_of_training - start_time_of_training

        training_stats = net.get_training_stats()

        out_dict["is_testing_done"] = True
        out_dict["final_testing_data_accuracy"] = training_stats["testing"][-1]
        out_dict["final_validation_data_accuracy"] = training_stats["validation"][-1]
        out_dict["testing_data_accuracy_list"] = training_stats["testing"]
        out_dict["validation_data_accuracy_list"] = training_stats["validation"]
        out_dict["total_time_training"] = training_duration

        with open(f"/home/jiri/PycharmProjects/NeuralNetwork/final_testing_batch_sizes/data{run_number}.json",
                  "w") as outfile:
            dump(testing_dict, outfile, indent=4)

        print(10 * "=" + f"Training with batch size {bs} is completed." + 10 * "=")
        print(f"Total duration: {training_duration} | Average time per epoch: {training_duration / epochs}")
        print(
            f"Final testing data accuracy: {training_stats['testing'][-1]} | Final training data accuracy: {training_stats['validation'][-1]}")
        print()


def analyze_results():
    testing_dicts = []
    for i in range(3):
        with open(f"/home/jiri/PycharmProjects/NeuralNetwork/final_testing_batch_sizes/data{i + 1}.json",
                  "r") as infile:
            testing_dicts.append(load(infile))

    result_dict = get_empty_testing_dict()

    for bs, _ in result_dict.items():
        temp = [testing_dicts[i][str(bs)] for i in range(3)]
        result_dict[bs] = max(temp, key=lambda k: k["final_testing_data_accuracy"])

    with open(f"/home/jiri/PycharmProjects/NeuralNetwork/final_testing_batch_sizes/best_of_three_data.json",
              "w") as outfile:
        dump(result_dict, outfile, indent=4)


if __name__ == '__main__':
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    epochs = 30
    network_layout = [784, 30, 10]

    
    if False:
        for i in range(3):
            reset(i + 1)
    else:
        for i in range(3):
            test(i + 1)
        analyze_results()

    print("Done!")
