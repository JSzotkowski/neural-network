from json import load, dump
from time import time

import mnist_loader
from NeuralNetwork import NeuralNetwork

SGD_ETA = [0.01, 0.15, 0.3, 1.0]
SGDM_NAG_ETA = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
ADAGRAD_ETA = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
ADAM_ETA = [0.005, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
BETA1 = [0.9, 0.7]
BETA2 = [0.999, 0.9]
ADAGRAD_EPS = [0.1, 0.0001]


def get_empty_testing_dict():
    ans = dict()

    # SGD
    ans['SGD'] = []
    for eta in SGD_ETA:
        ans['SGD'].append({
            "method-name": f'SGD',
            "eta": eta,
            "beta1": None,
            "beta2": None,
            "eps": None,
            "is_testing_done": False,
            "final_testing_data_accuracy": None,
            "final_validation_data_accuracy": None,
            "testing_data_accuracy_list": [],
            "validation_data_accuracy_list": [],
            "total_time_training": None
        })

    # SGDM + NAG
    ans['SGDM'] = []
    ans['NAG'] = []
    for eta in SGDM_NAG_ETA:
        for beta1 in BETA1:
            ans['SGDM'].append({
                "method-name": f'SGDM',
                "eta": eta,
                "beta1": beta1,
                "beta2": None,
                "eps": None,
                "is_testing_done": False,
                "final_testing_data_accuracy": None,
                "final_validation_data_accuracy": None,
                "testing_data_accuracy_list": [],
                "validation_data_accuracy_list": [],
                "total_time_training": None
            })
            ans['NAG'].append({
                "method-name": f'NAG',
                "eta": eta,
                "beta1": beta1,
                "beta2": None,
                "eps": None,
                "is_testing_done": False,
                "final_testing_data_accuracy": None,
                "final_validation_data_accuracy": None,
                "testing_data_accuracy_list": [],
                "validation_data_accuracy_list": [],
                "total_time_training": None
            })

    # AdaGrad
    ans['AdaGrad'] = []
    for eta in ADAGRAD_ETA:
        for eps in ADAGRAD_EPS:
            ans['AdaGrad'].append({
                "method-name": f'AdaGrad',
                "eta": eta,
                "beta1": None,
                "beta2": None,
                "eps": eps,
                "is_testing_done": False,
                "final_testing_data_accuracy": None,
                "final_validation_data_accuracy": None,
                "testing_data_accuracy_list": [],
                "validation_data_accuracy_list": [],
                "total_time_training": None
            })

    # Adam
    ans['Adam'] = []
    for eta in ADAM_ETA:
        for beta1 in BETA1:
            for beta2 in BETA2:
                ans['Adam'].append({
                    "method-name": f'Adam',
                    "eta": eta,
                    "beta1": beta1,
                    "beta2": beta2,
                    "eps": 0.1,
                    "is_testing_done": False,
                    "final_testing_data_accuracy": None,
                    "final_validation_data_accuracy": None,
                    "testing_data_accuracy_list": [],
                    "validation_data_accuracy_list": [],
                    "total_time_training": None
                })

    return ans


def reset(run_number):
    testing_dict = get_empty_testing_dict()

    with open(f"/home/jiri/PycharmProjects/NeuralNetwork/final_testing/data{run_number}.json",
              "w") as outfile:
        dump(testing_dict, outfile, indent=4)

    net = NeuralNetwork(network_layout)
    net.save_to_file(
        f"/home/jiri/PycharmProjects/NeuralNetwork/final_testing/randomly_initialized_network{run_number}.txt"
    )


def test(run_number):
    done = 0

    with open(f"/home/jiri/PycharmProjects/NeuralNetwork/final_testing/data{run_number}.json",
              "r") as infile:
        testing_dict = load(infile)

    for method_name, list_of_method_variants in testing_dict.items():
        for out_dict in list_of_method_variants:
            done += 1
            if out_dict["is_testing_done"] is True:
                continue

            net = NeuralNetwork(network_layout)
            net.init_from_json(
                f"/home/jiri/PycharmProjects/NeuralNetwork/final_testing/randomly_initialized_network{run_number}.txt"
            )
            mn = out_dict["method-name"]
            eta = out_dict["eta"]
            beta1 = out_dict["beta1"]
            beta2 = out_dict["beta2"]
            eps = out_dict["eps"]
            start_time_of_training = time()
            if mn == "SGD":
                net.train_using_stochastic_gradient_descent(training_data, epochs, batch_size, eta, test_data, validation_data)
            elif mn == "SGDM":
                net.train_using_stochastic_gradient_descent_with_momentum(training_data, epochs, batch_size, eta, beta1, test_data, validation_data)
            elif mn == "NAG":
                net.train_using_nesterov(training_data, epochs, batch_size, eta, beta1, test_data, validation_data)
            elif mn == "AdaGrad":
                net.train_using_adagrad(training_data, epochs, batch_size, eta, eps, test_data, validation_data)
            elif mn == "Adam":
                net.train_using_adam(training_data, epochs, batch_size, eta, beta1, beta2, eps, test_data, validation_data)
            else:
                print("Something's wrong, I can feel it!")
                exit(1)
            end_time_of_training = time()

            training_duration = end_time_of_training - start_time_of_training

            training_stats = net.get_training_stats()

            out_dict["is_testing_done"] = True
            out_dict["final_testing_data_accuracy"] = training_stats["testing"][-1]
            out_dict["final_validation_data_accuracy"] = training_stats["validation"][-1]
            out_dict["testing_data_accuracy_list"] = training_stats["testing"]
            out_dict["validation_data_accuracy_list"] = training_stats["validation"]
            out_dict["total_time_training"] = training_duration

            with open(f"/home/jiri/PycharmProjects/NeuralNetwork/final_testing/data{run_number}.json",
                      "w") as outfile:
                dump(testing_dict, outfile, indent=4)

            print()
            print(10 * "=" + f"Training with {mn}, eta: {eta}, beta1&2: {beta1}&{beta2} is completed." + 10 * "=")
            print(f"Total duration: {training_duration} | Average time per epoch: {training_duration / epochs}")
            print(
                f"Final testing data accuracy: {training_stats['testing'][-1]} | Final training data accuracy: {training_stats['validation'][-1]}")
            print()
            print(f'Run {run_number} / 3; Done: {done} / 80')
            print()
            print()
            print()


def analyze_results():
    testing_dicts = []
    for i in range(3):
        with open(f"/home/jiri/PycharmProjects/NeuralNetwork/final_testing/data{i + 1}.json",
                  "r") as infile:
            testing_dicts.append(load(infile))

    result_dict = get_empty_testing_dict()

    for method_name, list_of_method_variants in result_dict.items():
        result_dict[method_name] = []
        for ix, variant_dict in enumerate(list_of_method_variants):
            temp = [testing_dicts[i][method_name][ix] for i in range(3)]
            result_dict[method_name].append(max(temp, key=lambda k: k["final_testing_data_accuracy"]))

    with open(f"/home/jiri/PycharmProjects/NeuralNetwork/final_testing/best_of_three_data.json",
              "w") as outfile:
        dump(result_dict, outfile, indent=4)


if __name__ == '__main__':
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    batch_size = 16
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
