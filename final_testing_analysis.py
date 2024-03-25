from json import load, dump

if __name__ == '__main__':
    with open(f"/home/jiri/PycharmProjects/NeuralNetwork/final_testing/best_of_three_data.json",
              "r") as infile:
        data = load(infile)

    dicts = dict()
    for method_name, list_of_method_variants in data.items():
        dicts[method_name] = max(list_of_method_variants, key=lambda k: k["final_testing_data_accuracy"])

    results_to_sort = [d for d in dicts.values()]

    results_to_sort.sort(key=lambda k: -k["final_testing_data_accuracy"])

    for rs in results_to_sort:
        eta = rs["eta"] if rs["eta"] is not None else "\\mc{---}"
        beta1 = rs["beta1"] if rs["beta1"] is not None else "\\mc{---}"
        beta2 = rs["beta2"] if rs["beta2"] is not None else "\\mc{---}"
        eps = rs["eps"] if rs["eps"] is not None else "\\mc{---}"
        print(f"{rs['method-name']} & {eta}  & {beta1}  & {beta2}  & {eps} & {rs['final_testing_data_accuracy']:.2f} & {rs['total_time_training']:.2f}\\\\")

    with open(f"/home/jiri/PycharmProjects/NeuralNetwork/final_testing/best_hyperparameters.json", "w") as outfile:
        dump(dicts, outfile)

    print("Done!")
