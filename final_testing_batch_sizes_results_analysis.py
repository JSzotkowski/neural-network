from json import load, dump

if __name__ == '__main__':
    with open(f"/home/jiri/PycharmProjects/NeuralNetwork/final_testing_batch_sizes/best_of_three_data.json",
              "r") as infile:
        dicts = load(infile)

    for bs, d in dicts.items():
        print(f"{bs} & {d['final_testing_data_accuracy']:.2f} & {d['total_time_training']:.2f} & {int(30 * 60000 / int(bs))} \\\\")

    print("Done!")
