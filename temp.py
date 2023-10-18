import json

if __name__ == '__main__':
    with open("/standard_vs_stoch/eta div k/1696498851.3823464.json", "r") as infile:
        d = json.load(infile)

    rs = []
    for key, value in d.items():
        value["batch_size"] = key
        rs.append(value)

    with open("/standard_vs_stoch/eta div k/list_variant.json", "w") as outfile:
        json.dump(rs, outfile, indent=4)
