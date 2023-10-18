import numpy as np


def load_digit() -> np.ndarray:
    rs = []
    with open('digit.txt', 'r') as infile:
        for line in infile.readlines():
            rs.extend(list(map(float, list(line.strip()))))
    return np.reshape(rs, (784, 1))


if __name__ == '__main__':
    print(load_digit())
