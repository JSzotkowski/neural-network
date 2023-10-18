import numpy as np


class NelderMeadHelper:
    @staticmethod
    def replace_worst_point_in_simplex(simplex, new_point):
        simplex.pop()
        simplex.append(new_point)

    @staticmethod
    def replace_all_points_except_the_best(simplex, sigma):
        x1 = simplex[0]
        for i, xi in enumerate(simplex):
            if i == 0:
                continue
            shrunk = x1.copy()
            shrunk += sigma * (xi - x1)
            simplex[i] = shrunk

    @staticmethod
    def get_contracted_point_on_the_inside(centroid, worst, ro):
        rs = centroid.copy()
        rs += ro * (worst - centroid)
        return rs

    @staticmethod
    def get_contracted_point_on_the_outside(centroid, reflected, ro):
        rs = centroid.copy()
        rs += ro * (reflected - centroid)
        return rs

    @staticmethod
    def get_expanded_point(centroid, reflected, gamma):
        rs = centroid.copy()
        rs += gamma * (reflected - centroid)
        return rs

    @staticmethod
    def get_reflected_point(centroid, worst, alpha):
        rs = centroid.copy()
        rs += alpha * (centroid - worst)
        return rs

    @staticmethod
    def get_canonical_vector(i, n):
        rs = np.zeros((n, 1), int)
        rs[i - 1] = 1
        return rs

    @staticmethod
    def get_initial_simplex(simplex, centroid, n, canonical_vectors_length=1.0):
        for i in range(n):
            i += 1
            ei = NelderMeadHelper.get_canonical_vector(i, n) * canonical_vectors_length
            simplex.append(ei + centroid.copy())

        last_simplex_vertex = centroid.copy()
        last_simplex_vertex *= (n + 1)
        for v in simplex:
            last_simplex_vertex -= v

        simplex.append(last_simplex_vertex)

    @staticmethod
    def get_centroid_of_a_simplex(simplex):
        c = simplex[0].copy()
        for v in simplex:
            c += v
        c -= simplex[0]
        c /= len(simplex)
        return c
