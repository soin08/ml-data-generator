from dataclasses import dataclass
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List
import math
import numpy as np
import random

MAX_DIMENSION_SIZE = 10


@dataclass
class Properties:
    radius: float = 5
    num_samples: int = 1000   # number of points
    num_features: int = 2    # dimensions
    center: List[float] = None
    scale_coefs: List[float] = None


class Set:
    def __init__(self, properties: Properties):
        self.props = properties
        self.max_cords = [-1e6] * self.props.num_features
        self.min_cords = [1e6] * self.props.num_features
        self.points = np.ndarray(shape=(self.props.num_samples, self.props.num_features), dtype=float)
        if self.props.scale_coefs is None or not len(self.props.scale_coefs) == self.props.num_features:
            self.props.scale_coefs = [1] * self.props.num_features

        if self.props.center is None or not len(self.props.center) == self.props.num_features:
            self.props.center = [0] * self.props.num_features

    def create_points(self):
        for point_id in range(self.props.num_samples):
            vector = np.random.uniform(-1, 1, self.props.num_features)
            radius = np.random.uniform(0, self.props.radius)
            norm = math.sqrt(sum([el * el for el in vector]))
            self.points[point_id] = [cord / norm * radius + self.props.center[ix] for ix, cord in enumerate(vector)]
            self.points[point_id] *= self.props.scale_coefs
            self.max_cords = [max(self.max_cords[i], self.points[point_id][i]) for i in range(self.props.num_features)]
            self.min_cords = [min(self.min_cords[i], self.points[point_id][i]) for i in range(self.props.num_features)]


class World:
    def __init__(self, prop_list: List[Properties]):
        self.prop_list = prop_list
        self.sets = []
        self.max_cords = [-1e6] * MAX_DIMENSION_SIZE
        self.min_cords = [1e6] * MAX_DIMENSION_SIZE

    def create_sets(self):
        for prop in self.prop_list:
            figure = Set(prop)
            figure.create_points()
            self.sets.append(figure)
            for i in range(MAX_DIMENSION_SIZE):
                if i >= figure.props.num_features:
                    break
                self.max_cords[i] = max(self.max_cords[i], figure.max_cords[i])
                self.min_cords[i] = min(self.min_cords[i], figure.min_cords[i])

    def plot_2d(self, cord_x=0, cord_y=1):
        plt.figure(figsize=(5, 5))
        plt.axis([self.min_cords[cord_x], self.max_cords[cord_x], self.min_cords[cord_y], self.max_cords[cord_y]])

        for s in self.sets:
            x = s.points[:, cord_x] if s.points.shape[1] >= cord_x + 1 else np.zeros(s.points.shape[0])
            y = s.points[:, cord_y] if s.points.shape[1] >= cord_y + 1 else np.zeros(s.points.shape[0])
            plt.scatter(x, y)
        plt.show()

    def plot_3d(self):
        fig = plt.figure()
        ax = Axes3D(fig)
        for s in self.sets:
            x = s.points[:, 0]
            y = s.points[:, 1] if s.points.shape[1] >= 2 else np.zeros(s.points.shape[0])
            z = s.points[:, 2] if s.points.shape[1] >= 3 else np.zeros(s.points.shape[0])
            ax.scatter(x, y, z)
        plt.show()

    def save_csv(self, csvFile):
        if len(self.sets) == 0:
            return

        point_arrays = []
        for ix, s in enumerate(self.sets):
            label_coumn = np.ndarray(shape=(s.points.shape[0], 1))
            label_coumn.fill(ix)
            points = np.append(s.points, label_coumn, axis=1)
            point_arrays.append(points)

        merged_array = np.concatenate(point_arrays, axis=0)
        np.savetxt(csvFile, X=merged_array, fmt="%.4f", delimiter=",")


def generate_props(num_sets):
    num_samples_set = (500, 1000, 1500, 2000, 2500, 3000)
    num_features = random.randint(3, 10)
    props = []

    for _ in range(num_sets):
        num_samples = num_samples_set[random.randint(0, len(num_samples_set) - 1)]
        center = [random.randint(-10, 10) for _ in range(num_features)]
        scale_coefs = [random.randint(1, 3) for _ in range(num_features)]
        props.append(Properties(radius=random.randint(3, 10),
                                num_samples=num_samples,
                                num_features=num_features,
                                center=center,
                                scale_coefs=scale_coefs))
    return props


if __name__ == "__main__":
    props = generate_props(3)
    world = World(props)
    world.create_sets()
    world.save_csv("data.csv")
    world.plot_3d()

