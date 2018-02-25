from dataclasses import dataclass
import matplotlib.pyplot as plt
from typing import List
import math
import numpy as np


@dataclass
class Properties:
    radius: int = 5
    num_samples: int = 1000   # number of points
    num_features: int = 2    # dimensions
    center: np.ndarray = np.zeros(num_features)


class Set:
    def __init__(self, properties: Properties):
        self.props = properties
        self.points = np.ndarray(shape=(self.props.num_samples, self.props.num_features), dtype=float)

    def create_points(self):
        for point_id in range(self.props.num_samples):
            vector = np.random.uniform(-1, 1, self.props.num_features)
            radius = np.random.uniform(0, self.props.radius)
            norm = math.sqrt(sum([el * el for el in vector]))
            # print(type(vector[0]))
            # print(self.props.center)
            # print(vector)
            self.points[point_id] = [cord / norm * radius + self.props.center[ix] for ix, cord in enumerate(vector)]


class World:
    def __init__(self, prop_list: List[Properties]):
        self.prop_list = prop_list
        self.sets = []

    def create_sets(self):
        for prop in self.prop_list:
            set = Set(prop)
            set.create_points()
            self.sets.append(set)

    def plot_2d(self):
        plt.figure(figsize=(10, 10))
        for set in self.sets:
            plt.scatter(set.points[:, 0], set.points[:, 1])
        # plt.plot(self.points[:, feature1], self.points[:, feature2], 'ro')
        plt.show()


if __name__ == "__main__":
    props = [Properties(radius=10, num_samples=int(1e3), num_features=3, center=np.array([5, 5, 5])),
             Properties(radius=3, num_samples=int(1e2), num_features=3, center=np.array([16, 16, 6]))]

    world = World(props)
    world.create_sets()
    world.plot_2d()
    # obj.plot(0, 2)


