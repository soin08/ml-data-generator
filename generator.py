from dataclasses import dataclass
import matplotlib.pyplot as plt
from typing import List
import math
import numpy as np


@dataclass
class Properties:
    radius: float = 5
    num_samples: int = 10   #number of points
    num_features: int = 2    #dimension
    intersects: List[int] = None
    distances: List[int] = None


class Object:
    def __init__(self, properties : Properties, scale_coefs : List = None):
        self.prop = properties
        self.points = np.ndarray(shape=(self.prop.num_samples, self.prop.num_features), dtype=float)
        self.scale_coefs = scale_coefs if scale_coefs and len(scale_coefs) == self.prop.num_features else [1] * self.prop.num_features
        self.create_points()

    def create_points(self):
        for point_id in range(self.prop.num_samples):
            vector = np.random.uniform(-1, 1, self.prop.num_features)
            radius = np.random.uniform(0, self.prop.radius)
            norm = math.sqrt(sum([el * el for el in vector]))
            self.points[point_id] = vector / norm * radius
            self.points[point_id] *= self.scale_coefs

    def plot(self, feature1: int, feature2: int):
        plt.figure(figsize=(5, 5))
        axis_val = self.prop.radius * max(self.scale_coefs)
        plt.axis([-axis_val, axis_val, -axis_val, axis_val])
        plt.plot(self.points[:, feature1], self.points[:, feature2], 'ro')
        plt.show()


if __name__ == "__main__":
    obj = Object(Properties(radius=5, num_samples=int(1e3), num_features=2), scale_coefs=[3, 4])
    obj.plot(0, 1)

