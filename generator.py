from dataclasses import dataclass
from typing import List
import math
import numpy as np


@dataclass
class Properties:
    radius: int = 5
    num_samples: int = 100   #number of points
    num_features: int = 2    #dimension
    intersects: List[int] = None
    distances: List[int] = None


class Object:
    def __init__(self, properties : Properties):
        self.prop = properties
        self.points = np.ndarray(shape=(self.prop.num_samples, self.prop.num_features), dtype=float)
        self.create_points()

    def create_points(self):
       for point_id in range(self.prop.num_samples):
           vector = np.random.uniform(-1, 1, self.prop.num_features)
           radius = np.random.uniform(0, self.prop.radius)
           norm = math.sqrt(sum([el * el for el in vector]))
           self.points[point_id] = [cord / norm * radius for cord in vector]


if __name__ == "__main__":
    obj = Object(Properties())
