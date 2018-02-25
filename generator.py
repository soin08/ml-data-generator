from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class SetProperties:
    radius: int = 5
    num_samples: int = 100
    num_features: int = 2
    intersects: List[int] = None
    distances: List[int] = None


if __name__ == "__main__":
    circle = SetProperties(num_samples=1234, distances=[1,2,3])
    print (circle.distances)
