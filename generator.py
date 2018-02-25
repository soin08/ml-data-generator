from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class SetProperties:
    num_classes: int
    num_samples: int
    num_features: int
    intersects: List[int]
    linearly_separable: List[int]
    distances: List[int]
