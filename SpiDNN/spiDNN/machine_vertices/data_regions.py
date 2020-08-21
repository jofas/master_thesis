from enum import Enum


class DataRegions(Enum):
    SYSTEM = 0
    BASE_PARAMS = 1
    KEYS = 2
    WEIGHTS = 3
    SOFTMAX_PARAMS = 4
    TRAINABLE_PARAMS = 5
    NEXT_LAYER_WEIGHTS = 6
