from enum import Enum


class DataRegions(Enum):
    """
    Enumeration for data regions in SDRAM (the SpiNNaker side
    equivalent is implemented in SpiDNN/src/spiDNN.h.
    It is a human readable way to discern what SDRAM data region
    stores what.
    """
    SYSTEM = 0
    BASE_PARAMS = 1
    KEYS = 2
    WEIGHTS = 3
    SOFTMAX_PARAMS = 4
    TRAINABLE_PARAMS = 5
    NEXT_LAYER_WEIGHTS = 6
