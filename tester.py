import itertools
from math import floor, ceil
import numpy as np
as_strided = np.lib.stride_tricks.as_strided

from utils.utils import Batcher, MNIST
from activation import Relu
