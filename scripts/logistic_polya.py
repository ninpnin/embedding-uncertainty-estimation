import array
import numpy as np
from polyagamma import random_polyagamma


# Get a 5 by 1 array of PG(1, 2) variates.
o = random_polyagamma(z=2, size=5)