import numpy as np

central_diff = np.array(
        [[-1],
         [ 0],
         [ 1]])

sobel = np.array(
        [[-2, -1, -2],
         [ 0,  0,  0],
         [ 2,  1,  2]])

scharr = np.array(
        [[-3, -10, -3],
         [ 0,   0,  0],
         [ 3,  10,  3]])

prewitt = np.array(
        [[-1, -1, -1],
         [ 0,  0,  0],
         [ 1,  1,  1]])

diffops = {
        'Sobel'         : sobel,
        'Central Diff'  : central_diff,
        'Shcarr'        : scharr,
        'Prewitt'       : prewitt
}

def get_diffop(name, default=sobel):
    return diffops.get(name, default)

