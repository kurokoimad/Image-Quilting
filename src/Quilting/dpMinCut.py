import numpy as np
from sys import maxsize

#Dynamic Programming refines the boundaries between two overlapping image blocks

def minCut(img_overlap, out_overlap, location):
    #calculate the squared differences between corresponding pixel values
    diff = img_overlap - out_overlap

    #summing squared differences in the energy matrix
    Matrix = np.sum(np.multiply(diff, diff), axis=2)

    if location == "horizontal":
        Matrix = np.transpose(Matrix)

    Row, Col = Matrix.shape[:2]

    #initializing both cut and dp matrix with zeros
    cut = np.ones((Row, Col))
    DP = np.zeros((Row, Col))

    #initializing the first row of the dp matrix
    for i in range(Col):
        DP[0, i] = Matrix[0, i]

    for i in range(1, Row):
        for j in range(Col):
            paths = [DP[i - 1, j]]
            if j != 0:
                paths.append(DP[i - 1, j - 1])
            if j != Col - 1:
                paths.append(DP[i - 1, j + 1])
            DP[i, j] = Matrix[i, j] + min(paths) #update the dp matrix

    #find the index of the minimum energy in the last row
    min_val = min_idx = maxsize
    for i in range(Col):
        if min(min_idx, Matrix[Row - 1, i]) < min_val:
            min_idx = i

    #find the optimal cut path
    cut[Row-1, min_idx] = 0
    cut[Row-1, min_idx + 1:Col] = 1
    cut[Row-1, 0 : min_idx] = -1

    for i in range(Row - 2, -1, -1):
        for j in range(Col):
            if min_idx < Col - 1:
                if Matrix[i, min_idx + 1] == min(Matrix[i, max(0, min_idx - 1) : min_idx + 2]):
                    min_idx = min_idx + 1
            if min_idx > 0:
                if Matrix[i, min_idx - 1] == min(Matrix[i, min_idx - 1 : min(Col - 1, min_idx + 2)]) : min_idx = min_idx - 1
            cut[i, min_idx] = 0
            cut[i, min_idx + 1 : Col] = 1
            cut[i, 0 : min_idx] = -1

    #transpose
    if location == "horizontal":
        cut = np.transpose(cut)

    return cut
