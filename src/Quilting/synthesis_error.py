from math import floor
from matplotlib.pyplot import axis
import numpy as np

def synthesis_error(img, out_slice, block_size, overlap_size, location):
    #height and width of the input image
    Height, Width = img.shape[:2]
    
    Matrix_error = np.zeros((Height - block_size, Width - block_size), dtype=float)

    #loop through the pixels in the input image
    for i in range(Height - block_size):
        for j in range(Width - block_size):
            row, col = i, j

            if location == "left":
                e_row = row + block_size
                e_col = col + overlap_size
            elif location == "up":
                e_row = row + overlap_size
                e_col = col + block_size
            elif location == "corner":
                e_row = row + overlap_size
                e_col = col + overlap_size

            #extract the block from the input image
            img_block = img[row : e_row, col : e_col, :]

            #calculate the squared error between the output slice and the input block
            diff = out_slice - img_block
            err = np.reshape(diff, (diff.shape[0] * diff.shape[1] * diff.shape[2], 1))
            err = np.multiply(err, err)
            ssum = np.sum(err, axis=0) #ssum = sum of squared errors

            #update the error matrix
            Matrix_error[i, j] += ssum

    return Matrix_error
