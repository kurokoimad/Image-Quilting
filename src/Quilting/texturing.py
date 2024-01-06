from math import ceil
from random import randint
import matplotlib.pyplot as plt
import numpy as np
from Quilting.border_cutting import * 
from Quilting.dpMinCut import *  

class textureMain():
    def __init__(self, img, block_size, overlap_size, num_blocks, H, W, H1, W1, tolerance_factor):
        self.img = img
        self.block_size = block_size
        self.overlap_size = overlap_size
        self.num_blocks = num_blocks
        self.H = H
        self.W = W
        self.H1 = H1
        self.W1 = W1
        self.tolerance_factor = tolerance_factor

    def generateOutputMask(self):
        # Initialize an output image mask with zeros
        outimg = np.zeros((self.H1, self.W1, 3))
        return outimg

    def overlaps(self, i, j):
        l = r = t = b = 0
        if (i == 0 and j == 0):
            b = r = self.block_size
        elif (i == 0 and j != 0):
            b = self.block_size
            l = (j-1)*self.block_size - (j-1)*self.overlap_size
            r = l + self.block_size
        elif (j == 0 and i != 0):
            r = self.block_size
            t = (i-1)*self.block_size - (i-1)*self.overlap_size
            b = t + self.block_size
        else:
            l = (j-1)*self.block_size - (j-1)*self.overlap_size
            r = l + self.block_size
            t = (i-1)*self.block_size - (i-1)*self.overlap_size
            b = t + self.block_size

        return [l, r, t, b]

    def createTexture(self):
        outimg = self.generateOutputMask()  # Initialize output image mask
        img_np = self.img

        for i in range(0, self.num_blocks):
            for j in range(0, self.num_blocks):
                # Calculate overlap regions
                ov_left, ov_right, ov_top, ov_bottom = self.overlaps(i, j)

                # Initialize an error matrix for the synthesis process
                E = np.zeros(
                    (self.H - self.block_size, self.W - self.block_size))

                if (i == 0 and j == 0):
                    # If it's the first block, randomly select a region from the input image
                    row_id = randint(0, self.H - self.block_size)
                    col_id = randint(0, self.W - self.block_size)
                    temp = img_np[row_id:row_id+self.block_size,
                                  col_id:col_id+self.block_size, :]
                    outimg[ov_top:ov_bottom, ov_left:ov_right, :] = temp
                    continue

                elif (i == 0):
                    # If it's in the first row, synthesize based on the left overlap
                    sliced_cut = outimg[ov_top:ov_bottom,
                                        ov_left:ov_left + self.overlap_size, :]
                    E = synthesis_error(
                        img_np, sliced_cut, self.block_size, self.overlap_size, "left")

                elif (j == 0):
                    # If it's in the first column, synthesize based on the top overlap
                    sliced_cut = outimg[ov_top:ov_top +
                                        self.overlap_size, ov_left:ov_right, :]
                    E = synthesis_error(
                        img_np, sliced_cut, self.block_size, self.overlap_size, "up")

                else:
                    # If it's in the middle, synthesize based on both left and top overlaps
                    sliced_cut = outimg[ov_top:ov_bottom,
                                        ov_left:ov_left + self.overlap_size, :]
                    E = synthesis_error(
                        img_np, sliced_cut, self.block_size, self.overlap_size, "left")

                    sliced_cut = outimg[ov_top:ov_top +
                                        self.overlap_size, ov_left:ov_right, :]
                    E = E + synthesis_error(
                        img_np, sliced_cut, self.block_size, self.overlap_size, "up")

                    sliced_cut = outimg[ov_top:ov_top +
                                        self.overlap_size, ov_left:ov_left + self.overlap_size, :]
                    E = E - synthesis_error(
                        img_np, sliced_cut, self.block_size, self.overlap_size, "corner")

                # Find the minimum error and its location
                mn_E = np.min(E)
                E_col = np.reshape(E, (E.shape[0]*E.shape[1], 1))
                matches = np.asarray(
                    np.where(E_col <= (1+self.tolerance_factor)*mn_E))
                matches = np.reshape(
                    matches, (matches.shape[0]*matches.shape[1], 1))
                im_r, im_c = np.unravel_index(matches, E.shape)
                rr = int(im_r[randint(0, im_r.shape[1]-1)][0])
                cc = int(im_c[randint(0, im_c.shape[1]-1)][0])

                # Initialize a boundary matrix for minCut
                boundary = np.ones((self.block_size, self.block_size))

                # Apply minCut horizontally if not in the first row
                if(i != 0):
                    img_overlap = img_np[rr:rr +
                                         self.overlap_size, cc:cc+self.block_size, :]
                    out_overlap = outimg[ov_top:ov_top +
                                         self.overlap_size, ov_left:ov_right, :]
                    cut = minCut(img_overlap, out_overlap, "horizontal")
                    boundary[0:self.overlap_size,
                             0:self.block_size] = np.double(cut >= 0)

                # Apply minCut vertically if not in the first column
                if(j != 0):
                    img_overlap = img_np[rr:rr +
                                         self.block_size, cc:cc+self.overlap_size, :]
                    out_overlap = outimg[ov_top:ov_bottom,
                                         ov_left:ov_left + self.overlap_size, :]
                    cut = minCut(img_overlap, out_overlap, "vertical")
                    boundary[0:self.block_size, 0:self.overlap_size] = np.multiply(
                        boundary[0:self.block_size, 0:self.overlap_size], np.double(cut >= 0))

                # Repeat the boundary matrix along the third axis to match the image channels
                boundary = np.repeat(boundary[:, :, np.newaxis], 3, axis=2)

                # Apply border cutting based on the boundary matrix
                border = np.multiply(
                    outimg[ov_top:ov_bottom, ov_left:ov_right, :], (boundary == 0))
                right_part = np.multiply(
                    img_np[rr:rr+self.block_size, cc:cc+self.block_size, :], (boundary == 1))
                outimg[ov_top:ov_bottom, ov_left:ov_right,
                       :] = border + right_part

            print("Progress=", i)
        return outimg
