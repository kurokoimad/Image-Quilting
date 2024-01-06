from pathlib import Path
from math import floor
import cv2 as cv
from numpy.core.fromnumeric import shape
from Quilting.texturing import *
from tkinter import Tk, filedialog

CURRENT_FOLDER = Path(__file__).parent

#YOU CAN USE THIS MAIN METHOD CREATE A TEXTURE FROM THE GIVEN 
#SAMPLES IN THE 'src/Textures' FILE 

if __name__ == '__main__':

    # Reading image file
    img = cv.imread('src/Textures/texture4.jpeg')

    # define the constants
    block_size = 50
    overlap_size = block_size//6
    num_blocks = 10
    tolerance_factor = 0.1

    # convert image to double
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)/255.0
    H, W = img.shape[:2]
    H1 = W1 = num_blocks * (block_size - overlap_size)

    # generate output mask
    texFunc = textureMain(img, block_size, overlap_size, num_blocks, H, W, H1, W1, tolerance_factor)
    texFunc.generateOutputMask()
    output_img = texFunc.createTexture()
    proper_h1 = proper_w1 = (block_size-overlap_size)*num_blocks - (overlap_size*(num_blocks-1))
    plt.imshow(output_img[0:proper_h1, 0:proper_w1, :])
    plt.axis('off')
    plt.savefig('src/Result/texture4', bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()