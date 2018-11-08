import numpy as np
import math
import os

from scipy.ndimage import filters
import cv2


import operators


def makeGray(image):
    print ( image.shape )
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def generateNormals(hmap, method='sobel', strength=2.5, level=7):
    ''' Generates Normals given a normal map'''
    hmap = hmap / 255 # should we linearize
    op = operators.get_diffop(method)

    dy = filters.convolve(hmap, op, mode='wrap')
    dx = filters.convolve(hmap, op.transpose(), mode='wrap')
    dz = np.ones(dy.shape, dy.dtype)
    if math.fabs(strength < 0.01):
        strength = math.copysign(0.1, strength)
    dz_factor = 1.0 / strength * (1.0 + math.pow(2.0, level)/(2**8-1))
    #dz *= 1.0 / strength * (1.0 + math.pow(2.0, level));
    dz *= dz_factor

    normalmap = np.zeros((*hmap.shape, 3), dx.dtype)
    normalmap[:,:,0] = dz
    normalmap[:,:,1] = dy
    normalmap[:,:,2] = dx

    norm = np.linalg.norm(normalmap, axis=2)
    normalmap[:,:,0] /= norm
    normalmap[:,:,1] /= norm
    normalmap[:,:,2] /= norm

    print(normalmap.max(), normalmap.min())

    normalmap = (normalmap + 1) / 2 * 255
    return normalmap.astype('uint8')


def main(image_path=r'c:\Users\Quixel\Downloads\dice.jpg', show=True):
    image_dir, image_name = os.path.split(image_path)
    image_name, image_ext = os.path.splitext(image_name)

    im = cv2.imread(image_path)
    gray = makeGray(im)

    normal = generateNormals(gray)

    cv2.imwrite(
            os.path.join(image_dir, image_name + '_NMAP.' + image_ext),
            normal)

    if show:
        cv2.imshow('Original Image', im)
        cv2.imshow('GrayScale Image', gray)
        cv2.imshow('Normal Map', normal)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
