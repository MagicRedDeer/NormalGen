import numpy as np
import math
import os
import random

from scipy.ndimage import filters
import cv2

from . import operators


def load_image(filename):
    image = cv2.imread(filename)
    if image is None:
        raise IOError('The Image file cannot be opened')
    return image


def makeGray(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def normalizePerPixel(normalmap):
    norm = np.linalg.norm(normalmap, axis=2)
    normalmap[:, :, 0] /= norm
    normalmap[:, :, 1] /= norm
    normalmap[:, :, 2] /= norm
    return normalmap


def generateNormals(hmap, method='Sobel', strength=1, level=1):
    ''' Generates Normals given a normal map'''
    hmap = hmap / 255  # should we linearize
    op = operators.get_diffop(method)

    dy = filters.convolve(hmap, op, mode='wrap')
    dx = filters.convolve(hmap, op.transpose(), mode='wrap')
    dz = np.ones(dy.shape, dy.dtype)
    if math.fabs(strength < 0.01):
        strength = math.copysign(0.1, strength)
    dz_factor = 1.0 / strength * (1.0 + math.pow(2.0, level) / (2**8 - 1))
    #dz *= 1.0 / strength * (1.0 + math.pow(2.0, level));
    dz *= dz_factor

    normalmap = np.zeros((*hmap.shape, 3), dx.dtype)
    normalmap[:, :, 0] = dz
    normalmap[:, :, 1] = -dy
    normalmap[:, :, 2] = -dx

    norm = np.linalg.norm(normalmap, axis=2)
    normalmap[:, :, 0] /= norm
    normalmap[:, :, 1] /= norm
    normalmap[:, :, 2] /= norm

    normalmap = (normalmap + 1) / 2 * 255
    return normalmap.astype('uint8')


# look for speedups using cython or pyopencl
def generateAmbientOcclusion(hmap,
                             nmap,
                             size=20,
                             height_scaling=10,
                             scale=(1, 1),
                             intensity=1,
                             max_samples=24,
                             seed=0):
    hmap = hmap / 255 * height_scaling
    nmap = nmap / 255
    nmap = (nmap - 0.5) * 2

    width = size * 2 + 1
    sampler = np.zeros((width, width))
    sampler[size, size] = -1
    total_samples = width * width
    sample_ratio = min(1, max_samples / total_samples * math.pi / 2)

    ao = np.zeros(hmap.shape)
    distV = np.zeros(nmap.shape)

    random.seed(seed)
    num_samples = 0
    for y in range(-size, size + 1):
        for x in range(-size, size):
            if x == 0 and y == 0:
                continue
            if sample_ratio < 1 and random.random() < sample_ratio:
                continue

            if math.sqrt(x * x + y * y) <= size:
                new_sampler = sampler.copy()
                new_sampler[y + size, x + size] = 1

                disth = filters.convolve(hmap, new_sampler, mode='reflect')
                distV[:, :, 0] = disth
                distV[:, :, 1] = x * scale[0]
                distV[:, :, 2] = y * scale[1]
                distV = normalizePerPixel(distV)

                _ao = (distV * nmap).sum(axis=2) * intensity
                ao += np.clip(_ao, 0, 1)

                num_samples += 1

    print(num_samples)
    ao /= (num_samples)
    ao = (1 - ao)
    ao3 = np.zeros((*ao.shape, 3), dtype='uint8')
    ao = (ao * 255).astype('uint8')

    ao3[:, :, 0] = ao
    ao3[:, :, 1] = ao
    ao3[:, :, 2] = ao

    return ao3


def main(image_path=r'c:\Users\Quixel\Downloads\dice.jpg', show=True):
    image_dir, image_name = os.path.split(image_path)
    image_name, image_ext = os.path.splitext(image_name)

    im = cv2.imread(image_path)
    gray = makeGray(im)

    normal = generateNormals(gray)
    ao = generateAmbientOcclusion(gray, normal)

    print(ao.max(), ao.min())

    cv2.imwrite(
        os.path.join(image_dir, image_name + '_NMAP.' + image_ext), normal)
    cv2.imwrite(os.path.join(image_dir, image_name + '_AO.' + image_ext), ao)

    if show:
        cv2.imshow('Original Image', im)
        cv2.imshow('GrayScale Image', gray)
        cv2.imshow('Normal Map', normal)
        cv2.imshow('Occlusion Map', ao)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
