import numpy as np
import matplotlib.pyplot as plt
import cv2

from sift import SIFT

def show_initial_image(sift: SIFT, image):
    image = sift.create_initial_image(image)
    plt.imshow(image.astype('uint8'))
    plt.gray()
    plt.show()
    
def show_gaussian_pyramid(sift: SIFT, image):
    image = sift.create_initial_image(image)
    n_octaves = sift.compute_n_octaves(image.shape)
    images = sift.build_gaussian_pyramid(image, n_octaves)
    shape = images.shape
    
    fig, ax = plt.subplots(shape[0], shape[1], figsize=(2 * shape[0], 2 * shape[1]))
    plt.gray()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    for i in range(shape[0]):
        for j in range(shape[1]):
            ax[i][j].imshow(images[i][j].astype('uint8'))
            ax[i][j].axis("off")
    plt.show()
    
def show_dog_pyramid(sift: SIFT, image):
    image = sift.create_initial_image(image)
    n_octaves = sift.compute_n_octaves(image.shape)
    images = sift.build_gaussian_pyramid(image, n_octaves)
    dogs = sift.build_dog_pyramid(images)
    shape = dogs.shape
    
    fig, ax = plt.subplots(shape[0], shape[1], figsize=(2 * shape[0], 2 * shape[1]))
    plt.gray()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    for i in range(shape[0]):
        for j in range(shape[1]):
            ax[i][j].imshow(dogs[i][j].astype('uint8'))
            ax[i][j].axis("off")
    plt.show()

if __name__ == '__main__':
    sift = SIFT()
    image = cv2.imread('./images/dict1.jpeg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (750, 1000))
    
    # show_initial_image(sift, image)
    # show_gaussian_pyramid(sift, image)
    show_dog_pyramid(sift, image)