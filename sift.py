import numpy as np
import cv2
from typing import Tuple

class SIFT:
    sigma: float
    # asssumed gaussian blur for input image
    init_sigma: float
    n_octave_layers: int
    contrast_threshold: float

    def __init__(self):
        self.sigma = 1.6
        self.init_sigma = 0.5
        self.n_octave_layers = 3
        self.contrast_threshold = 0.04
    

    def create_initial_image(self, image: np.ndarray) -> np.ndarray:
        image = image.astype('float32') # float64 -> float32
        image = cv2.resize(image, dsize=None, fx=2.0, fy=2.0)
        
        sig_diff = np.sqrt(max((self.sigma ** 2 - (2 * self.init_sigma) ** 2), 0.01))
        return cv2.GaussianBlur(image, None, sigmaX=sig_diff, sigmaY=sig_diff) # init_sigma -> sigma
        

    def compute_n_octaves(self, image_shape: Tuple[int, int]) -> int:
        a = min(image_shape)
        b = np.log(a)
        c = b / np.log(2.0)
        d = np.round(c - 1.0)
        return int(np.round(np.log(min(image_shape)) / np.log(2.0) - 1.0))
        

    def build_gaussian_pyramid(self, image: np.ndarray, n_octaves: int) -> np.ndarray:
        n_images_per_octave = self.n_octave_layers + 3
        # compute Gaussian sigmas
        sigs = np.zeros(n_images_per_octave)
        sigs[0] = self.sigma
        k = 2.0 ** (1 / self.n_octave_layers)

        for i in range(1, n_images_per_octave):
            sig_prev = k ** (i-1) * self.sigma
            sig_total = sig_prev * k
            sigs[i] = np.sqrt(sig_total ** 2 - sig_prev ** 2)
            
        images = []
        for octave_idx in range(n_octaves):
            images_in_octave = []
            for sig in sigs[1:]:
                image = cv2.GaussianBlur(image, None, sigmaX=sig, sigmaY=sig)
                images_in_octave.append(image)
            images.append(images_in_octave)
            # self.n_octave_layers == n_images_per_octave - 3 
            base = images_in_octave[-3]
            image = cv2.resize(base, None, fx=0.5, fy=0.5)

        return np.array(images)
    
    def build_dog_pyramid(self, images: np.ndarray) -> np.ndarray:
        dog_images = []
        for images_in_octave in images:
            dog_images_in_octave = []
            for first_image, second_image in zip(images_in_octave, images_in_octave[1:]):
                dog_images_in_octave.append(cv2.subtract(second_image, first_image))
            dog_images.append(dog_images_in_octave)

        return np.array(dog_images)