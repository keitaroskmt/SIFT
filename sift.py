import numpy as np
import cv2
from typing import Tuple, Optional

class SIFT:
    sigma: float
    # asssumed gaussian blur for input image
    init_sigma: float
    n_octave_layers: int
    contrast_threshold: float
    # width of border in which to ignore keypoints
    img_border: int 
    # maximum steps of keypoint interpolation before failure
    max_interp_steps: int

    def __init__(self):
        self.sigma = 1.6
        self.init_sigma = 0.5
        self.n_octave_layers = 3
        self.contrast_threshold = 0.04
        self.img_border = 5
        self.max_interp_steps = 5
    

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
        
    
    def find_scale_space_extrema(self, images: np.ndarray, dog_images: np.ndarray):
        keypoints = []
        
        for octave_idx, dog_images_in_octave in enumerate(dog_images):
            for image_idx, (prev_image, cur_image, next_image) in enumerate(zip(dog_images_in_octave, dog_images_in_octave[1:], dog_images_in_octave[2:])):
                for row in range(self.img_border, cur_image.shape[0] - self.img_border):
                    for col in range(self.img_border, cur_image.shape[1] - self.img_border):
                        if self.is_extreme(prev_image[row-1:row+2, col-1:col+2], cur_image[row-1:row+2, col-1:col+2], next_image[row-1:row+2, col-1:col+2]):
                            pass


    def is_extreme(self, prev_region: np.ndarray, cur_region: np.ndarray, next_region: np.ndarray) -> bool:
        threshold = np.floor(0.5 * self.contrast_threshold / self.n_octave_layers * 255)
        # TODO
        # threshold = np.floor(0.5 * self.contrast_threshold / self.n_octave_layers * 255 * 48)
        
        val = cur_region[1, 1]
        if abs(val) > threshold:
            if val > 0:
                return np.all(val >= prev_region) and np.all(val >= cur_region) and np.all(val >= next_region)
            elif val < 0:
                return np.all(val <= prev_region) and np.all(val <= cur_region) and np.all(val <= next_region)
        return False
        
        
    def adjust_local_extrema(self, dog_images_in_ocatave: np.ndarray, row: int, col: int, image_idx: int, octave_idx: int, sigma: float) -> Optional[cv2.KeyPoint]:
        image_shape = dog_images_in_ocatave[0].shape

        for i in range(self.max_interp_steps):
            prev_image, cur_image, next_image = dog_images_in_ocatave[image_idx-1:image_idx+2]
            pixel_cube = np.array([prev_image, cur_image, next_image])
            # TODO unit8, or float32?
            
            # calculate approximate gradient
            dx = 0.5 * (pixel_cube[1, 1, 2] - pixel_cube[1, 1, 0])
            dy = 0.5 * (pixel_cube[1, 2, 1] - pixel_cube[1, 0, 1])
            ds = 0.5 * (pixel_cube[2, 1, 1] - pixel_cube[0, 1, 1])
            
            dD = np.array([dx, dy, ds])
            
            # calculate approximate hessian
            dxx = pixel_cube[1, 1, 2] - 2 * pixel_cube[1, 1, 1] + pixel_cube[1, 1, 0]
            dyy = pixel_cube[1, 2, 1] - 2 * pixel_cube[1, 1, 1] + pixel_cube[1, 0, 1]
            dss = pixel_cube[2, 1, 1] - 2 * pixel_cube[1, 1, 1] + pixel_cube[0, 1, 1]
            dxy = 0.25 * (pixel_cube[1, 2, 2] - pixel_cube[1, 2, 0] - pixel_cube[1, 0, 2] + pixel_cube[1, 0, 0])
            dxs = 0.25 * (pixel_cube[2, 1, 2] - pixel_cube[0, 1, 2] - pixel_cube[2, 1, 0] + pixel_cube[0, 1, 0])
            dys = 0.25 * (pixel_cube[2, 2, 1] - pixel_cube[2, 0, 1] - pixel_cube[0, 2, 1] + pixel_cube[0, 0, 1])
            
            H = np.array([[dxx, dxy, dxs],
                          [dxy, dyy, dys],
                          [dxs, dys, dss]])
                        
            # x = -H^{-1} dD
            # use lstsq in case of a singular matrix
            x = -np.linalg.lstsq(H, dD, rcond=None)[0]
            
            if (abs(x[0]) < 0.5 and abs(x[1]) < 0.5 and abs(x[2]) < 0.5):
                break
            
            row += int(np.round(x[0]))
            col += int(np.round(x[1]))
            image_idx += int(np.round(x[2]))
            
            if image_idx < 1 or image_idx > self.n_octave_layers or row < self.img_border or row >= image_shape[0] - self.img_border or col < self.img_border or col >= image_shape[1] - self.img_border:
                return None
            
        if i >= self.max_interp_steps:
            return None
            
        contr = pixel_cube[1, 1, 1] + 0.5 * dD.dot(x)
        
        # G.Lowe(2004) p.11
        contrast_threshold = 0.04
        if abs(contr) * self.n_octave_layers < contrast_threshold:
            return None
            
        H_xy = H[:2, :2]
        tr = np.trace(H_xy)
        det = np.linalg.det(H_xy)
        
        # ratio of eigenvalues
        edge_threshold = 10
        # G.Lowe(2004) p.12
        if det <= 0 or (tr ** 2) * edge_threshold >= ((edge_threshold + 1) ** 2) * det:
            return None
            
        keypoint = cv2.KeyPoint()
        keypoint.pt = (row + x[0]) * (2 ** octave_idx), (col + x[1]) * (2 ** octave_idx)
        keypoint.octave = octave_idx + (image_idx << 8) + ((np.round(x[2] + 0.5) * 255) << 16)
        keypoint.size = sigma * (2 ** ((image_idx + x[2]) / self.n_octave_layers)) * (2 ** (octave_idx + 1))
        keypoint.response = abs(contr)
        
        return keypoint
        
    def calc_keypoint_with_orientations(self, keypoint: cv2.Keypoint, image: np.ndarray, octave_idx: int, sigma: float):
        scl_octv = keypoint.size * 0.5 / (2 ** octave_idx)
        radius = int(np.round(4.5 * scl_octv))
        expf_scale = -1.0 / (2.0 * (sigma ** 2))
        num_bins = 36
        raw_hist = np.zeros(num_bins)
        smooth_hist = np.zeros(num_bins)
    
        for i in range(-radius, radius + 1):
            x = int(keypoint.pt.x / (2 ** octave_idx)) + i
            if x <= 0 or x >= image.shape[0] - 1:
                continue

            for j in range(-radius, radius + 1):
                y = int(keypoint.pt.y / (2 ** octave_idx)) + j
                if y <= 0 or y >= image.shape[1] - 1:
                    continue
                
                dx = image[x, y+1] - image[x, y-1]
                dy = image[x-1, y] - image[x+1, y]
                
                mag = np.sqrt(dx ** 2 + dy ** 2)
                ori = np.rad2deg(np.arctan2(dy, dx))
                weight = np.exp(expf_scale * (i ** 2 + j ** 2))
                hist_idx = int(np.round(ori / 360.0 * num_bins)) % num_bins
                raw_hist[hist_idx] += weight * mag
                
        for i in range(num_bins):
            smooth_hist[i] = raw_hist[i] * (6.0 / 16.0) + \
                (raw_hist[(i+1) % num_bins] + raw_hist[(i-1) % num_bins]) * (4.0 / 16.0) + \
                (raw_hist[(i+2) % num_bins] + raw_hist[(i-2) % num_bins]) * (1.0 / 16.0)
            
        max_val = np.max(smooth_hist)
        
        keypoints = []
        mag_threshold = max_val * 0.8
        for i in range(num_bins):
            left = i-1 if i > 0 else num_bins-1
            right = i+1 if i < num_bins-1 else 0
            val = smooth_hist[i]
            left_val = smooth_hist[left]
            right_val = smooth_hist[right]
            
            if val > left_val and val > right_val and val > mag_threshold:
                # quadratic interpolation
                interpolated_idx = (i + 0.5 * (left_val - right_val) / (left_val - 2 * val + right_val)) % num_bins
                angle = 360.0 - (360.0 / num_bins * interpolated_idx)
                if abs(angle - 360.0) < 1e-7:
                    angle = 0.0
                keypoints.append(cv2.KeyPoint(*keypoint.pt, keypoint.size, angle, keypoint.response, keypoint.octave))
                
        return keypoints

    



                
                
