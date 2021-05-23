import numpy as np
import cv2
from typing import List, Tuple, Optional

class SIFT:
    sigma: float
    init_sigma: float # asssumed gaussian blur for input image
    n_octave_layers: int
    contrast_threshold: float
    img_border: int  # width of border in which to ignore keypoints
    max_interp_steps: int # maximum steps of keypoint interpolation before failure
    ori_sig_fctr: float # determines gaussian sigma for orientation assignment
    descr_scl_fctr: float # determines the size of a single descriptor orientation histogram
    descr_mag_thr: float # threshold on magnitude of elements of descriptor vector
    flt_epsilon: float

    def __init__(self):
        self.sigma = 1.6
        self.init_sigma = 0.5
        self.n_octave_layers = 3
        self.contrast_threshold = 0.04
        self.img_border = 5
        self.max_interp_steps = 5
        self.ori_sig_fctr = 1.5
        self.descr_scl_fctr = 3.0
        self.descr_mag_thr = 0.2
        self.flt_epsilon = 1e-7
        

    def detect_and_compute(self, image: np.ndarray):
        """
        main function
        """
        init_image = self.create_initial_image(image)
        n_octaves = self.compute_n_octaves(init_image.shape)
        gaussian_pyramid = self.build_gaussian_pyramid(init_image, n_octaves)
        dog_pyramid = self.build_dog_pyramid(gaussian_pyramid)
        keypoints = self.find_scale_space_extrema(gaussian_pyramid, dog_pyramid)
        keypoints = self.remove_duplicated_sorted(keypoints)
        keypoints = self.convert_keypoints_size(keypoints) 
        descriptors = self.calc_sift_descriptors(keypoints, gaussian_pyramid)
        return keypoints, descriptors
    

    def create_initial_image(self, image: np.ndarray) -> np.ndarray:
        """
        create the initial image from input by upsampling by 2 and blur it
        """
        image = image.astype('float32')
        image = cv2.resize(image, dsize=None, fx=2.0, fy=2.0)
        
        sig_diff = np.sqrt(max((self.sigma ** 2 - (2 * self.init_sigma) ** 2), 0.01))
        return cv2.GaussianBlur(image, None, sigmaX=sig_diff, sigmaY=sig_diff) # init_sigma -> sigma
        

    def compute_n_octaves(self, image_shape: Tuple[int, int]) -> int:
        """
        calculate number of octaves
        """
        return int(np.round(np.log(min(image_shape)) / np.log(2.0) - 1.0))
        

    def build_gaussian_pyramid(self, image: np.ndarray, n_octaves: int) -> np.ndarray:
        """
        build gaussian pyramid
        """
        n_images_per_octave = self.n_octave_layers + 3
        # compute Gaussian sigmas
        sigs = np.zeros(n_images_per_octave)
        sigs[0] = self.sigma
        k = 2.0 ** (1 / self.n_octave_layers)

        for i in range(1, n_images_per_octave):
            sig_prev = (k ** (i-1)) * self.sigma
            sig_total = sig_prev * k
            sigs[i] = np.sqrt(sig_total ** 2 - sig_prev ** 2)
            
        images = []
        for _ in range(n_octaves):
            images_in_octave = []
            images_in_octave.append(image)
            for sig in sigs[1:]:
                image = cv2.GaussianBlur(image, None, sigmaX=sig, sigmaY=sig)
                images_in_octave.append(image)
            images.append(images_in_octave)
            # self.n_octave_layers == n_images_per_octave - 3 
            base = images_in_octave[-3]
            image = cv2.resize(base, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)

        return np.array(images, dtype=object)
    

    def build_dog_pyramid(self, images: np.ndarray) -> np.ndarray:
        """
        build difference of gaussian (DOG) pyramid 
        """
        dog_images = []
        for images_in_octave in images:
            dog_images_in_octave = []
            for first_image, second_image in zip(images_in_octave, images_in_octave[1:]):
                dog_images_in_octave.append(cv2.subtract(second_image, first_image))
            dog_images.append(dog_images_in_octave)

        return np.array(dog_images, dtype=object)
        
    
    def find_scale_space_extrema(self, images: np.ndarray, dog_images: np.ndarray) -> List[cv2.KeyPoint]:
        """
        find scale-space extrema in the image pyramid
        """
        keypoints = []
        
        for octave_idx, dog_images_in_octave in enumerate(dog_images):
            for image_idx, (prev_image, cur_image, next_image) in enumerate(zip(dog_images_in_octave, dog_images_in_octave[1:], dog_images_in_octave[2:])):
                for row in range(self.img_border, cur_image.shape[0] - self.img_border):
                    for col in range(self.img_border, cur_image.shape[1] - self.img_border):
                        if self.is_extreme(prev_image[row-1:row+2, col-1:col+2], cur_image[row-1:row+2, col-1:col+2], next_image[row-1:row+2, col-1:col+2]):
                            res = self.adjust_local_extrema(dog_images_in_octave, row, col, image_idx + 1, octave_idx, self.sigma)
                            
                            if res:
                                keypoint, new_image_idx = res
                                keypoint_with_orientations = self.calc_keypoint_with_orientations(keypoint, images[octave_idx][new_image_idx], octave_idx)
                                for keypoint_with_orientation in keypoint_with_orientations:
                                    keypoints.append(keypoint_with_orientation)
        return keypoints


    def is_extreme(self, prev_region: np.ndarray, cur_region: np.ndarray, next_region: np.ndarray) -> bool:
        """
        return True if the center element is greater than or less than all its neighbors, False otherwise
        """
        threshold = np.floor(0.5 * self.contrast_threshold / self.n_octave_layers * 255)
        
        val = cur_region[1, 1]
        if abs(val) > threshold:
            if val > 0:
                return np.all(val >= prev_region) and np.all(val >= cur_region) and np.all(val >= next_region)
            elif val < 0:
                return np.all(val <= prev_region) and np.all(val <= cur_region) and np.all(val <= next_region)
        return False
        
        
    def adjust_local_extrema(self, dog_images_in_octave: np.ndarray, row: int, col: int, image_idx: int, octave_idx: int, sigma: float) -> Optional[cv2.KeyPoint]:
        """
        adjust pixel positions of scale-space extrema
        """
        image_shape = dog_images_in_octave[0].shape

        for i in range(self.max_interp_steps):
            prev_image, cur_image, next_image = dog_images_in_octave[image_idx-1:image_idx+2]
            # [0, 255] -> [0, 1]
            pixel_cube = np.array([prev_image[row-1:row+2, col-1:col+2], cur_image[row-1:row+2, col-1:col+2], next_image[row-1:row+2, col-1:col+2]]).astype('float32') / 255.0
            
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
            
            col += int(np.round(x[0]))
            row += int(np.round(x[1]))
            image_idx += int(np.round(x[2]))
            
            if image_idx < 1 or image_idx > self.n_octave_layers or row < self.img_border or row >= image_shape[0] - self.img_border or col < self.img_border or col >= image_shape[1] - self.img_border:
                return None
            
        if i >= self.max_interp_steps - 1:
            return None
            
        contr = pixel_cube[1, 1, 1] + 0.5 * dD.dot(x)
        
        # G.Lowe(2004) p.11
        if abs(contr) * self.n_octave_layers < self.contrast_threshold:
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
        keypoint.pt = (col + x[0]) * (2 ** octave_idx), (row + x[1]) * (2 ** octave_idx)
        keypoint.octave = octave_idx + (image_idx << 8) + (int(np.round((x[2] + 0.5) * 255)) * (1 << 16))
        keypoint.size = sigma * (2 ** ((image_idx + x[2]) / self.n_octave_layers)) * (2 ** (octave_idx + 1))
        keypoint.response = abs(contr)
        
        return (keypoint, image_idx)
        

    def calc_keypoint_with_orientations(self, keypoint: cv2.KeyPoint, image: np.ndarray, octave_idx: int) -> List[cv2.KeyPoint]:
        """
        calculate orientations for each keypoint
        """
        scl_octv = keypoint.size * 0.5 / (2 ** octave_idx)
        radius = int(np.round(4.5 * scl_octv))
        expf_scale = -1.0 / (2.0 * ((self.ori_sig_fctr * scl_octv) ** 2))
        num_bins = 36
        raw_hist = np.zeros(num_bins)
        smooth_hist = np.zeros(num_bins)
    
        for i in range(-radius, radius + 1):
            y = int(keypoint.pt[1] / (2 ** octave_idx)) + i
            if y <= 0 or y >= image.shape[0] - 1:
                continue

            for j in range(-radius, radius + 1):
                x = int(keypoint.pt[0] / (2 ** octave_idx)) + j
                if x <= 0 or x >= image.shape[1] - 1:
                    continue
                
                dx = image[y, x+1] - image[y, x-1]
                dy = image[y-1, x] - image[y+1, x]
                
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
                if abs(angle - 360.0) < self.flt_epsilon:
                    angle = 0.0
                keypoints.append(cv2.KeyPoint(*keypoint.pt, keypoint.size, angle, keypoint.response, keypoint.octave))
                
        return keypoints


    def remove_duplicated_sorted(self, keypoints: List[cv2.KeyPoint]) -> List[cv2.KeyPoint]:
        """
        remove duplicated keypoints and sort it
        """
        if len(keypoints) <= 1:
            return keypoints

        keypoints.sort(key=lambda kpt: (kpt.pt, kpt.size, kpt.angle, kpt.response, kpt.octave, kpt.class_id))
        unique_keypoints = [keypoints[0]]
        
        for next_keypoint in keypoints[1:]:
            last = unique_keypoints[-1]
            if last.pt != next_keypoint.pt or last.size != next_keypoint.size or last.angle != next_keypoint.angle:
                unique_keypoints.append(next_keypoint)
        return unique_keypoints
        
    def convert_keypoints_size(self, keypoints: List[cv2.KeyPoint]) -> List[cv2.KeyPoint]:
        """
        convert keypoints to input image size
        """
        converted_keypoints = []
        for keypoint in keypoints:
            keypoint.pt = tuple(0.5 * np.array(keypoint.pt))
            keypoint.size *= 0.5
            keypoint.octave = (keypoint.octave & ~255) | ((keypoint.octave - 1) & 255)
            converted_keypoints.append(keypoint)
        return converted_keypoints


    def unpack_octave(keypoint: cv2.KeyPoint) -> Tuple[int, int, float]:
        """
        return octave, layer, scale from keypoint
        """
        octave = keypoint.octave & 255
        layer = (keypoint.octave >> 8) & 255
        octave = octave if octave < 128 else (-128 | octave)
        scale = 1.0 / (1 << octave) if octave >= 0 else (1 << -octave)
        
        return octave, layer, scale

        
    def calc_sift_descriptors(self, keypoints: List[cv2.KeyPoint], images: np.ndarray) -> np.ndarray:
        """
        calulate sift descriptors for each keypoint
        """
        descriptors = []
        window_width = 4
        num_bins = 8
        
        for keypoint in keypoints:
            octave, layer, scale = SIFT.unpack_octave(keypoint)
            image = images[octave+1, layer] 
            num_rows, num_cols = image.shape
            point = np.round(scale * np.array(keypoint.pt)).astype('int')
            
            angle = 360.0 - keypoint.angle
            cos_t = np.cos(np.deg2rad(angle))
            sin_t = np.sin(np.deg2rad(angle))
            
            bins_per_deg = num_bins / 360.0
            exp_scale = -1.0 / (window_width ** 2 * 0.5)

            hist_width = self.descr_scl_fctr * 0.5 * scale * keypoint.size
            radius = int(np.round(hist_width * np.sqrt(2) * (window_width + 1) * 0.5))
            # clip the radius to the diagonal of the image to avoid autobuffer too large exception
            radius = min(radius, int(np.sqrt(num_rows ** 2 + num_cols ** 2)))

            cos_t /= hist_width
            sin_t /= hist_width
            
            row_bin_list = []
            col_bin_list = []
            mag_list = []
            ori_bin_list = []
            hist = np.zeros((window_width+2, window_width+2, num_bins))

            for row in range(-radius, radius + 1):
                for col in range(-radius, radius + 1):
                    col_rot = col * cos_t - row * sin_t
                    row_rot = col * sin_t + row * cos_t
                    row_bin = row_rot + window_width / 2 - 0.5
                    col_bin = col_rot + window_width / 2 - 0.5

                    if row_bin > -1 and row_bin < window_width and col_bin > -1 and col_bin < window_width:
                        window_row = int(np.round(point[1] + row))
                        window_col = int(np.round(point[0] + col))
                        
                        if window_row > 0 and window_row < num_rows - 1 and window_col > 0 and window_col < num_cols - 1:
                            dx = image[window_row, window_col+1] - image[window_row, window_col-1]
                            dy = image[window_row-1, window_col] - image[window_row+1, window_col]
                            weight = np.exp((col_rot ** 2 + row_rot ** 2) * exp_scale)
                            mag = np.sqrt(dx ** 2 + dy ** 2)
                            ori = np.rad2deg(np.arctan2(dy, dx)) % 360
                            row_bin_list.append(row_bin)
                            col_bin_list.append(col_bin)
                            mag_list.append(weight * mag)
                            ori_bin_list.append((ori - angle) * bins_per_deg)
            
            for row_bin, col_bin, mag, ori_bin in zip(row_bin_list, col_bin_list, mag_list, ori_bin_list):
                # trilinear interpolation
                row0, col0, ori0 = np.floor([row_bin, col_bin, ori_bin]).astype('int')
                row_bin -= row0
                col_bin -= col0
                ori_bin -= ori0
                
                ori0 = ori0 + num_bins if ori0 < 0 else ori0
                ori0 = ori0 - num_bins if ori0 >= num_bins else ori0
                
                r1 = mag * row_bin
                r0 = mag * (1 - row_bin)
                rc11 = r1 * col_bin
                rc10 = r1 * (1 - col_bin)
                rc01 = r0 * col_bin
                rc00 = r0 * (1 - col_bin)
                rco111 = rc11 * ori_bin
                rco110 = rc11 * (1 - ori_bin)
                rco101 = rc10 * ori_bin
                rco100 = rc10 * (1 - ori_bin)
                rco011 = rc01 * ori_bin
                rco010 = rc01 * (1 - ori_bin)
                rco001 = rc00 * ori_bin
                rco000 = rc00 * (1 - ori_bin)
                
                # base index is (row0+1, col0+1, ori0)
                hist[row0 + 1, col0 + 1, ori0] += rco000
                hist[row0 + 1, col0 + 1, (ori0 + 1) % num_bins] += rco001
                hist[row0 + 1, col0 + 2, ori0] += rco010
                hist[row0 + 1, col0 + 2, (ori0 + 1) % num_bins] += rco011
                hist[row0 + 2, col0 + 1, ori0] += rco100
                hist[row0 + 2, col0 + 1, (ori0 + 1) % num_bins] += rco101
                hist[row0 + 2, col0 + 2, ori0] += rco110
                hist[row0 + 2, col0 + 2, (ori0 + 1) % num_bins] += rco111
                
            # window_width * window_width * num_bins = 128
            descriptor = hist[1:-1, 1:-1, :].flatten()
            threshold = np.linalg.norm(descriptor) * self.descr_mag_thr
            descriptor[descriptor > threshold] = threshold
            descriptor /= max(np.linalg.norm(descriptor), self.flt_epsilon)
            # convert float descriptor to unsigned char 
            descriptor = np.round(512 * descriptor)
            descriptor[descriptor < 0] = 0
            descriptor[descriptor > 255] = 255
            descriptors.append(descriptor)
            
        return np.array(descriptors, dtype='float32')