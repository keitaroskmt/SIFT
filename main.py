import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    size = (1700, 2000)
    img1 = cv2.imread('./images/dict1.jpeg')
    img2 = cv2.imread('./images/dict2.jpeg')
    img1 = cv2.resize(img1, size)
    img2 = cv2.resize(img2, size)
    
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    good = []
    for match1, match2 in matches:
        if match1.distance < 0.75 * match2.distance:
            good.append([match1])
    
    sift_matches= cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

    plt.imshow(sift_matches, cmap='gray')
    plt.show()
    

