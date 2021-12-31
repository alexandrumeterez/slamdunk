import numpy as np
import cv2

def ratio_test(matches, kp1, kp2, magic):
    pts1 = []
    pts2 = []

    for i, (m, n) in enumerate(matches):
        if m.distance < magic * n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    return np.int32(pts1), np.int32(pts2)

def get_R_t(E):
    U, s, Vt = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    t = U @ W @ s @ U.T
    R = U @ W.T @ Vt

    return t, R