"""Motion-based filtering utilities for video frames and clips."""
import cv2, numpy as np

def flow_motion_score(prev_gray, cur_gray, downscale=0.25):
    h, w = prev_gray.shape
    small = cv2.resize(prev_gray, (0,0), fx=downscale, fy=downscale)
    small_next = cv2.resize(cur_gray, (0,0), fx=downscale, fy=downscale)
    flow = cv2.calcOpticalFlowFarneback(small, small_next, None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)
    mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
    return np.clip(mag.mean() / (255*downscale), 0, 1)

def is_active(prev_gray, cur_gray, thresh=0.15):
    return flow_motion_score(prev_gray, cur_gray) > thresh