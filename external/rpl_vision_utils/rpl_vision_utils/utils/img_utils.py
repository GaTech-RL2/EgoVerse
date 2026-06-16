"""Utility functions that are standard to process """
import numpy as np
import cv2

def cv2_draw_polygon(img: np.ndarray,
                     points: list=[],
                     color: tuple=(22, 22, 22),
                     line_thickness: int=2) -> None:
    for i in range(len(points) - 1):
        cv2.line(img, points[i], points[i+1], color, thickness=line_thickness)
    cv2.line(img, points[len(points)-1], points[0], color, thickness=line_thickness)

def resize_img(img: np.ndarray, 
               camera_type: str, 
               img_w: int=128, 
               img_h: int=128, 
               offset_w: int=0, 
               offset_h: int=0,
               fx: float=None,
               fy: float=None) -> np.ndarray:
    if camera_type == "k4a":
        if fx is None:
            fx = 0.2
        if fy is None:
            fy = 0.2
        resized_img = cv2.resize(img, (0, 0), fx=fx, fy=fy)
        w = resized_img.shape[0]
        h = resized_img.shape[1]
    if camera_type == "rs":
        if fx is None:
            fx = 0.2
        if fy is None:
            fy = 0.3
        resized_img = cv2.resize(img, (0, 0), fx=fx, fy=fy)
        w = resized_img.shape[0]
        h = resized_img.shape[1]
    resized_img = resized_img[w//2-img_w//2:w//2+img_w//2, h//2-img_h//2:h//2+img_h//2, :]
    return resized_img


def cv2_undistort(img: np.ndarray,
                  intrinsics_matrix,
                  distortion):
    return cv2.undistort(img_color, intrinsics_matrix, distortion, None)

