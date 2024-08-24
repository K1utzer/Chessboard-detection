import typing
import cv2
import numpy as np

MatLike = np.ndarray

def process_image(screenshot: MatLike) -> tuple[int, int, int, int] | None:
    gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    kernel = np.ones((5, 5), np.uint8)
    eroded = cv2.erode(cv2.dilate(edges, kernel, iterations=1), kernel, iterations=1)
    contours, _ = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return find_chessboard_coordinates(contours)

def find_chessboard_coordinates(contours: typing.Sequence[MatLike])-> tuple[int, int, int, int] | None:
    max_area = 0
    chessboard_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if(area < 1000): # board should be a larger contour Area
            continue
        for epsilon_factor in np.linspace(0.01, 0.1, 10):
            epsilon = epsilon_factor * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if(len(approx) == 4):
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)
                if(0.9 < aspect_ratio < 1.1 and area > max_area):
                    chessboard_contour = approx
                    max_area = area
    if(chessboard_contour is not None):
        x, y, w, h = cv2.boundingRect(chessboard_contour)
        return x, y, w, h
    else:
        return None
