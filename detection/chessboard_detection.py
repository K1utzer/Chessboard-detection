from typing import Sequence
from .bounding_box import BoundingBox
import cv2
import numpy as np

CANNY_THRESHOLD_LOW = 50
CANNY_THRESHOLD_HIGH = 150
MORPHOLOGY_KERNEL_SIZE = 5
MINIMUM_BOARD_AREA = 500
MINIMUM_ASPECT_RATIO = 0.9
MAXIMUM_ASPECT_RATIO = 1.1
EPSILON_FACTORS = np.linspace(0.01, 0.1, 10)

HOUGH_RHO_RESOLUTION = 1
HOUGH_THETA_RESOLUTION = np.pi / 180
HOUGH_THRESHOLD = 80
MIN_GRID_LINE_COVERAGE = 0.50
HOUGH_MIN_LINE_LENGTH = 50
HOUGH_MAX_LINE_GAP = 10

LINE_ANGLE_TOLERANCE = 10

LINE_MERGE_DISTANCE = 15

GRID_SPACING_TOLERANCE = 0.20


MatLike = np.ndarray

def process_image(screenshot: MatLike) -> BoundingBox  | None:
    gray = cv2.cvtColor(
        screenshot,
        cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(
        gray,
        CANNY_THRESHOLD_LOW,
        CANNY_THRESHOLD_HIGH,
        apertureSize=3)

    kernel = np.ones(
        (MORPHOLOGY_KERNEL_SIZE, MORPHOLOGY_KERNEL_SIZE),
        np.uint8)

    processed_edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel
    )
    contours, _ = cv2.findContours(
        processed_edges,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE)

    candidate_boxes = find_square_candidates(contours)

    for candidate_box in candidate_boxes:
        if is_chessboard(
            processed_edges[candidate_box.y:
            candidate_box.y + candidate_box.height,
            candidate_box.x:
            candidate_box.x + candidate_box.width
            ]):
                return candidate_box

    return None

def find_square_candidates(contours: Sequence[MatLike])-> list[BoundingBox]:
    largest_area  = 0.0
    candidate_boxes  = []

    for contour in contours:
        contour_area = cv2.contourArea(contour)

        if contour_area < MINIMUM_BOARD_AREA:
            continue

        perimeter  = cv2.arcLength(contour, True)

        for epsilon_factor in EPSILON_FACTORS:
            epsilon = epsilon_factor * perimeter

            approx = cv2.approxPolyDP(
                contour,
                epsilon,
                True)

            if len(approx) != 4:
                continue

            x, y, width, height = cv2.boundingRect(approx)

            aspect_ratio = width / float(height)

            if not (
                MINIMUM_ASPECT_RATIO
                < aspect_ratio
                < MAXIMUM_ASPECT_RATIO
            ):
                continue

            candidate_boxes.append(BoundingBox(
                x=x,
                y=y,
                width=width,
                height=height
            ))
            break

    candidate_boxes.sort(
        key=lambda box: box.width * box.height,
        reverse=True
    )

    return candidate_boxes

def is_chessboard(image: MatLike) -> bool:
    lines = cv2.HoughLinesP(
        image,
        rho=HOUGH_RHO_RESOLUTION,
        theta=HOUGH_THETA_RESOLUTION,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LENGTH,
        maxLineGap=HOUGH_MAX_LINE_GAP
    )

    if lines is None:
        return False

    lines = np.asarray(lines).reshape(-1, 4)

    image_height, image_width = image.shape[:2]

    minimum_horizontal_length = (
        image_width * MIN_GRID_LINE_COVERAGE
    )

    minimum_vertical_length = (
        image_height * MIN_GRID_LINE_COVERAGE
    )

    vertical_positions: list[int] = []
    horizontal_positions: list[int] = []

    for line in lines:
        x1, y1, x2, y2 = line

        delta_x = x2 - x1
        delta_y = y2 - y1

        angle = abs(np.degrees(np.arctan2(delta_y, delta_x)))

        # horizontal line
        if angle < LINE_ANGLE_TOLERANCE  or angle > 180 - LINE_ANGLE_TOLERANCE :
            line_length = abs(delta_x)

            if line_length < minimum_horizontal_length:
                continue

            y_position = int((y1 + y2) / 2)
            horizontal_positions.append(y_position)

        # vertical line
        elif abs(angle - 90) < LINE_ANGLE_TOLERANCE:
            line_length = abs(delta_y)

            if line_length < minimum_vertical_length:
                continue

            x_position = int((x1 + x2) / 2)
            vertical_positions.append(x_position)

    horizontal_positions = merge_line_positions(
        horizontal_positions
    )

    vertical_positions = merge_line_positions(
        vertical_positions
    )

    print(f"Horizontal candidates: {horizontal_positions}")
    print(f"Vertical candidates: {vertical_positions}")

    print(f"Raw Hough lines: {len(lines)}")
    print(f"Horizontal lines: {len(horizontal_positions)} -> {horizontal_positions}")
    print(f"Vertical lines: {len(vertical_positions)} -> {vertical_positions}")

    horizontal_spacing = image_height / 8
    vertical_spacing = image_width / 8

    if not has_regular_spacing(
        horizontal_positions,
        image_height / 8
    ):
        return False

    if not has_regular_spacing(
        vertical_positions,
        image_width / 8
    ):
        return False

    return True


def merge_line_positions(positions: list[int]) -> list[int]:

    if not positions:
        return []

    sorted_positions = sorted(positions)

    groups: list[list[int]] = [
        [sorted_positions[0]]
    ]

    for position in sorted_positions[1:]:
        current_group = groups[-1]

        group_center = int(np.mean(current_group))

        if abs(position - group_center) <= LINE_MERGE_DISTANCE:
            current_group.append(position)
        else:
            groups.append([position])

    return [
        int(np.mean(group))
        for group in groups
    ]

def has_regular_spacing(positions: list[int],expected_spacing: float) -> bool:
    if len(positions) < 6:
        return False

    tolerance = expected_spacing * GRID_SPACING_TOLERANCE

    matched_grid_lines: set[int] = set()

    for position in positions:
        grid_index = round(position / expected_spacing)

        if grid_index < 0 or grid_index > 8:
            continue

        expected_position = grid_index * expected_spacing

        if abs(position - expected_position) <= tolerance:
            matched_grid_lines.add(grid_index)

    return len(matched_grid_lines) >= 7
