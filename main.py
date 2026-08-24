MINIMUM_PYTHON_VERSION = (3, 10)
import sys
if sys.version_info < MINIMUM_PYTHON_VERSION:
    raise RuntimeError(
        f"Python {MINIMUM_PYTHON_VERSION[0]}.{MINIMUM_PYTHON_VERSION[1]} or newer is required. "
        f"Current version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )

import argparse
import mss
import numpy as np
import cv2
import os
from detection.chessboard_detection import process_image

PREVIEW_WIDTH = 1280

MatLike = np.ndarray

def save_image(image: MatLike, filename: str, debug_dir: str) -> None:
    cv2.imwrite(os.path.join(debug_dir, filename), image)

def capture_all_monitors():
    with mss.MSS() as screen_capture:
        #monitor = sct.monitors[1]
        #return np.array(sct.grab(monitor))
        virtual_monitor = screen_capture.monitors[0]
        screenshot = np.array(screen_capture.grab(virtual_monitor))
        return screenshot[:, :, :3]

def main(image_show: bool = False, image_save: bool = False) -> None:
    if image_save:
        debug_dir = "image_output"
        os.makedirs(debug_dir, exist_ok=True)

    screenshot = capture_all_monitors()

    board_box = process_image(screenshot)

    if board_box is None:
        print("No Board found")
        return

    chessboard_img = screenshot[
        board_box.y:
        board_box.y + board_box.height,
        board_box.x:
        board_box.x + board_box.width] \
    .copy()

    marked_img = screenshot.copy()

    cv2.rectangle(
        marked_img,
        (board_box.x, board_box.y),
        (board_box.x + board_box.width,
            board_box.y + board_box.height),
        (0, 255, 0),
        3
    )
    print(f"Board found: start point ({board_box.x}, {board_box.y}), width {board_box.width}, height {board_box.height}")

    if image_save:
        save_image(marked_img, "01_chessboard_marked.png", debug_dir)
        save_image(chessboard_img, "02_chessboard.png", debug_dir)

    if image_show:
        scale_factor = PREVIEW_WIDTH / marked_img.shape[1]
        new_height = int(marked_img.shape[0] * scale_factor)

        board_scale = new_height / chessboard_img.shape[0]
        board_width = int(chessboard_img.shape[1] * board_scale)

        resized_image = cv2.resize(marked_img, (PREVIEW_WIDTH, new_height))
        resized_image2 = cv2.resize(chessboard_img, (board_width, new_height))

        images_to_show = cv2.hconcat([resized_image, resized_image2])

        cv2.imshow('detected chessboard', images_to_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_show", help="Show Images.", action="store_true")
    parser.add_argument("--image_save", help="Save Images to File.", action="store_true")
    args = parser.parse_args()
    main(image_show=args.image_show, image_save=args.image_save)
