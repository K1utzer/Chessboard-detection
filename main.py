import argparse
import mss
import numpy as np
import cv2
import os
from chessboard_detection import process_image

MatLike = np.ndarray

def save_image(image: MatLike, filename: str, debug_dir: str) -> None:
    cv2.imwrite(os.path.join(debug_dir, filename), image)

def capture_all_monitors():
    with mss.mss() as sct:
        #monitor = sct.monitors[1]
        #return np.array(sct.grab(monitor))
        monitors = sct.monitors[1:]
        screenshots = []
        for monitor in monitors:
            screenshot = np.array(sct.grab(monitor))
            screenshot = screenshot[:, :, :3]
            screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
            screenshots.append(screenshot)
        combined_screenshot = cv2.hconcat(screenshots)
        return combined_screenshot

def main(image_show: bool = False, image_save: bool = False) -> None:
    if image_save:
        debug_dir = "image_output"
        os.makedirs(debug_dir, exist_ok=True)

    screenshot = capture_all_monitors()

    coordinates = process_image(screenshot)

    if(coordinates is not None):
        x, y, w, h = coordinates
        marked_img = screenshot.copy()
        cv2.drawContours(marked_img, [cv2.approxPolyDP(np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]), 0, True)], -1, (0, 255, 0), 3)
        print(f"Board found: start point ({x}, {y}), width {w}, height {h}")
        chessboard_img = marked_img[y:y+h, x:x+w]
        if image_save:
            save_image(marked_img, "01_chessboard_marked.png", debug_dir)
            save_image(chessboard_img, "02_chessboard.png", debug_dir)
        if image_show:
            width = 1280
            scale_factor = width / marked_img.shape[1]
            new_height = int(marked_img.shape[0] * scale_factor)
            resized_image = cv2.resize(marked_img, (width, new_height))
            resized_image2 = cv2.resize(chessboard_img, (new_height, new_height))
            images_to_show = cv2.hconcat([resized_image, resized_image2])
            cv2.imshow('detected chessboard', images_to_show)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print("No Board found")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_show", help="Show Images.", action="store_true")
    parser.add_argument("--image_save", help="Save Images to File.", action="store_true")
    args = parser.parse_args()
    main(image_show=args.image_show, image_save=args.image_save)
