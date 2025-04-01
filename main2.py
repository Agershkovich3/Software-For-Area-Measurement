import cv2
import numpy as np
import os

def draw_largest_difference_box(background_path, sample_path, output_path="difference_box_output.png"):
    if not os.path.exists(background_path) or not os.path.exists(sample_path):
        print("Error: One or both image paths do not exist.")
        return

    bg_img = cv2.imread(background_path)
    sample_img = cv2.imread(sample_path)

    if bg_img is None or sample_img is None:
        print("Error: Unable to load one or both images.")
        return

    if bg_img.shape != sample_img.shape:
        print("Resizing sample to match background...")
        sample_img = cv2.resize(sample_img, (bg_img.shape[1], bg_img.shape[0]))

    bg_blur = cv2.GaussianBlur(bg_img, (5, 5), 0)
    sample_blur = cv2.GaussianBlur(sample_img, (5, 5), 0)

    bg_lab = cv2.cvtColor(bg_blur, cv2.COLOR_BGR2LAB)
    sample_lab = cv2.cvtColor(sample_blur, cv2.COLOR_BGR2LAB)


    diff = cv2.absdiff(bg_lab, sample_lab)

    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_diff, 10, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    thresh = cv2.erode(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No significant difference found.")
        return

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    padding = 0
    x1 = max(x - padding, 0)
    y1 = max(y - padding, 0)
    x2 = min(x + w + padding, sample_img.shape[1] - 1)
    y2 = min(y + h + padding, sample_img.shape[0] - 1)

    box_width = x2 - x1
    box_height = y2 - y1

    print(f"Sample Bounding Box:")
    print(f"  Top-left:     ({x1}, {y1})")
    print(f"  Bottom-right: ({x2}, {y2})")
    print(f"  Width:        {box_width} px")
    print(f"  Height:       {box_height} px")


    cv2.rectangle(sample_img, (x1, y1), (x2, y2), (0, 0, 255), 4)

   
    cv2.imwrite(output_path, sample_img)
    print(f"Saved output with largest bounding box to: {output_path}")

if __name__ == "__main__":
    draw_largest_difference_box("img_4171.png", "img_4172.png")
from PyQt5.QtWidgets import QGraphicsRectItem
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QPen

class DraggableBox(QGraphicsRectItem):
    def __init__(self, x, y, w, h):
        super().__init__(x, y, w, h)
        self.setFlags(
            QGraphicsRectItem.ItemIsMovable |
            QGraphicsRectItem.ItemIsSelectable |
            QGraphicsRectItem.ItemSendsGeometryChanges
        )
        self.setPen(QPen(Qt.red, 3))

    def itemChange(self, change, value):
        if change == QGraphicsRectItem.ItemPositionChange:
            # Optional: clamp position or print updates
            print(f"Box moved to: {value.x()}, {value.y()}")
        return super().itemChange(change, value)
