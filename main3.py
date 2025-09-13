import sys
import os
import cv2
import numpy as np
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QGraphicsRectItem, QGraphicsView, QGraphicsScene, QGraphicsItem, QStyleOptionGraphicsItem
from PyQt5.QtCore import Qt, QRectF, QSizeF, QPointF
from PyQt5.QtGui import QPen, QBrush, QCursor

class ResizableBox(QGraphicsRectItem):
    def __init__(self, x, y, w, h):
        super().__init__(x, y, w, h)
        self.setFlags(
            QGraphicsItem.ItemIsSelectable |
            QGraphicsItem.ItemSendsGeometryChanges
        )
        self.setPen(QPen(Qt.red, 5))
        self.setBrush(QBrush(Qt.transparent))
        self.setAcceptHoverEvents(True)
        self.edge_margin = 6
        self.resize_direction = None

    def hoverMoveEvent(self, event):
        pos = event.pos()
        rect = self.rect()
        if abs(pos.x() - rect.left()) < self.edge_margin:
            self.setCursor(Qt.SizeHorCursor)
            self.resize_direction = 'left'
        elif abs(pos.x() - rect.right()) < self.edge_margin:
            self.setCursor(Qt.SizeHorCursor)
            self.resize_direction = 'right'
        elif abs(pos.y() - rect.top()) < self.edge_margin:
            self.setCursor(Qt.SizeVerCursor)
            self.resize_direction = 'top'
        elif abs(pos.y() - rect.bottom()) < self.edge_margin:
            self.setCursor(Qt.SizeVerCursor)
            self.resize_direction = 'bottom'
        else:
            self.setCursor(Qt.ArrowCursor)
            self.resize_direction = None
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event):
        self.start_pos = event.pos()
        self.start_rect = self.rect()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.resize_direction:
            delta = event.pos() - self.start_pos
            rect = QRectF(self.start_rect)
            if self.resize_direction == 'left':
                rect.setLeft(rect.left() + delta.x())
            elif self.resize_direction == 'right':
                rect.setRight(rect.right() + delta.x())
            elif self.resize_direction == 'top':
                rect.setTop(rect.top() + delta.y())
            elif self.resize_direction == 'bottom':
                rect.setBottom(rect.bottom() + delta.y())
            self.setRect(rect)
        super().mouseMoveEvent(event)

# Insert this at the top
SEGMENT_WIDTH = 100  # Adjustable sub-box width

class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Box Viewer")
        self.resize(1200, 900)

        self.scene = QGraphicsScene()
        self.graphic = QGraphicsView(self.scene)
        self.graphic.setMouseTracking(True)
        self.graphic.setRenderHint(QtGui.QPainter.Antialiasing)
        self.graphic.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.graphic.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.graphic.setDragMode(QGraphicsView.ScrollHandDrag)
        self.initial_zoom_applied = False  

        self.pbLoadImage = QtWidgets.QPushButton("Load Image Pair (Background then Sample)")
        self.pbLoadImage.clicked.connect(self.loadImage)

        self.vlayout = QtWidgets.QVBoxLayout()
        self.vlayout.addWidget(self.graphic)
        self.vlayout.addWidget(self.pbLoadImage)

        self.pbDone = QtWidgets.QPushButton("Done")
        self.pbDone.clicked.connect(self.recordBoxCoordinates)
        self.vlayout.addWidget(self.pbDone)

        self.setLayout(self.vlayout)
        self.current_box = None
        self.sub_boxes = []

    def loadImage(self):
        bg_filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Background Image")
        if not bg_filename:
            return

        sample_filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Sample Image")
        if not sample_filename:
            return

        processed_image_path, main_box, sub_boxes = self.processImage(bg_filename, sample_filename)

        if processed_image_path:
            pixmap = QtGui.QPixmap(processed_image_path)
            self.scene.clear()
            self.scene.addPixmap(pixmap)

            rect = QtCore.QRectF(pixmap.rect())
            self.scene.setSceneRect(rect)

            if not self.initial_zoom_applied:
                self.graphic.fitInView(rect, Qt.KeepAspectRatio)
                self.initial_zoom_applied = True

            if self.current_box:
                self.scene.removeItem(self.current_box)
                self.current_box = None

            # Draw red resizable main box
            x1, y1, w, h = main_box
            self.current_box = ResizableBox(x1, y1, w, h)
            self.scene.addItem(self.current_box)

            # # Draw blue sub-boxes
            # self.sub_boxes = []
            # for x, y, w, h in sub_boxes:
            #     box = QGraphicsRectItem(x, y, w, h)
            #     box.setPen(QPen(Qt.blue, 1))
            #     self.scene.addItem(box)
            #     self.sub_boxes.append(box)

    def recordBoxCoordinates(self):
        if self.current_box:
            # Get current red box dimensions
            rect = self.current_box.rect()
            x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
            print("Original Box:")
            print(f"  Top-left: ({int(x)}, {int(y)})")
            print(f"  Width:    {int(w)} px")
            print(f"  Height:   {int(h)} px")

        # Calculate segment-aligned bounds
            num_segments = int(np.ceil(w / SEGMENT_WIDTH))
            new_width = num_segments * SEGMENT_WIDTH
            pad = new_width - w
            x1 = x - pad / 2
            x1 = max(0, min(x1, self.scene.width() - new_width))  # keep within image

            for box in self.sub_boxes:
                self.scene.removeItem(box)
            self.sub_boxes.clear()


            self.current_box.setRect(x1, y, new_width, h)

 
            for i in range(num_segments):
                xi = x1 + i * SEGMENT_WIDTH
                box = QGraphicsRectItem(xi, y, SEGMENT_WIDTH, h)
                box.setPen(QPen(Qt.blue, 1))
                self.scene.addItem(box)
                self.sub_boxes.append(box)

            print("Updated (Snapped) Box:")
            print(f"  Top-left: ({int(x1)}, {int(y)})")
            print(f"  Width:    {int(new_width)} px")
            print(f"  Height:   {int(h)} px")
            print("Sub-Box Coordinates:")
            for i, box in enumerate(self.sub_boxes):
                rect = box.rect()
                x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
                print(f"  Sub-box {i+1}: ({int(x)}, {int(y)}) width={int(w)} height={int(h)}")



    def processImage(self, background_path, sample_path):
        output_path = "difference_box_output.png"

        if not os.path.exists(background_path) or not os.path.exists(sample_path):
            print("Error: Image file not found!")
            return None, (0, 0, 0, 0), []

        bg_img = cv2.imread(background_path)
        sample_img = cv2.imread(sample_path)

        if bg_img is None or sample_img is None:
            print("Error: Unable to load images.")
            return None, (0, 0, 0, 0), []

        if bg_img.shape != sample_img.shape:
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
            return None, (0, 0, 0, 0), []

        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Centered buffer logic
        num_segments = int(np.ceil(w / SEGMENT_WIDTH))
        new_width = num_segments * SEGMENT_WIDTH
        pad = new_width - w
        x1 = max(x - pad // 2, 0)
        x1 = min(x1, sample_img.shape[1] - new_width)
        box_width = new_width
        box_height = h

        # Build sub-boxes
        sub_boxes = []
        for i in range(num_segments):
            xi = x1 + i * SEGMENT_WIDTH
            sub_boxes.append((xi, y, SEGMENT_WIDTH, h))

        # Save image (optional)
        cv2.imwrite(output_path, sample_img)

        return output_path, (x1, y, box_width, box_height), sub_boxes
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    mw = MainWindow()
    mw.show()
    sys.exit(app.exec_())
