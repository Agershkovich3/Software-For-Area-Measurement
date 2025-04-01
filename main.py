import sys
import os
import cv2
import numpy as np
from PyQt5 import QtCore, QtWidgets, QtGui
from PIL import Image, ImageQt

class MyQGraphicsScene(QtWidgets.QGraphicsScene):
    """ Custom QGraphicsScene with mouse tracking and event handling """
    def __init__(self):
        super().__init__()
        self.mouse_move_callback = None
        self.double_click_callback = None

    def setMouseMoveCallback(self, callback):
        self.mouse_move_callback = callback

    def setDoubleClickCallback(self, callback):
        self.double_click_callback = callback

    def mouseMoveEvent(self, event):
        if self.mouse_move_callback:
            self.mouse_move_callback(event.scenePos().x(), event.scenePos().y())
        super().mouseMoveEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self.double_click_callback:
            self.double_click_callback(event.scenePos().x(), event.scenePos().y())
        super().mouseDoubleClickEvent(event)

class MainWindow(QtWidgets.QWidget):
    """ Main application window with interactive features """
    def __init__(self):
        super().__init__()

        # Setup scene and graphics view
        self.scene = MyQGraphicsScene()
        self.scene.setMouseMoveCallback(self.updateReadout)
        self.scene.setDoubleClickCallback(self.doubleClick)
        self.graphic = QtWidgets.QGraphicsView(self.scene)
        self.graphic.setMouseTracking(True)

        # UI Elements
        self.readout = QtWidgets.QLabel("x =    y = ")
        self.pbAddLine = QtWidgets.QPushButton("Add Line")
        self.pbDone = QtWidgets.QPushButton("Done")
        self.pbLoadImage = QtWidgets.QPushButton("Load Image")

        # Button connections
        self.pbAddLine.clicked.connect(self.addLine)
        self.pbDone.clicked.connect(self.makeReport)
        self.pbLoadImage.clicked.connect(self.loadImage)

        # Layouts
        self.vlayout = QtWidgets.QVBoxLayout()
        self.hlayout = QtWidgets.QHBoxLayout()

        self.hlayout.addWidget(self.readout)
        self.hlayout.addWidget(self.pbAddLine)
        self.hlayout.addWidget(self.pbDone)
        self.hlayout.addWidget(self.pbLoadImage)

        self.vlayout.addWidget(self.graphic)
        self.vlayout.addLayout(self.hlayout)

        self.setLayout(self.vlayout)

        self.scene.addRect(0, 0, 256, 256)
        self.lines = []

    def updateReadout(self, x, y):
        """ Updates the QLabel with mouse coordinates """
        self.readout.setText(f"x = {x:.2f}  y = {y:.2f}")

    def doubleClick(self, x, y):
        """ Handles double-click events """
        print(f"Double-click at ({x:.2f},{y:.2f})")

    def addLine(self):
        """ Adds a draggable blue line to the scene """
        lineItem = self.scene.addLine(32, 32, 32, 64, QtGui.QPen(QtCore.Qt.blue))
        lineItem.setFlags(lineItem.flags() | QtWidgets.QGraphicsItem.ItemIsMovable)
        self.lines.append(lineItem)

    def makeReport(self):
        """ Prints the positions of all lines """
        for line in self.lines:
            x1 = line.pos().x() + line.line().x1()
            y1 = line.pos().y() + line.line().y1()
            x2 = line.pos().x() + line.line().x2()
            y2 = line.pos().y() + line.line().y2()
            print(f"Line from ({x1:.2f},{y1:.2f}) to ({x2:.2f},{y2:.2f})")

    def loadImage(self):
        """ Opens file dialog, processes image, and displays it """
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Image")
        if not filename:
            return

        processed_image_path = self.processImage(filename)

        if processed_image_path:
            pixmap = QtGui.QPixmap(processed_image_path)
            self.scene.clear()  # Clear previous items
            self.scene.addPixmap(pixmap)

            # Convert QRect to QRectF to fix the previous error
            rect = QtCore.QRectF(pixmap.rect())
            self.scene.setSceneRect(rect)

    def processImage(self, image_path):
        """ Processes image to detect non-white regions and draws a bounding box """
        output_path = "processed_output.png"

        if not os.path.exists(image_path):
            print("Error: Image file not found!")
            return None

        # Load image
        image = cv2.imread(image_path)

        if image is None:
            print("Error: Unable to load image.")
            return None

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Threshold to detect non-white regions
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            x, y, w, h = cv2.boundingRect(np.vstack(contours))
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle

        # Save the processed image
        cv2.imwrite(output_path, image)
        return output_path

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    mw = MainWindow()
    mw.show()
    sys.exit(app.exec_())
