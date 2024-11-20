import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QTabWidget, QCheckBox, QFormLayout, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget,
                             QPushButton, QGroupBox, QGridLayout, QSlider, QFileDialog, QDialog, QScrollArea, QComboBox, QMessageBox, QMenu, QAction, QInputDialog, QToolBar, QSizePolicy)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from skimage import feature, io
from scipy.ndimage import sobel, gaussian_filter
import heapq
from coshrem.shearletsystem import EdgeSystem
from coshrem.util.image import overlay, mask, thin_mask, curvature_rgb
from coshrem.util.curvature import curvature
import coshrem.util
from sklearn.metrics import precision_recall_fscore_support
from shapely.geometry import LineString
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
                             QPushButton, QFileDialog, QComboBox, QListWidget)
import os
import geopandas as gpd
from shapely.geometry import Polygon
from osgeo import gdal, osr
import rasterio
from skimage import measure
from PyQt5.QtGui import QIcon, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QPointF
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTextEdit
from scipy.spatial.distance import directed_hausdorff
import traceback
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom
import re
from utility import edgelink, cleanedgelist
from PyQt5.QtCore import QTimer
from skimage.morphology import skeletonize
from skimage.morphology import thin
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QPushButton, QGraphicsScene, QGraphicsView, QGraphicsEllipseItem, QGraphicsPathItem, QGraphicsItem)
from PyQt5.QtGui import QImage, QPixmap, QPainterPath, QPen
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QPushButton, QGraphicsScene, QGraphicsView,
    QGraphicsEllipseItem, QGraphicsPathItem, QGraphicsItem, QApplication,
    QHBoxLayout, QMessageBox, QMenu, QAction, QGraphicsPixmapItem
)
from PyQt5.QtGui import QImage, QPixmap, QPainterPath, QPen, QColor
from PyQt5.QtCore import Qt, QPointF, QRectF

from PyQt5.QtGui import QTransform
from collections import deque
import torch
from PyQt5.QtCore import (Qt, pyqtSignal)
from PyQt5.QtWidgets import QSpinBox
 # Define a NodeItem representing control pointsA
# Define a NodeItem representing control points
from skimage import exposure
from skimage.util import img_as_float, img_as_ubyte

class PreprocessingWindow(QDialog):
    def __init__(self, image, parent=None):
        super().__init__(parent)
        self.original_image = image.copy()  # Keep original
        self.current_image = image.copy()   # Working copy
        self.mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
        self.drawing = False
        self.points = []
        self.initUI()

    def initUI(self):
        # Set a larger initial window size
        screen = QApplication.primaryScreen().geometry()
        self.setGeometry(100, 100, min(screen.width() * 0.8, 1200), 
                        min(screen.height() * 0.8, 800))
        self.setWindowTitle('Image Preprocessing - Draw Mask')

        layout = QVBoxLayout()

        # Create the graphics scene and view
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.view.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        layout.addWidget(self.view)

        # Display the original image
        self.update_display()
        
        # Set scene rect to image size
        height, width = self.current_image.shape[:2]
        self.scene.setSceneRect(0, 0, width, height)

        # Add enhancement controls
        enhance_layout = QHBoxLayout()
        self.ahe_button = QPushButton("Apply AHE")
        self.ahe_button.setCheckable(True)
        self.ahe_button.clicked.connect(self.toggle_ahe)
        enhance_layout.addWidget(self.ahe_button)
        layout.addLayout(enhance_layout)

        # Add zoom controls
        zoom_layout = QHBoxLayout()
        self.zoom_in_btn = QPushButton("Zoom In")
        self.zoom_out_btn = QPushButton("Zoom Out")
        self.fit_btn = QPushButton("Fit to View")
        
        self.zoom_in_btn.clicked.connect(lambda: self.zoom(1.2))
        self.zoom_out_btn.clicked.connect(lambda: self.zoom(0.8))
        self.fit_btn.clicked.connect(self.fit_to_view)
        
        zoom_layout.addWidget(self.zoom_in_btn)
        zoom_layout.addWidget(self.zoom_out_btn)
        zoom_layout.addWidget(self.fit_btn)
        layout.addLayout(zoom_layout)

        # Button layout
        button_layout = QHBoxLayout()
        self.clear_button = QPushButton('Clear Mask')
        self.complete_button = QPushButton('Complete Mask')
        self.apply_button = QPushButton('Apply Mask')
        
        self.clear_button.clicked.connect(self.clear_mask)
        self.complete_button.clicked.connect(self.complete_mask)
        self.apply_button.clicked.connect(self.apply_mask)
        
        button_layout.addWidget(self.clear_button)
        button_layout.addWidget(self.complete_button)
        button_layout.addWidget(self.apply_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

        # Install event filter for mouse events
        self.view.viewport().installEventFilter(self)

        # Initialize path for drawing
        self.current_path = QPainterPath()
        self.path_item = None

        # Fit the view to the scene contents
        self.fit_to_view()
    def update_display(self):
        height, width = self.current_image.shape[:2]
        image = QImage(self.current_image.data, width, height, width, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(image)
        if hasattr(self, 'image_item'):
            self.image_item.setPixmap(pixmap)
        else:
            self.image_item = self.scene.addPixmap(pixmap)
    def toggle_ahe(self, checked):
        try:
            if checked:
                # Convert to float and apply AHE
                img_float = img_as_float(self.original_image)
                img_eq = exposure.equalize_adapthist(img_float)
                self.current_image = img_as_ubyte(img_eq)
                self.ahe_button.setText("Remove AHE")
            else:
                self.current_image = self.original_image.copy()
                self.ahe_button.setText("Apply AHE")
            
            self.update_display()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to apply AHE: {str(e)}")
            self.ahe_button.setChecked(False)
    def fit_to_view(self):
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        self.view.centerOn(self.scene.sceneRect().center())

    def zoom(self, factor):
        self.view.scale(factor, factor)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit_to_view()

    def eventFilter(self, obj, event):
        if obj == self.view.viewport():
            if event.type() == event.MouseButtonPress and event.button() == Qt.LeftButton:
                self.start_drawing(event)
            elif event.type() == event.MouseMove and self.drawing:
                self.continue_drawing(event)
            elif event.type() == event.MouseButtonRelease and event.button() == Qt.LeftButton:
                self.end_drawing(event)
        return super().eventFilter(obj, event)

    def start_drawing(self, event):
        scene_pos = self.view.mapToScene(event.pos())
        self.drawing = True
        self.points = [scene_pos]
        self.current_path = QPainterPath()
        self.current_path.moveTo(scene_pos)
        
        if self.path_item:
            self.scene.removeItem(self.path_item)
        self.path_item = self.scene.addPath(self.current_path, 
                                          QPen(Qt.red, 2, Qt.SolidLine))

    def continue_drawing(self, event):
        if self.drawing:
            scene_pos = self.view.mapToScene(event.pos())
            self.points.append(scene_pos)
            self.current_path.lineTo(scene_pos)
            if self.path_item:
                self.scene.removeItem(self.path_item)
            self.path_item = self.scene.addPath(self.current_path, 
                                              QPen(Qt.red, 2, Qt.SolidLine))

    def end_drawing(self, event):
        self.drawing = False
        if len(self.points) > 2:
            self.current_path.lineTo(self.points[0])
            if self.path_item:
                self.scene.removeItem(self.path_item)
            self.path_item = self.scene.addPath(self.current_path, 
                                              QPen(Qt.red, 2, Qt.SolidLine))

    def clear_mask(self):
        if self.path_item:
            self.scene.removeItem(self.path_item)
        self.points = []
        self.current_path = QPainterPath()
        self.mask = np.ones(self.original_image.shape[:2], dtype=np.uint8) * 255

    def complete_mask(self):
        if len(self.points) > 2:
            # Create mask from points
            points = [(p.x(), p.y()) for p in self.points]
            points = np.array(points, dtype=np.int32)
            
            # Create new mask
            self.mask = np.zeros(self.original_image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(self.mask, [points], 255)

            # Show preview of masked image
            masked_image = cv2.bitwise_and(self.original_image, self.original_image, 
                                         mask=self.mask)
            
            # Update the display
            height, width = masked_image.shape[:2]
            image = QImage(masked_image.data, width, height, width, 
                         QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(image)
            self.image_item.setPixmap(pixmap)

    def apply_mask(self):
        if np.any(self.mask):
            self.accept()
        else:
            QMessageBox.warning(self, "Warning", "Please draw a mask first.")

    def get_masked_image(self):
        return cv2.bitwise_and(self.current_image, self.current_image, mask=self.mask)
class NodeItem(QGraphicsEllipseItem):
    def __init__(self, x, y, radius=1, parent=None):
        super().__init__(-radius, -radius, 2*radius, 2*radius, parent)
        self.setPos(x, y)
        self.setBrush(QColor('blue'))
        self.setPen(QPen(Qt.black))
        self.setFlags(
            QGraphicsItem.ItemIsMovable |
            QGraphicsItem.ItemSendsGeometryChanges |
            QGraphicsItem.ItemIsSelectable
        )
        self.lines = []  # Lines connected to this node

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged:
            for line in self.lines:
                line.update_path()
        return super().itemChange(change, value)

    def contextMenuEvent(self, event):
        # Access the parent ManualInterpretationWindow
        parent_window = self.scene().parent()
        if not isinstance(parent_window, ManualInterpretationWindow):
            return

        menu = QMenu()
        delete_action = QAction('Delete Node', self)
        delete_action.triggered.connect(lambda: parent_window.delete_node(self))
        menu.addAction(delete_action)
        menu.exec_(event.screenPos())

# Define a LineItem representing lines composed of nodes
class LineItem(QGraphicsPathItem):
    def __init__(self, nodes=None, parent=None):
        super().__init__()
        self.nodes = nodes if nodes else []
        pen = QPen(QColor('green'))  # Changed color to green for better contrast
        pen.setWidth(1)  # Increased width for better visibility
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        self.setPen(pen)
        self.setFlags(
            QGraphicsItem.ItemIsSelectable  # Lines are selectable but not movable
        )
        self.setZValue(1)  # Ensure lines are above the background
        self.update_path()
        self.parent = parent

    def update_path(self):
        path = QPainterPath()
        if self.nodes:
            path.moveTo(self.nodes[0].pos())
            for node in self.nodes[1:]:
                path.lineTo(node.pos())
        self.setPath(path)

    def add_node(self, node):
        self.nodes.append(node)
        node.lines.append(self)
        self.update_path()

    def remove_node(self, node):
        if node in self.nodes:
            self.nodes.remove(node)
            node.lines.remove(self)
            self.update_path()

    def contextMenuEvent(self, event):
        menu = QMenu()
        delete_action = menu.addAction("Delete Line")
        action = menu.exec_(event.screenPos())
        if action == delete_action:
            # Remove the line and its nodes
            for node in self.nodes:
                self.scene().removeItem(node)
            self.scene().removeItem(self)
            # Remove from the parent's lines list
            if self in self.parent.lines:
                self.parent.lines.remove(self)


class Network(torch.nn.Module): # Neural Network for Hessian Edge Detection
    def __init__(self):
        super().__init__()

        self.netVggOne = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggTwo = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggThr = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggFou = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggFiv = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netScoreOne = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreTwo = torch.nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreThr = torch.nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreFou = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreFiv = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.netCombine = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0),
            torch.nn.Sigmoid()
        )
    # end

    def forward(self, tenInput):
        tenInput = tenInput * 255.0
        tenInput = tenInput - torch.tensor(
            data=[104.00698793, 116.66876762, 122.67891434],
            dtype=tenInput.dtype,
            device=tenInput.device
        ).view(1, 3, 1, 1)

        tenVggOne = self.netVggOne(tenInput)
        tenVggTwo = self.netVggTwo(tenVggOne)
        tenVggThr = self.netVggThr(tenVggTwo)
        tenVggFou = self.netVggFou(tenVggThr)
        tenVggFiv = self.netVggFiv(tenVggFou)

        tenScoreOne = self.netScoreOne(tenVggOne)
        tenScoreTwo = self.netScoreTwo(tenVggTwo)
        tenScoreThr = self.netScoreThr(tenVggThr)
        tenScoreFou = self.netScoreFou(tenVggFou)
        tenScoreFiv = self.netScoreFiv(tenVggFiv)

        tenScoreOne = torch.nn.functional.interpolate(
            input=tenScoreOne,
            size=(tenInput.shape[2], tenInput.shape[3]),
            mode='bilinear',
            align_corners=False
        )
        tenScoreTwo = torch.nn.functional.interpolate(
            input=tenScoreTwo,
            size=(tenInput.shape[2], tenInput.shape[3]),
            mode='bilinear',
            align_corners=False
        )
        tenScoreThr = torch.nn.functional.interpolate(
            input=tenScoreThr,
            size=(tenInput.shape[2], tenInput.shape[3]),
            mode='bilinear',
            align_corners=False
        )
        tenScoreFou = torch.nn.functional.interpolate(
            input=tenScoreFou,
            size=(tenInput.shape[2], tenInput.shape[3]),
            mode='bilinear',
            align_corners=False
        )
        tenScoreFiv = torch.nn.functional.interpolate(
            input=tenScoreFiv,
            size=(tenInput.shape[2], tenInput.shape[3]),
            mode='bilinear',
            align_corners=False
        )

        return self.netCombine(torch.cat([
            tenScoreOne,
            tenScoreTwo,
            tenScoreThr,
            tenScoreFou,
            tenScoreFiv
        ], 1))
    # end
    def forward_side(self, tenInput):
        tenInput = tenInput * 255.0
        tenInput = tenInput - torch.tensor(
            data=[104.00698793, 116.66876762, 122.67891434],
            dtype=tenInput.dtype,
            device=tenInput.device
        ).view(1, 3, 1, 1)

        tenVggOne = self.netVggOne(tenInput)
        tenVggTwo = self.netVggTwo(tenVggOne)
        tenVggThr = self.netVggThr(tenVggTwo)
        tenVggFou = self.netVggFou(tenVggThr)
        tenVggFiv = self.netVggFiv(tenVggFou)

        tenScoreOne = self.netScoreOne(tenVggOne)
        tenScoreTwo = self.netScoreTwo(tenVggTwo)
        tenScoreThr = self.netScoreThr(tenVggThr)
        tenScoreFou = self.netScoreFou(tenVggFou)
        tenScoreFiv = self.netScoreFiv(tenVggFiv)

        tenScoreOne = torch.nn.functional.interpolate(
            input=tenScoreOne,
            size=(tenInput.shape[2], tenInput.shape[3]),
            mode='bilinear',
            align_corners=False
        )
        tenScoreTwo = torch.nn.functional.interpolate(
            input=tenScoreTwo,
            size=(tenInput.shape[2], tenInput.shape[3]),
            mode='bilinear',
            align_corners=False
        )
        tenScoreThr = torch.nn.functional.interpolate(
            input=tenScoreThr,
            size=(tenInput.shape[2], tenInput.shape[3]),
            mode='bilinear',
            align_corners=False
        )
        tenScoreFou = torch.nn.functional.interpolate(
            input=tenScoreFou,
            size=(tenInput.shape[2], tenInput.shape[3]),
            mode='bilinear',
            align_corners=False
        )
        tenScoreFiv = torch.nn.functional.interpolate(
            input=tenScoreFiv,
            size=(tenInput.shape[2], tenInput.shape[3]),
            mode='bilinear',
            align_corners=False
        )

        return [tenScoreOne, tenScoreTwo, tenScoreThr, tenScoreFou, tenScoreFiv]


def parse_path(path_data):
    commands = re.findall(r'([MLHVCSQTAZ])([^MLHVCSQTAZ]*)', path_data.upper())
    points = []
    current = [0, 0]
    for cmd, params in commands:
        params = [float(p) for p in params.strip().split()]
        if cmd == 'M':
            points.extend(params)
            current = params[-2:]
        elif cmd == 'L':
            points.extend(params)
            current = params[-2:]
        elif cmd == 'H':
            for x in params:
                points.extend([x, current[1]])
                current[0] = x
        elif cmd == 'V':
            for y in params:
                points.extend([current[0], y])
                current[1] = y
        elif cmd == 'Z':
            if points:
                points.extend(points[:2])
        # Note: We're ignoring curve commands (C, S, Q, T, A) for simplicity
    return points
def debug_print(message):
        print(f"DEBUG: {message}")

def euclidean_dist(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def a_star(costs, start, end):
    rows, cols = costs.shape
    visited = np.zeros_like(costs, dtype=bool)
    g_cost = np.full_like(costs, np.inf)  # Cost from start to current node
    f_cost = np.full_like(costs, np.inf)  # Total cost of node (g + heuristic)
    g_cost[start] = 0
    f_cost[start] = euclidean_dist(start, end)
    predecessors = np.full(costs.shape + (2,), -1, dtype=int)
    pq = [(f_cost[start], start)]  # Priority queue storing (f_cost, node)

    while pq:
        _, current = heapq.heappop(pq)
        if current == end:
            break
        if visited[current]:
            continue
        visited[current] = True

        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Neighboring nodes
            ny, nx = current[0] + dy, current[1] + dx
            if 0 <= ny < rows and 0 <= nx < cols and not visited[ny, nx]:
                temp_g_cost = g_cost[current] + costs[ny, nx]
                if temp_g_cost < g_cost[ny, nx]:
                    g_cost[ny, nx] = temp_g_cost
                    f_cost[ny, nx] = temp_g_cost + euclidean_dist((ny, nx), end)
                    predecessors[ny, nx] = current
                    heapq.heappush(pq, (f_cost[ny, nx], (ny, nx)))

    path = []
    node = end
    while node != start and predecessors[node[0], node[1]][0] != -1:
        path.append(node)
        node = tuple(predecessors[node[0], node[1]])
    path.append(start)
    path.reverse()
    return path
#

class FilterTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.input_image = None
        self.filtered_image = None
        self.edge_linked_image = None
        self.edge_link_window = None  # Add this line
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        image_layout = QHBoxLayout()

        # Input image
        self.input_figure = Figure(figsize=(5, 5))
        self.input_canvas = FigureCanvas(self.input_figure)

        # Filtered image
        self.filtered_figure = Figure(figsize=(5, 5))
        self.filtered_canvas = FigureCanvas(self.filtered_figure)

        image_layout.addWidget(self.input_canvas)
        image_layout.addWidget(self.filtered_canvas)

        layout.addLayout(image_layout)

        # Gaussian filter controls
        gaussian_layout = QHBoxLayout()
        self.gaussian_checkbox = QCheckBox("Apply Gaussian Filter")
        self.gaussian_checkbox.stateChanged.connect(self.update_filter)
        gaussian_layout.addWidget(self.gaussian_checkbox)

        self.gaussian_sigma = QSlider(Qt.Horizontal)
        self.gaussian_sigma.setRange(1, 50)  # Sigma from 0.1 to 5.0
        self.gaussian_sigma.setValue(10)  # Default value 1.0
        self.gaussian_sigma.valueChanged.connect(self.update_filter)
        gaussian_layout.addWidget(self.gaussian_sigma)

        self.gaussian_sigma_label = QLabel("1.0")
        gaussian_layout.addWidget(self.gaussian_sigma_label)

        layout.addLayout(gaussian_layout)

        # Controls layout will be added by subclasses
        self.controls_layout = QVBoxLayout()
        layout.addLayout(self.controls_layout)

        # Skeletonization checkbox
        self.skeletonize_checkbox = QCheckBox("Apply Skeletonization")
        self.skeletonize_checkbox.stateChanged.connect(self.update_filter)
        layout.addWidget(self.skeletonize_checkbox)

        # Edge Link button
        self.edge_link_button = QPushButton("Edge Link")
        self.edge_link_button.clicked.connect(self.open_edge_link_window)
        layout.addWidget(self.edge_link_button)

        self.setLayout(layout)

        # Call the method to create filter-specific controls
        self.create_filter_controls()
    def open_edge_link_window(self):
        if self.filtered_image is None:
            QMessageBox.warning(self, "Error", "Apply a filter first")
            return
            
        # Create new window or show existing
        if not self.edge_link_window:
            self.edge_link_window = EdgeLinkWindow(self.filtered_image, self)
            # Connect signal to update filtered image
            self.edge_link_window.edge_link_updated.connect(self.update_from_edge_link)
        
        self.edge_link_window.show()

    def update_from_edge_link(self, new_image):
        """Update the filtered image from edge link results"""
        self.filtered_image = new_image
        self.show_filtered_image()
    def create_filter_controls(self):
        """
        Method to be overridden by subclasses to add their specific controls.
        """
        pass

    def set_input_image(self, image):
        self.input_image = image
        self.show_input_image()
        self.update_filter()

    def show_input_image(self):
        self.input_figure.clear()
        ax = self.input_figure.add_subplot(111)
        ax.imshow(self.input_image, cmap='gray')
        ax.axis('off')
        self.input_canvas.draw()

    def show_filtered_image(self):
        self.filtered_figure.clear()
        ax = self.filtered_figure.add_subplot(111)
        ax.imshow(self.filtered_image, cmap='gray')
        ax.axis('off')
        self.filtered_canvas.draw()

    def update_filter(self):
        if self.input_image is not None:
            # Apply Gaussian filter if checkbox is checked
            if self.gaussian_checkbox.isChecked():
                sigma = self.gaussian_sigma.value() / 10.0  # Convert to float
                self.gaussian_sigma_label.setText(f"{sigma:.1f}")
                blurred_image = cv2.GaussianBlur(self.input_image, (0, 0), sigma)
            else:
                blurred_image = self.input_image

            # Apply the specific filter (to be implemented in subclasses)
            self.apply_filter(blurred_image)

            # Apply skeletonization if checkbox is checked
            if self.skeletonize_checkbox.isChecked():
                self.filtered_image = self.apply_skeletonization(self.filtered_image)

            self.show_filtered_image()

    def apply_skeletonization(self, image):
        return skeletonize(image > 0).astype(np.uint8) * 255

    def open_edge_link_window(self):
        if self.filtered_image is None:
            QMessageBox.warning(self, "Error", "Apply a filter first")
            return
        
        # Create new window if not exists or show existing
        if self.edge_link_window is None:
            self.edge_link_window = EdgeLinkWindow(self.filtered_image, self)
            self.edge_link_window.edge_link_updated.connect(self.update_from_edge_link)
        
        self.edge_link_window.show()
        self.edge_link_window.process_and_display()  # Force initial update

    def apply_filter(self, image):
        # To be implemented in subclasses
        pass


class ManualInterpretationWindow(QDialog):
    def __init__(self, original_image, filtered_image, parent=None):
        super().__init__(parent)

        # Validate input images
        self.validate_image(original_image)
        self.validate_image(filtered_image)

        # Initialize images
        self.original_image = original_image
        self.filtered_image = filtered_image
        self.display_image = original_image.copy()  # Display the original image

        # Initialize drawing attributes
        self.lines = []
        self.current_line = []
        self.is_drawing = False
        self.is_semi_auto = False
        self.is_edit_mode = False
        self.semi_auto_start_point = None

        # Initialize Undo/Redo stacks
        self.undo_stack = deque()
        self.redo_stack = deque()

        # Initialize UI
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Manual Interpretation')
        self.setGeometry(100, 100, 1200, 800)  # Adjusted size for better layout

        # Main layout
        layout = QVBoxLayout()

        # Create QGraphicsScene and QGraphicsView
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setDragMode(QGraphicsView.NoDrag)  # Ensure no interference with item interactions
        layout.addWidget(self.view)

        # Display the original image
        self.show_original_image()

        # Overlay filtered lines as LineItems
        self.display_filtered_lines()

        # Fit the view to the scene
        self.view.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)

        # Control Buttons Layout
        button_layout = QHBoxLayout()

        # Existing buttons
        self.toggle_drawing_button = QPushButton("Enable Manual Drawing")
        self.toggle_drawing_button.setCheckable(True)
        self.toggle_drawing_button.clicked.connect(self.toggle_drawing)
        button_layout.addWidget(self.toggle_drawing_button)

        self.toggle_semi_auto_button = QPushButton("Enable Semi-Auto Drawing")
        self.toggle_semi_auto_button.setCheckable(True)
        self.toggle_semi_auto_button.clicked.connect(self.toggle_semi_auto)
        button_layout.addWidget(self.toggle_semi_auto_button)

        self.edit_mode_button = QPushButton("Enter Edit Mode")
        self.edit_mode_button.setCheckable(True)
        self.edit_mode_button.clicked.connect(self.toggle_edit_mode)
        button_layout.addWidget(self.edit_mode_button)

        # **New Button: Enable/Disable Edgelink**
        self.toggle_edgelink_button = QPushButton("Disable Edgelink")
        self.toggle_edgelink_button.setCheckable(True)
        self.toggle_edgelink_button.setChecked(False)  # Initially, edgelink is enabled
        self.toggle_edgelink_button.clicked.connect(self.toggle_edgelink)
        button_layout.addWidget(self.toggle_edgelink_button)

        # **Existing Button: Show/Hide Nodes**
        self.toggle_nodes_button = QPushButton("Hide Nodes")
        self.toggle_nodes_button.setCheckable(True)
        self.toggle_nodes_button.setChecked(True)  # Initially, nodes are visible
        self.toggle_nodes_button.clicked.connect(self.toggle_nodes_visibility)
        button_layout.addWidget(self.toggle_nodes_button)

        # **Optional Enhancement: Undo and Redo Buttons**
        undo_redo_layout = QHBoxLayout()
        self.undo_button = QPushButton("Undo")
        self.undo_button.clicked.connect(self.undo_action)
        self.redo_button = QPushButton("Redo")
        self.redo_button.clicked.connect(self.redo_action)
        undo_redo_layout.addWidget(self.undo_button)
        undo_redo_layout.addWidget(self.redo_button)
        layout.addLayout(undo_redo_layout)

        layout.addLayout(button_layout)

        self.setLayout(layout)

        # Connect mouse events
        self.view.viewport().installEventFilter(self)

    def show_original_image(self):
        # Convert the original image to QImage
        height, width = self.display_image.shape
        bytes_per_line = width
        q_image = QImage(self.display_image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_image)

        # Add the image to the scene
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.pixmap_item.setZValue(0)  # Ensure it's at the back
        self.scene.addItem(self.pixmap_item)

        # Ensure the scene's origin matches the image's origin
        self.scene.setSceneRect(0, 0, width, height)

    def display_filtered_lines(self):
        # Clear existing lines from the scene
        for line in self.lines:
            for node in line.nodes:
                self.scene.removeItem(node)
            self.scene.removeItem(line)
        self.lines.clear()

        # Use the Edgelink class for edge linking
        minlength = 3
        edge_linker = edgelink(self.filtered_image, minlength)
        edge_linker.get_edgelist()
        edge_lists = [np.array(edge) for edge in edge_linker.edgelist if len(edge) > 0]

        # Optionally post-process edge lists (e.g., using cleanedgelist)
        processed_edge_lists = cleanedgelist(edge_lists, minlength)  # Adjust min_length as needed

        self.edge_map = self.filtered_image  # Or assign appropriately based on your Edgelink implementation

        # Add lines to the scene
        for edge in processed_edge_lists:
            if len(edge) >= 2:
                # Simplify the edge if necessary
                epsilon = 0.05 * cv2.arcLength(edge, True)
                approx = cv2.approxPolyDP(edge, epsilon, True)
                points = [(point[0][1], point[0][0]) for point in approx]

                sampled_points = points  # Adjust sampling rate if needed

                if len(sampled_points) >= 2:
                    self.add_line(sampled_points)

    def add_line(self, points):
        line = LineItem(parent=self)
        line.setZValue(1)
        line.setFlags(QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsMovable)
        self.scene.addItem(line)
        self.lines.append(line)


        for x, y in points:
            node = NodeItem(x, y, radius=2)  # Increased radius for better visibility
            node.setZValue(2)  # Ensure nodes are above lines
            self.scene.addItem(node)
            line.add_node(node)

        # Record action for undo
        self.undo_stack.append(('add_line', line))
        self.redo_stack.clear()

    def toggle_drawing(self, checked):
        self.is_drawing = checked
        self.is_semi_auto = False
        self.is_edit_mode = False
        self.semi_auto_start_point = None
        self.toggle_drawing_button.setText("Disable Manual Drawing" if self.is_drawing else "Enable Manual Drawing")
        self.toggle_semi_auto_button.setText("Enable Semi-Auto Drawing")
        self.edit_mode_button.setText("Enter Edit Mode")
        self.toggle_edgelink_button.setChecked(True)
        self.toggle_edgelink_button.setText("Disable Edgelink")
        self.toggle_nodes_button.setChecked(True)
        self.toggle_nodes_button.setText("Hide Nodes")
        self.show_overlay()

    def toggle_semi_auto(self, checked):
        self.is_semi_auto = checked
        self.is_drawing = False
        self.is_edit_mode = False
        self.semi_auto_start_point = None
        self.toggle_semi_auto_button.setText(
            "Disable Semi-Auto Drawing" if self.is_semi_auto else "Enable Semi-Auto Drawing")
        self.toggle_drawing_button.setText("Enable Manual Drawing")
        self.edit_mode_button.setText("Enter Edit Mode")
        self.toggle_edgelink_button.setChecked(True)
        self.toggle_edgelink_button.setText("Disable Edgelink")
        self.toggle_nodes_button.setChecked(True)
        self.toggle_nodes_button.setText("Hide Nodes")
        self.show_overlay()

    def toggle_edit_mode(self, checked):
        self.is_edit_mode = checked
        self.is_drawing = False
        self.is_semi_auto = False
        self.semi_auto_start_point = None
        self.edit_mode_button.setText("Exit Edit Mode" if self.is_edit_mode else "Enter Edit Mode")

        if self.is_edit_mode:
            QMessageBox.information(self, "Edit Mode",
                                    "Edit Mode Enabled.\nSelect and drag nodes to edit the lines.")
        self.show_overlay()

    # **New Method: Toggle Edgelink**
    def toggle_edgelink(self, checked):
        self.use_edgelink = not checked  # Toggle the state
        self.toggle_edgelink_button.setText("Enable Edgelink" if not self.use_edgelink else "Disable Edgelink")
        self.process_and_display_lines()

    # **Existing Method: Toggle Nodes Visibility**
    def toggle_nodes_visibility(self, checked):
        # If checked, show nodes; else, hide them
        if checked:
            self.toggle_nodes_button.setText("Hide Nodes")
        else:
            self.toggle_nodes_button.setText("Show Nodes")

        for line in self.lines:
            for node in line.nodes:
                node.setVisible(checked)

    def image_to_lines(self, binary_image):
        """
        Converts a binary image into a list of lines using contour detection.

        Parameters:
            binary_image (numpy.ndarray): The binary image from which to extract lines.

        Returns:
            List[List[Tuple[int, int]]]: A list of lines, each represented as a list of (x, y) tuples.
        """
        # Find contours in the binary image
        contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        lines = []
        for contour in contours:
            # Approximate the contour to reduce the number of points
            epsilon = 0.05 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Convert contour to list of (x, y) points
            points = [(point[0][1], point[0][0]) for point in approx]
            lines.append(points)

        return lines
    def process_and_display_lines(self): # New method for processing and displaying lines
        # Clear existing lines from the scene
        for line in self.lines:
            for node in line.nodes:
                self.scene.removeItem(node)
            self.scene.removeItem(line)
        self.lines.clear()

        # Use the Edgelink class for edge linking
        if self.use_edgelink:
            minlength=3
            edge_linker = edgelink(self.filtered_image, minlength)
            edge_linker.get_edgelist()
            edge_lists = [np.array(edge) for edge in edge_linker.edgelist if len(edge) > 0]
            processed_edge_lists = cleanedgelist(edge_lists, minlength)  # Adjust min_length as needed
        else:
            # If edgelink is disabled, use the original filtered image
            _, binary_image = cv2.threshold(self.filtered_image, 127, 255, cv2.THRESH_BINARY)
            edge_lists = self.image_to_lines(binary_image)
            processed_edge_lists = [np.array(edge) for edge in edge_lists if len(edge) > 0]

        self.edge_map = self.filtered_image  # Or assign appropriately based on your Edgelink implementation

        # Add lines to the scene
        for edge in processed_edge_lists:
            if len(edge) >= 2:
                # Simplify the edge if necessary
                epsilon = 0.05 * cv2.arcLength(edge, True)
                approx = cv2.approxPolyDP(edge, epsilon, True)
                points = [(point[0][1], point[0][0]) for point in approx]

                sampled_points = points  # Adjust sampling rate if needed

                if len(sampled_points) >= 2:
                    self.add_line(sampled_points)

    # **Optional Enhancement: Undo and Redo Methods**
    def undo_action(self):
        if not self.undo_stack:
            return
        action, obj = self.undo_stack.pop()
        if action == 'add_line':
            # Remove the line
            for node in obj.nodes.copy():
                self.scene.removeItem(node)
            self.scene.removeItem(obj)
            self.lines.remove(obj)
            # Push to redo stack
            self.redo_stack.append(('add_line', obj))
        elif action == 'add_node':
            # Remove the node
            self.scene.removeItem(obj)
            obj.lines[0].nodes.remove(obj)
            # Push to redo stack
            self.redo_stack.append(('add_node', obj))
        elif action == 'delete_node':
            # Re-add the node and reconnect lines
            self.scene.addItem(obj)
            for line in obj.lines:
                line.add_node(obj)
            self.lines.append(line)
            # Push to redo stack
            self.redo_stack.append(('delete_node', obj))
        elif action == 'delete_line':
            # Re-add the line and reconnect nodes
            self.scene.addItem(obj)
            for node in obj.nodes:
                self.scene.addItem(node)
                node.lines.append(obj)
            self.lines.append(obj)
            # Push to redo stack
            self.redo_stack.append(('delete_line', obj))
        self.show_overlay()

    def redo_action(self):
        if not self.redo_stack:
            return
        action, obj = self.redo_stack.pop()
        if action == 'add_line':
            # Re-add the line
            self.scene.addItem(obj)
            self.lines.append(obj)
            for node in obj.nodes:
                self.scene.addItem(node)
                node.lines.append(obj)
            # Push back to undo stack
            self.undo_stack.append(('add_line', obj))
        elif action == 'add_node':
            # Re-add the node
            self.scene.addItem(obj)
            obj.lines[0].add_node(obj)
            # Push back to undo stack
            self.undo_stack.append(('add_node', obj))
        elif action == 'delete_node':
            # Remove the node again
            for line in obj.lines.copy():
                line.remove_node(obj)
                if len(line.nodes) < 2:
                    self.delete_line(line)
            self.scene.removeItem(obj)
            # Push back to undo stack
            self.undo_stack.append(('delete_node', obj))
        elif action == 'delete_line':
            # Remove the line again
            for node in obj.nodes.copy():
                obj.remove_node(node)
                self.scene.removeItem(node)
            self.scene.removeItem(obj)
            self.lines.remove(obj)
            # Push back to undo stack
            self.undo_stack.append(('delete_line', obj))
        self.show_overlay()

    def show_overlay(self):
        # Update the scene to reflect edit mode
        for line in self.lines:
            line.setFlags(QGraphicsItem.ItemIsSelectable)  # Lines are selectable but not movable

        for line in self.lines:
            for node in line.nodes:
                if self.is_edit_mode:
                    node.setFlags(
                        QGraphicsItem.ItemIsMovable |
                        QGraphicsItem.ItemSendsGeometryChanges |
                        QGraphicsItem.ItemIsSelectable
                    )
                else:
                    node.setFlags(QGraphicsItem.ItemIsSelectable)

        # Highlight selected items
        for line in self.lines:
            if line.isSelected():
                line.setPen(QPen(QColor('green'), 3))  # Increased width for better visibility
            else:
                line.setPen(QPen(QColor('green'), 3))  # Keep consistent pen settings

            for node in line.nodes:
                if node.isSelected():
                    node.setBrush(QColor('yellow'))
                else:
                    node.setBrush(QColor('blue'))
        # Refresh the scene to apply changes
        self.scene.update()

    def eventFilter(self, source, event):
        if event.type() == event.MouseButtonPress:
            if self.is_drawing and event.button() == Qt.LeftButton:
                scene_pos = self.view.mapToScene(event.pos())
                x, y = int(scene_pos.x()), int(scene_pos.y())
                self.current_line.append((x, y))
                self.add_manual_line(x, y)
            elif self.is_semi_auto and event.button() == Qt.LeftButton:
                scene_pos = self.view.mapToScene(event.pos())
                x, y = int(scene_pos.x()), int(scene_pos.y())
                if not self.semi_auto_start_point:
                    self.semi_auto_start_point = (x, y)
                    QMessageBox.information(self, "Semi-Auto Drawing",
                                            "Start point set. Click again to set end point.")
                else:
                    end_point = (x, y)
                    self.semi_automatic_tracking(self.semi_auto_start_point, end_point)
                    self.semi_auto_start_point = None
            return True  # Event handled
        return super().eventFilter(source, event)

    def add_manual_line(self, x, y):
        if len(self.current_line) == 1:
            # Create a new line
            self.current_line_item = LineItem()
            self.current_line_item.setZValue(1)  # Ensure lines are above the background
            self.scene.addItem(self.current_line_item)
            self.lines.append(self.current_line_item)

            # Create a node
            node = NodeItem(x, y, radius=5)  # Increased radius for better visibility
            node.setZValue(2)  # Ensure nodes are above lines
            self.scene.addItem(node)
            self.current_line_item.add_node(node)

            # Record action for undo
            self.undo_stack.append(('add_line', self.current_line_item))
            self.redo_stack.clear()
        else:
            # Add a node and update the line
            node = NodeItem(x, y, radius=5)
            node.setZValue(2)
            self.scene.addItem(node)
            self.current_line_item.add_node(node)

            # Record action for undo
            self.undo_stack.append(('add_node', node))
            self.redo_stack.clear()

    def semi_automatic_tracking(self, start_point, end_point):
        # Use A* algorithm to find a path from start_point to end_point on the edge_map
        path = self.a_star(self.edge_map, start_point, end_point)

        if path:
            # Convert path to list of (x, y) tuples
            tracked_line = [(point[1], point[0]) for point in path]  # Swap to (x, y)
            self.add_line(tracked_line)
            # Fit view to include the new line
            self.view.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)
        else:
            QMessageBox.warning(self, "Path Not Found",
                                "Could not find a path between the selected points. "
                                "Try selecting closer points or adjusting the filter settings.")

    def a_star(self, cost_map, start, goal):
        def heuristic(a, b):
            return np.hypot(b[0] - a[0], b[1] - a[1])

        neighbors = [(-1, -1), (-1, 0), (-1, 1),
                     (0, -1),         (0, 1),
                     (1, -1),  (1, 0),  (1, 1)]

        close_set = set()
        came_from = {}
        gscore = {start: 0}
        fscore = {start: heuristic(start, goal)}
        oheap = []
        heapq.heappush(oheap, (fscore[start], start))

        while oheap:
            current = heapq.heappop(oheap)[1]

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path

            close_set.add(current)

            for i, j in neighbors:
                neighbor = current[0] + i, current[1] + j
                if (0 <= neighbor[0] < cost_map.shape[0] and 0 <= neighbor[1] < cost_map.shape[1]):
                    tentative_g_score = gscore[current] + cost_map[neighbor]
                    if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, float('inf')):
                        continue
                    if tentative_g_score < gscore.get(neighbor, float('inf')):
                        came_from[neighbor] = current
                        gscore[neighbor] = tentative_g_score
                        fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                        heapq.heappush(oheap, (fscore[neighbor], neighbor))

        return None

    def edit_nearest_point(self, x, y):
        min_distance = float('inf')
        nearest_line = None
        nearest_point_index = None

        for line in self.lines:
            if isinstance(line, LineItem):
                for j, node in enumerate(line.nodes):
                    distance = np.sqrt((x - node.pos().x()) ** 2 + (y - node.pos().y()) ** 2)
                    if distance < min_distance:
                        min_distance = distance
                        nearest_line = line
                        nearest_point_index = j

        if nearest_line is not None and min_distance < 10:  # Threshold for selection
            node = nearest_line.nodes[nearest_point_index]
            node.setSelected(True)
            nearest_line.setSelected(True)
            self.show_overlay()

    def delete_node(self, node):
        for line in node.lines.copy():
            line.remove_node(node)
            if len(line.nodes) < 2:
                self.delete_line(line)
        self.scene.removeItem(node)
        self.show_overlay()

        # Record action for undo
        self.undo_stack.append(('delete_node', node))
        self.redo_stack.clear()

    def delete_line(self, line):
        for node in line.nodes:
            if line in node.lines:
                node.lines.remove(line)
        if line in self.lines:
            self.lines.remove(line)
        self.scene.removeItem(line)
        self.show_overlay()

        # Record action for undo
        self.undo_stack.append(('delete_line', line))
        self.redo_stack.clear()

    # Optional: Implement zoom functionality
    def wheelEvent(self, event):
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor

        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor

        self.view.scale(zoom_factor, zoom_factor)

    def contextMenuEvent(self, event):
        # This method can be removed or kept empty to prevent the dialog from showing its own context menu
        pass

    def validate_image(self, image):
        if image is None:
            raise ValueError("Invalid image. Please load a valid grayscale image.")
        if len(image.shape) != 2:
            raise ValueError("Image is not grayscale. Please load a grayscale image.")

    # **Optional Enhancement: Undo and Redo Methods**
    def undo_action(self):
        if not self.undo_stack:
            return
        action, obj = self.undo_stack.pop()
        if action == 'add_line':
            # Remove the line
            for node in obj.nodes.copy():
                self.scene.removeItem(node)
            self.scene.removeItem(obj)
            self.lines.remove(obj)
            # Push to redo stack
            self.redo_stack.append(('add_line', obj))
        elif action == 'add_node':
            # Remove the node
            self.scene.removeItem(obj)
            obj.lines[0].nodes.remove(obj)
            # Push to redo stack
            self.redo_stack.append(('add_node', obj))
        elif action == 'delete_node':
            # Re-add the node and reconnect lines
            self.scene.addItem(obj)
            for line in obj.lines:
                line.add_node(obj)
            self.lines.append(line)
            # Push to redo stack
            self.redo_stack.append(('delete_node', obj))
        elif action == 'delete_line':
            # Re-add the line and reconnect nodes
            self.scene.addItem(obj)
            for node in obj.nodes:
                self.scene.addItem(node)
                node.lines.append(obj)
            self.lines.append(obj)
            # Push to redo stack
            self.redo_stack.append(('delete_line', obj))
        self.show_overlay()

    def redo_action(self):
        if not self.redo_stack:
            return
        action, obj = self.redo_stack.pop()
        if action == 'add_line':
            # Re-add the line
            self.scene.addItem(obj)
            self.lines.append(obj)
            for node in obj.nodes:
                self.scene.addItem(node)
                node.lines.append(obj)
            # Push back to undo stack
            self.undo_stack.append(('add_line', obj))
        elif action == 'add_node':
            # Re-add the node
            self.scene.addItem(obj)
            obj.lines[0].add_node(obj)
            # Push back to undo stack
            self.undo_stack.append(('add_node', obj))
        elif action == 'delete_node':
            # Remove the node again
            for line in obj.lines.copy():
                line.remove_node(obj)
                if len(line.nodes) < 2:
                    self.delete_line(line)
            self.scene.removeItem(obj)
            # Push back to undo stack
            self.undo_stack.append(('delete_node', obj))
        elif action == 'delete_line':
            # Remove the line again
            for node in obj.nodes.copy():
                obj.remove_node(node)
                self.scene.removeItem(node)
            self.scene.removeItem(obj)
            self.lines.remove(obj)
            # Push back to undo stack
            self.undo_stack.append(('delete_line', obj))
        self.show_overlay()

class HEDFilterTab(FilterTab):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("HED Filter")
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.available_models = ['bsds500', 'pascal']  # Add other models if available
        self.selected_model = 'bsds500'  # Default model
        self.current_model = None  # Keep track of the current model loaded
        self.create_controls()
        self.init_model()

    def create_controls(self):
        # Add model selection combo box
        model_layout = QHBoxLayout()
        model_label = QLabel("Model:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(self.available_models)
        self.model_combo.setCurrentText(self.selected_model)
        self.model_combo.currentIndexChanged.connect(self.update_model)
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        self.controls_layout.addLayout(model_layout)

        # Add side output selection
        side_output_layout = QHBoxLayout()
        side_output_label = QLabel("Side Output:")
        self.side_output_combo = QComboBox()
        self.side_output_combo.addItems(['Combined', '1', '2', '3', '4', '5'])
        self.side_output_combo.setCurrentText('Combined')
        self.side_output_combo.currentIndexChanged.connect(self.on_side_output_changed)
        side_output_layout.addWidget(side_output_label)
        side_output_layout.addWidget(self.side_output_combo)
        self.controls_layout.addLayout(side_output_layout)

        # Add threshold slider
        self.threshold = 50  # Default threshold value
        self.create_threshold_slider()

    def on_side_output_changed(self, index):
        # index is the new index of the combo box, but we don't need it here
        self.apply_filter()
    def create_threshold_slider(self):
        threshold_layout = QHBoxLayout()
        threshold_label = QLabel("Threshold:")
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 255)
        self.threshold_slider.setValue(self.threshold)
        self.threshold_slider.valueChanged.connect(self.update_threshold)
        self.threshold_value_label = QLabel(str(self.threshold))
        threshold_layout.addWidget(threshold_label)
        threshold_layout.addWidget(self.threshold_slider)
        threshold_layout.addWidget(self.threshold_value_label)
        self.controls_layout.addLayout(threshold_layout)

    def update_threshold(self, value):
        self.threshold = value
        self.threshold_value_label.setText(str(value))
        self.apply_filter()

    def update_model(self):
        self.selected_model = self.model_combo.currentText()
        self.init_model()
        self.apply_filter()

    def init_model(self):
        if self.model is None or self.current_model != self.selected_model:
            try:
                self.current_model = self.selected_model
                QMessageBox.information(
                    self, "Loading Model",
                    f"Loading HED model '{self.selected_model}'. Please wait..."
                )
                self.model = Network().to(self.device).eval()
                # Load pre-trained weights
                model_url = f'http://content.sniklaus.com/github/pytorch-hed/network-{self.selected_model}.pytorch'
                state_dict = torch.hub.load_state_dict_from_url(
                    url=model_url,
                    file_name=f'hed-{self.selected_model}'
                )
                state_dict = {
                    strKey.replace('module', 'net'): tenWeight
                    for strKey, tenWeight in state_dict.items()
                }
                self.model.load_state_dict(state_dict)
            except Exception as e:
                QMessageBox.critical(
                    self, "Error",
                    f"Failed to load HED model '{self.selected_model}': {str(e)}"
                )
                self.model = None

    def apply_filter(self, image=None, *args, **kwargs):
        if self.input_image is None or self.model is None:
            print("Input image or model is None, cannot apply filter")
            return

        if image is None:
            image = self.input_image

        if image is None or not hasattr(image, 'shape') or len(image.shape) < 2:
            print("Invalid image provided to apply_filter")
            return

        height, width = image.shape[:2]
        print(f"Original image dimensions - width: {width}, height: {height}")
        if height == 0 or width == 0:
            print("Image has zero dimensions, cannot proceed")
            return

        # Use the blurred image if Gaussian filter is applied
        if self.gaussian_checkbox.isChecked():
            sigma = self.gaussian_sigma.value() / 10.0
            blurred_image = cv2.GaussianBlur(image, (0, 0), sigma)
        else:
            blurred_image = image

        # Convert grayscale to RGB
        image_rgb = cv2.cvtColor(blurred_image, cv2.COLOR_GRAY2RGB)

        # Resize the image to 1024x1024 as expected by the model
        model_input_size = (1024, 1024)
        image_rgb_resized = cv2.resize(image_rgb, model_input_size)

        # Convert to tensor
        np_image = image_rgb_resized[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)
        tenInput = torch.FloatTensor(np.ascontiguousarray(np_image))
        tenInput = tenInput.to(self.device).unsqueeze(0)

        # Run the model
        with torch.no_grad():
            tenInput = tenInput * 255.0
            tenInput = tenInput - torch.tensor(
                data=[104.00698793, 116.66876762, 122.67891434],
                dtype=tenInput.dtype,
                device=tenInput.device
            ).view(1, 3, 1, 1)

            # Inside apply_filter
            if self.side_output_combo.currentText() == 'Combined':
                output = self.model(tenInput)[0].cpu().numpy()
                print(f"Combined output shape: {output.shape}")
            else:
                side_outputs = self.model.forward_side(tenInput)
                side_index = int(self.side_output_combo.currentText()) - 1
                output_tensor = side_outputs[side_index]
                print(f"Side output {side_index + 1} tensor shape: {output_tensor.shape}")
                output = output_tensor.cpu().numpy()[0, 0, :, :]
                print(f"Extracted output shape: {output.shape}")
        # print(f"Side output {side_index + 1} tensor shape: {output_tensor.shape}")
        # print(f"Extracted output shape: {output.shape}")
        # Post-process the output
        edge_prob_map = np.clip(output, 0.0, 1.0)
        # print(f"edge_prob_map.shape: {edge_prob_map.shape}")

        # Apply threshold
        threshold_normalized = self.threshold / 255.0
        edge_binary = (edge_prob_map >= threshold_normalized).astype(np.uint8) * 255
        # print(f"edge_binary.shape: {edge_binary.shape}")

        # Remove extra dimensions
        edge_binary = np.squeeze(edge_binary)
        # print(f"edge_binary.shape after squeeze: {edge_binary.shape}")

        # Resize back to original image size
        # print(f"Resizing edge map to width: {width}, height: {height}")
        edge_map_resized = cv2.resize(
            edge_binary,
            (width, height),  # Ensure correct order
            interpolation=cv2.INTER_NEAREST
        )

        self.filtered_image = edge_map_resized
        self.show_filtered_image()


class LaplacianFilterTab(FilterTab):
    def __init__(self, parent=None):
        # Initialize subclass-specific attributes **before** calling the base class
        self.laplacian_kernel_size = 3  # Default kernel size
        super().__init__(parent)
        self.setWindowTitle("Laplacian Filter")
        # Do **not** call self.initUI() here; it's already called in the base class

    def create_filter_controls(self):
        # Add Kernel Size slider to controls_layout
        kernel_layout = QHBoxLayout()

        kernel_label = QLabel("Kernel Size:")
        self.kernel_slider = QSlider(Qt.Horizontal)
        self.kernel_slider.setRange(1, 7)  # Typically odd kernel sizes (1, 3, 5, 7)
        self.kernel_slider.setSingleStep(2)
        self.kernel_slider.setValue(self.laplacian_kernel_size)
        self.kernel_slider.setTickInterval(2)
        self.kernel_slider.setTickPosition(QSlider.TicksBelow)
        self.kernel_slider.valueChanged.connect(self.update_kernel_size)

        self.kernel_value_label = QLabel(str(self.laplacian_kernel_size))

        kernel_layout.addWidget(kernel_label)
        kernel_layout.addWidget(self.kernel_slider)
        kernel_layout.addWidget(self.kernel_value_label)

        self.controls_layout.addLayout(kernel_layout)

    def update_kernel_size(self, value):
        # Ensure kernel size is odd
        if value % 2 == 0:
            value += 1
            self.kernel_slider.setValue(value)
        self.laplacian_kernel_size = value
        self.kernel_value_label.setText(str(value))
        self.apply_filter()

    def apply_filter(self, image=None):
        if self.input_image is None:
            return

        # Use the blurred image if Gaussian filter is applied
        if self.gaussian_checkbox.isChecked():
            sigma = self.gaussian_sigma.value() / 10.0  # Convert to float
            blurred_image = cv2.GaussianBlur(self.input_image, (0, 0), sigma)
        else:
            blurred_image = self.input_image.copy()

        # Apply Laplacian filter
        laplacian = cv2.Laplacian(blurred_image, cv2.CV_64F, ksize=self.laplacian_kernel_size)
        laplacian = cv2.convertScaleAbs(laplacian)

        # Convert to binary image before skeletonization
        _, binary_image = cv2.threshold(laplacian, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Apply skeletonization if enabled
        if self.skeletonize_checkbox.isChecked():
            binary_image = self.apply_skeletonization(binary_image)
            self.filtered_image = binary_image
        else:
            self.filtered_image = laplacian

        self.show_filtered_image()

class RobertsFilterTab(FilterTab):
    def __init__(self, parent=None):
        # No additional attributes needed for Roberts filter
        super().__init__(parent)
        self.setWindowTitle("Roberts Filter")

    def create_filter_controls(self):
        # Roberts Filter doesn't require additional controls
        pass

    def apply_filter(self, image=None):
        if self.input_image is None:
            return

        # Use the blurred image if Gaussian filter is applied
        if self.gaussian_checkbox.isChecked():
            sigma = self.gaussian_sigma.value() / 10.0
            blurred_image = cv2.GaussianBlur(self.input_image, (0, 0), sigma)
        else:
            blurred_image = self.input_image.copy()

        # Apply Roberts filter
        kernel_roberts_x = np.array([[1, 0],
                                    [0, -1]], dtype=np.float32)
        kernel_roberts_y = np.array([[0, 1],
                                    [-1, 0]], dtype=np.float32)
        roberts_x = cv2.filter2D(blurred_image, cv2.CV_64F, kernel_roberts_x)
        roberts_y = cv2.filter2D(blurred_image, cv2.CV_64F, kernel_roberts_y)
        roberts = np.sqrt(roberts_x**2 + roberts_y**2)
        roberts = cv2.convertScaleAbs(roberts)

        # Convert to binary image before skeletonization
        _, binary_image = cv2.threshold(roberts, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Apply skeletonization if enabled
        if self.skeletonize_checkbox.isChecked():
            binary_image = self.apply_skeletonization(binary_image)
            self.filtered_image = binary_image
        else:
            self.filtered_image = roberts

        self.show_filtered_image()


class CannyFilterTab(FilterTab):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_controls()

    def setup_controls(self):
        # Threshold sliders with adjusted ranges for stronger edges
        self.threshold1 = QSlider(Qt.Horizontal)
        self.threshold1.setRange(100, 200)  # Higher range for stronger edges
        self.threshold1.setValue(120)  # Higher default for noise reduction
        self.threshold1.valueChanged.connect(self.update_filter)

        self.threshold2 = QSlider(Qt.Horizontal)
        self.threshold2.setRange(150, 255)  # Much higher min range
        self.threshold2.setValue(180)  # Higher default for major edges
        self.threshold2.valueChanged.connect(self.update_filter)

        # Add threshold labels and controls
        threshold1_layout = QHBoxLayout()
        threshold1_layout.addWidget(QLabel("Threshold 1:"))
        threshold1_layout.addWidget(self.threshold1)
        self.threshold1_value = QLabel(str(self.threshold1.value()))
        threshold1_layout.addWidget(self.threshold1_value)
        self.controls_layout.addLayout(threshold1_layout)

        threshold2_layout = QHBoxLayout()
        threshold2_layout.addWidget(QLabel("Threshold 2:"))
        threshold2_layout.addWidget(self.threshold2)
        self.threshold2_value = QLabel(str(self.threshold2.value()))
        threshold2_layout.addWidget(self.threshold2_value)
        self.controls_layout.addLayout(threshold2_layout)

    def apply_filter(self, image):
        self.filtered_image = cv2.Canny(
            image, 
            self.threshold1.value(), 
            self.threshold2.value(),
            apertureSize=3,  # Increased aperture for more stable edges
            L2gradient=True  # Enable L2gradient for better edge magnitude
        )
        
        self.threshold1_value.setText(str(self.threshold1.value()))
        self.threshold2_value.setText(str(self.threshold2.value()))

    def open_manual_interpretation(self):
        pass

class SobelFilterTab(FilterTab):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_controls()

    def setup_controls(self):
        # Kernel size slider
        self.ksize = QSlider(Qt.Horizontal)
        self.ksize.setRange(1, 31)
        self.ksize.setValue(3)
        self.ksize.setSingleStep(2)
        self.ksize.valueChanged.connect(self.update_filter)

        self.controls_layout.addWidget(QLabel("Kernel Size"))
        self.controls_layout.addWidget(self.ksize)

        # #

    def apply_filter(self, image):
        ksize = self.ksize.value()
        if ksize % 2 == 0:
            ksize += 1
            
        # Calculate Sobel gradients
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
        
        # Calculate gradient magnitude
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize to 0-255 range
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Apply threshold to get binary image
        _, binary = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY)
        
        self.filtered_image = binary


    def open_manual_interpretation(self):
        pass

class ShearletFilterTab(FilterTab):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.shearlet_system = None
        self.setup_controls()

    def setup_controls(self):
        # Min contrast slider
        self.min_contrast = QSlider(Qt.Horizontal)
        self.min_contrast.setRange(0, 100)
        self.min_contrast.setValue(10)
        self.min_contrast.valueChanged.connect(self.update_filter)

        self.controls_layout.addWidget(QLabel("Min Contrast"))
        self.controls_layout.addWidget(self.min_contrast)

        ##

    def set_input_image(self, image):
        super().set_input_image(image)
        self.shearlet_system = EdgeSystem(*image.shape)

    def apply_filter(self, image):
        if self.shearlet_system is not None:
            edges, _ = self.shearlet_system.detect(image, min_contrast=self.min_contrast.value())
            self.filtered_image = (edges * 255).astype(np.uint8)
        else:
            self.filtered_image = image.copy()

    def open_manual_interpretation(self):
        pass


class EdgeLinkWindow(QDialog):
    edge_link_updated = pyqtSignal(np.ndarray)
    
    def __init__(self, image, parent=None):
        super().__init__(parent)
        self.image = image
        self.edge_linked_image = None
        self.parent = parent
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Edge Link Visualization')
        self.setGeometry(100, 100, 800, 800)

        layout = QVBoxLayout()

        # Create matplotlib figure
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Create sliders
        self.min_length_slider = self.create_slider("Minimum Edge Length", 1, 50, 10)
        self.max_gap_slider = self.create_slider("Maximum Gap", 1, 20, 2)
        self.min_angle_slider = self.create_slider("Minimum Angle", 0, 90, 20)

        layout.addWidget(self.min_length_slider)
        layout.addWidget(self.max_gap_slider)
        layout.addWidget(self.min_angle_slider)

        # Add apply button
        self.apply_button = QPushButton("Apply to Main Window")
        self.apply_button.clicked.connect(self.apply_to_main)
        layout.addWidget(self.apply_button)

        self.setLayout(layout)
        
        # Initial processing
        self.process_and_display()

    def create_slider(self, name, min_val, max_val, default_val):
        container = QWidget()
        layout = QHBoxLayout()
        
        label = QLabel(name)
        layout.addWidget(label)
        
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val, max_val)  
        slider.setValue(default_val)
        slider.setTickPosition(QSlider.TicksBelow)
        # Direct connection to process_and_display
        slider.valueChanged.connect(lambda: self.process_and_display())
        
        value_label = QLabel(str(default_val))
        slider.valueChanged.connect(lambda v: value_label.setText(str(v)))
        
        layout.addWidget(slider)
        layout.addWidget(value_label)
        
        container.setLayout(layout)
        return container

    def process_and_display(self):
        if self.image is None:
            return

        # Get current values
        min_length = self.min_length_slider.findChild(QSlider).value()
        max_gap = self.max_gap_slider.findChild(QSlider).value()
        min_angle = self.min_angle_slider.findChild(QSlider).value()

        # Process edges
        edge_linker = edgelink(self.image, min_length)
        edge_linker.get_edgelist()
        edge_lists = [np.array(edge) for edge in edge_linker.edgelist if len(edge) > 0]
        processed_edges = self.post_process_edges(edge_lists, max_gap, min_angle)

        # Create and store output image
        self.edge_linked_image = self.create_edge_image(processed_edges)
        
        # Update visualization
        self.visualize_edge_lists(processed_edges)
        
        # Emit signal with result immediately
        self.edge_link_updated.emit(self.edge_linked_image)


    def create_edge_image(self, edge_lists):
        edge_image = np.zeros(self.image.shape, dtype=np.uint8)
        for edge in edge_lists:
            for point in edge:
                if 0 <= point[0] < edge_image.shape[0] and 0 <= point[1] < edge_image.shape[1]:
                    edge_image[int(point[0]), int(point[1])] = 255
        return edge_image

    def visualize_edge_lists(self, edge_lists):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        edge_image = self.create_edge_image(edge_lists)
        ax.imshow(edge_image, cmap='gray')
        ax.axis('off')
        self.canvas.draw()

    def post_process_edges(self, edge_lists, max_gap, min_angle):
        processed_edges = []
        for edge in edge_lists:
            new_edge = [edge[0]]
            for i in range(1, len(edge)):
                if self.point_distance(new_edge[-1], edge[i]) <= max_gap:
                    if len(new_edge) < 2 or self.angle_between_points(new_edge[-2], new_edge[-1], edge[i]) >= min_angle:
                        new_edge.append(edge[i])
            if len(new_edge) > 1:
                processed_edges.append(np.array(new_edge))
        return processed_edges

    def point_distance(self, p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def angle_between_points(self, p1, p2, p3):
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        if np.all(v1 == 0) or np.all(v2 == 0):
            return 0
            
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    def apply_to_main(self):
        if self.edge_linked_image is not None:
            self.edge_link_updated.emit(self.edge_linked_image)
            self.accept()

class PrecisionRecallDialog(QDialog):
    def __init__(self, parent=None, **filter_data):
        super().__init__(parent)
        self.setWindowTitle("Precision and Recall Metrics")
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        # Create the plot
        self.figure, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 10))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Create a text area for detailed information
        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        layout.addWidget(self.text_area)

        self.setLayout(layout)

        self.plot_data(filter_data)

    def plot_data(self, filter_data):
        filters = list(filter_data.keys())
        precisions = [data['precision'] for data in filter_data.values()]
        recalls = [data['recall'] for data in filter_data.values()]

        # Precision plot
        self.ax1.bar(filters, precisions, color=['blue', 'green', 'red'], alpha=0.7)
        self.ax1.set_ylabel('Precision')
        self.ax1.set_title('Precision by Filter')
        self.ax1.set_ylim(0, 1)

        # Recall plot
        self.ax2.bar(filters, recalls, color=['blue', 'green', 'red'], alpha=0.7)
        self.ax2.set_ylabel('Recall')
        self.ax2.set_title('Recall by Filter')
        self.ax2.set_ylim(0, 1)

        self.figure.tight_layout()
        self.canvas.draw()

        # Update text area with detailed information
        text = ""
        for filter_name, data in filter_data.items():
            text += f"{filter_name} Filter:\n"
            text += f"Precision: {data['precision']:.4f}\n"
            text += f"Recall: {data['recall']:.4f}\n"
            if 'low' in data and 'high' in data:
                text += f"Thresholds: low={data['low']}, high={data['high']}\n"
            elif 'ksize' in data:
                text += f"Kernel Size: {data['ksize']}\n"
            elif 'min_contrast' in data:
                text += f"Min Contrast: {data['min_contrast']}\n"
            text += "\n"
        self.text_area.setText(text)

class BatchProcessDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Process")
        self.setGeometry(100, 100, 600, 400)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Input files selection
        input_layout = QHBoxLayout()
        self.input_list = QListWidget()
        input_layout.addWidget(QLabel("Input Files:"))
        input_layout.addWidget(self.input_list)
        add_button = QPushButton("Add Files")
        add_button.clicked.connect(self.add_files)
        input_layout.addWidget(add_button)
        layout.addLayout(input_layout)

        # Filter selection
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Desired Filter:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["Canny", "Sobel", "Shearlet"])
        filter_layout.addWidget(self.filter_combo)
        layout.addLayout(filter_layout)

        # Output directory selection
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output Directory:"))
        self.output_edit = QLineEdit()
        output_layout.addWidget(self.output_edit)
        output_button = QPushButton("Browse")
        output_button.clicked.connect(self.select_output_dir)
        output_layout.addWidget(output_button)
        layout.addLayout(output_layout)

        # Process button
        process_button = QPushButton("Process")
        process_button.clicked.connect(self.process_batch)
        layout.addWidget(process_button)

        self.setLayout(layout)

    def add_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Input Files", "", "Image Files (*.png *.jpg *.bmp)")
        self.input_list.addItems(files)

    def select_output_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        self.output_edit.setText(directory)

    def process_batch(self):
        input_files = [self.input_list.item(i).text() for i in range(self.input_list.count())]
        output_dir = self.output_edit.text()
        filter_type = self.filter_combo.currentText()

        # Here you would call a method from the main window to process the files
        self.parent().process_batch_files(input_files, output_dir, filter_type)
        self.accept()

class SaveMaskDialog(QDialog):
    def __init__(self, image):
        super().__init__()
        self.image = image
        self.filtered_image = image
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Save Mask with Filter')
        self.setGeometry(100, 100, 600, 400)

        layout = QVBoxLayout()

        self.filter_type_combo = QComboBox()
        self.filter_type_combo.addItems(['canny', 'sobel'])
        self.filter_type_combo.currentIndexChanged.connect(self.update_filter)
        layout.addWidget(QLabel("Filter Type"))
        layout.addWidget(self.filter_type_combo)

        self.threshold1_slider = QSlider(Qt.Horizontal)
        self.threshold1_slider.setRange(0, 255)
        self.threshold1_slider.setValue(50)
        self.threshold1_slider.valueChanged.connect(self.update_filter)
        layout.addWidget(QLabel("Canny Threshold 1"))
        layout.addWidget(self.threshold1_slider)
        self.threshold1_label = QLabel("50")
        layout.addWidget(self.threshold1_label)

        self.threshold2_slider = QSlider(Qt.Horizontal)
        self.threshold2_slider.setRange(0, 255)
        self.threshold2_slider.setValue(150)
        self.threshold2_slider.valueChanged.connect(self.update_filter)
        layout.addWidget(QLabel("Canny Threshold 2"))
        layout.addWidget(self.threshold2_slider)
        self.threshold2_label = QLabel("150")
        layout.addWidget(self.threshold2_label)

        self.ksize_slider = QSlider(Qt.Horizontal)
        self.ksize_slider.setRange(1, 31)
        self.ksize_slider.setValue(3)
        self.ksize_slider.setSingleStep(2)
        self.ksize_slider.setTickInterval(2)
        self.ksize_slider.setTickPosition(QSlider.TicksBelow)
        self.ksize_slider.valueChanged.connect(self.update_filter)
        layout.addWidget(QLabel("Sobel Kernel Size"))
        layout.addWidget(self.ksize_slider)
        self.ksize_label = QLabel("3")
        layout.addWidget(self.ksize_label)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.apply_button = QPushButton("Apply Filter and Save Mask")
        self.apply_button.clicked.connect(self.save_mask)
        layout.addWidget(self.apply_button)

        self.setLayout(layout)
        self.update_filter()

    def update_filter(self):
        filter_type = self.filter_type_combo.currentText()
        if filter_type == 'canny':
            self.threshold1_slider.setEnabled(True)
            self.threshold2_slider.setEnabled(True)
            self.ksize_slider.setEnabled(False)
            self.apply_canny_filter()
        elif filter_type == 'sobel':
            self.threshold1_slider.setEnabled(False)
            self.threshold2_slider.setEnabled(False)
            self.ksize_slider.setEnabled(True)
            self.apply_sobel_filter()
        self.show_filtered_image()

    # def apply_canny_filter(self):
    #     threshold1 = self.threshold1_slider.value()
    #     threshold2 = self.threshold2_slider.value()
    #     self.filtered_image = cv2.Canny(self.image, threshold1, threshold2)
    #     self.threshold1_label.setText(str(threshold1))
    #     self.threshold2_label.setText(str(threshold2))

    # def apply_sobel_filter(self):
    #     ksize = self.ksize_slider.value()
    #     if ksize % 2 == 0:
    #         ksize += 1  # Ensure ksize is odd
    #     grad_x = cv2.Sobel(self.image, cv2.CV_64F, 1, 0, ksize=ksize)
    #     grad_y = cv2.Sobel(self.image, cv2.CV_64F, 0, 1, ksize=ksize)
    #     abs_grad_x = cv2.convertScaleAbs(grad_x)
    #     abs_grad_y = cv2.convertScaleAbs(grad_y)
    #     self.filtered_image = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    #     _, self.filtered_image = cv2.threshold(self.filtered_image, 0, 255, cv2.THRESH_BINARY_INV)
    #     self.ksize_label.setText(str(ksize))

    def show_filtered_image(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.imshow(self.filtered_image, cmap='gray')
        ax.axis('off')
        self.canvas.draw()

    def save_mask(self):
        mask_path = QFileDialog.getSaveFileName(self, "Save Mask", "", "PNG Files (*.png);;All Files (*)")[0]
        if mask_path:
            cv2.imwrite(mask_path, self.filtered_image)
        self.accept()


class ExportDialog(QDialog):
    def __init__(self, parent=None, export_type=""):
        super().__init__(parent)
        self.setWindowTitle(f"Export to {export_type}")
        self.setGeometry(100, 100, 300, 150)

        layout = QVBoxLayout()

        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["Original", "Canny", "Sobel", "Shearlet"])
        layout.addWidget(QLabel("Select Filter:"))
        layout.addWidget(self.filter_combo)

        self.export_button = QPushButton("Export")
        self.export_button.clicked.connect(self.accept)
        layout.addWidget(self.export_button)

        self.setLayout(layout)

    def get_selected_filter(self):
        return self.filter_combo.currentText()

class ImagePropertiesDialog(QDialog):
    def __init__(self, image):
        super().__init__()
        self.image = image
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Image Properties')
        self.setGeometry(100, 100, 600, 400)

        layout = QVBoxLayout()

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.properties_label = QLabel()
        layout.addWidget(self.properties_label)

        self.plot_histogram()
        self.show_properties()
        self.setLayout(layout)

    def plot_histogram(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.hist(self.image.ravel(), bins=256, color='black', alpha=0.7)
        ax.set_title('Histogram')
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Frequency')
        self.canvas.draw()

    def show_properties(self):
        mean = np.mean(self.image)
        median = np.median(self.image)
        std_dev = np.std(self.image)
        min_val = np.min(self.image)
        max_val = np.max(self.image)

        properties_text = (f"Mean: {mean:.2f}\n"
                           f"Median: {median:.2f}\n"
                           f"Standard Deviation: {std_dev:.2f}\n"
                           f"Min Value: {min_val}\n"
                           f"Max Value: {max_val}")

        self.properties_label.setText(properties_text)

class ShearWaveletWindow(QDialog):
    def __init__(self, title, image, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setGeometry(100, 100, 600, 400)

        layout = QVBoxLayout()
        figure = Figure(figsize=(5, 4), dpi=100)
        canvas = FigureCanvas(figure)
        layout.addWidget(canvas)

        ax = figure.add_subplot(111)
        ax.imshow(image, cmap='jet')
        ax.axis('off')

        self.setLayout(layout)



class FourierTransformDialog(QDialog):
    def __init__(self, image):
        super().__init__()
        self.image = image
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Fourier Transform')
        self.setGeometry(100, 100, 600, 400)

        layout = QVBoxLayout()

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.plot_fourier_transform()
        self.setLayout(layout)

    def plot_fourier_transform(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # Compute the 2D Fourier Transform
        f_transform = np.fft.fftshift(np.fft.fft2(self.image))
        magnitude_spectrum = np.log(np.abs(f_transform) + 1)

        ax.imshow(magnitude_spectrum, cmap='gray')
        ax.set_title('Fourier Transform (Magnitude Spectrum)')
        ax.axis('off')
        self.canvas.draw()
class IntensityProfileDialog(QDialog):
    def __init__(self, image):
        super().__init__()
        self.image = image
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Intensity Profile')
        self.setGeometry(100, 100, 600, 400)

        layout = QVBoxLayout()

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.plot_intensity_profile()
        self.setLayout(layout)

    def plot_intensity_profile(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # Here we just take the profile along the middle row
        mid_row = self.image[self.image.shape[0] // 2, :]
        ax.plot(mid_row, color='black')
        ax.set_title('Intensity Profile (Middle Row)')
        ax.set_xlabel('Column Index')
        ax.set_ylabel('Intensity Value')
        self.canvas.draw()
class ColorHistogramDialog(QDialog):
    def __init__(self, image):
        super().__init__()
        self.image = image
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Color Histogram')
        self.setGeometry(100, 100, 600, 400)

        layout = QVBoxLayout()

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.plot_color_histogram()
        self.setLayout(layout)

    def plot_color_histogram(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        if len(self.image.shape) == 2:
            # If the image is grayscale, just plot the histogram
            ax.hist(self.image.ravel(), bins=256, color='black', alpha=0.7)
        else:
            # If the image is colored, plot histograms for each channel
            colors = ('b', 'g', 'r')
            for i, color in enumerate(colors):
                hist = cv2.calcHist([self.image], [i], None, [256], [0, 256])
                ax.plot(hist, color=color)

        ax.set_title('Color Histogram')
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Frequency')
        self.canvas.draw()


class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.img = None
        self.mask = None
        self.filtered_img = None
        self.shearlet_system = None
        self.setGeometry(100, 100, 1200, 800)
        self.initUI()

    def initUI(self):
        self.setWindowTitle('DOMStudioImage')

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Create tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setMovable(True)  # Allow tabs to be moved
        main_layout.addWidget(self.tab_widget)

        # Add filter tabs
        self.add_filter_tab("Canny", CannyFilterTab)
        self.add_filter_tab("Sobel", SobelFilterTab)
        self.add_filter_tab("Shearlet", ShearletFilterTab)
        self.add_filter_tab("Laplacian", LaplacianFilterTab)
        self.add_filter_tab("Roberts", RobertsFilterTab)
        self.add_filter_tab("HED", HEDFilterTab)  # Add HED tab

        # Add "+" tab for creating new tabs
        self.tab_widget.addTab(QWidget(), "+")
        self.tab_widget.tabBarClicked.connect(self.handle_tab_click)

        # Load image button
        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.load_image)
        main_layout.addWidget(self.load_button)

        # Create a horizontal layout for the "Clean Short Edges" controls
        clean_edges_layout = QHBoxLayout()

        # Clean Short Edges button
        self.clean_edges_button = QPushButton("Clean Short Edges")
        self.clean_edges_button.clicked.connect(self.clean_short_edges)
        self.clean_edges_button.setEnabled(False)  # Initially disabled
        clean_edges_layout.addWidget(self.clean_edges_button)
        # Min Size label
        self.min_size_label = QLabel("Min Size:")
        clean_edges_layout.addWidget(self.min_size_label)

        # Min Size spin box
        self.min_size_spinbox = QSpinBox()
        self.min_size_spinbox.setMinimum(1)
        self.min_size_spinbox.setMaximum(1000)  # Adjust as needed
        self.min_size_spinbox.setValue(10)      # Default value
        self.min_size_spinbox.setSuffix(" px")
        clean_edges_layout.addWidget(self.min_size_spinbox)

         # Add the clean edges layout to the main layout
        main_layout.addLayout(clean_edges_layout)

        # Manual Interpretation button
        self.manual_interpretation_button = QPushButton("Manual Interpretation")
        self.manual_interpretation_button.clicked.connect(self.open_manual_interpretation)
        self.manual_interpretation_button.setEnabled(False)  # Initially disabled
        main_layout.addWidget(self.manual_interpretation_button)

        self.createMenus()

    def add_filter_tab(self, filter_name, filter_class):
        tab = filter_class(self)
        self.tab_widget.insertTab(self.tab_widget.count() - 1, tab, filter_name)
    def open_manual_interpretation(self):
        current_tab = self.tab_widget.currentWidget()
        if isinstance(current_tab, FilterTab) and current_tab.filtered_image is not None:
            self.manual_interpretation_window = ManualInterpretationWindow(
                original_image=self.img,
                filtered_image=current_tab.filtered_image,
                parent=self
            )
            self.manual_interpretation_window.show()
        else:
            QMessageBox.warning(
                self,
                "Warning",
                "Please load an image and apply a filter first before using manual interpretation."
            )

    def handle_tab_click(self, index):
        if self.tab_widget.tabText(index) == "+":
            filters = [
                "Canny",
                "Sobel",
                "Shearlet",
                "Laplacian",
                "Roberts",
                "HED"  # Add HED to the list
            ]
            filter_name, ok = QInputDialog.getItem(self, "Select Filter", "Choose a filter:", filters, 0, False)
            if ok and filter_name:
                filter_classes = {
                    "Canny": CannyFilterTab,
                    "Sobel": SobelFilterTab,
                    "Shearlet": ShearletFilterTab,
                    "Laplacian": LaplacianFilterTab,
                    "Roberts": RobertsFilterTab,
                    "HED": HEDFilterTab  # Map "HED" to HEDFilterTab
                }
                filter_class = filter_classes.get(filter_name)
                if filter_class:
                    new_filter = filter_class(self)
                    new_filter.setWindowTitle(filter_name)
                    self.tab_widget.insertTab(index, new_filter, filter_name)
                    self.tab_widget.setCurrentIndex(index)
                else:
                    QMessageBox.warning(self, "Error", f"Unknown filter: {filter_name}")

    

    def clean_short_edges(self):
        current_tab = self.tab_widget.currentWidget()
        if isinstance(current_tab, FilterTab) and current_tab.filtered_image is not None:
            # Get the filtered image
            image = current_tab.filtered_image.copy()
            
            # Get 'min_size' from the spin box
            min_size = self.min_size_spinbox.value()
            
            # Apply skeletonization first if not already done
            if not current_tab.skeletonize_checkbox.isChecked():
                image = skeletonize(image > 0).astype(np.uint8) * 255

            # Label connected components
            labeled, num_features = measure.label(image > 0, return_num=True, connectivity=2)
            
            # Filter components based on size
            cleaned_image = np.zeros_like(image)

            for label_id in range(1, num_features + 1):
                component = labeled == label_id
                size = np.sum(component)
                
                if size >= min_size:
                    cleaned_image[component] = 255

            # Update the filtered image
            current_tab.filtered_image = cleaned_image
            current_tab.show_filtered_image()
        else:
            QMessageBox.warning(
                self,
                "Warning",
                "Please load an image and apply a filter first before cleaning short edges."
            )

    def open_manual_interpretation(self):
        current_tab = self.tab_widget.currentWidget()
        if isinstance(current_tab, FilterTab) and self.img is not None:
            self.manual_interpretation_window = ManualInterpretationWindow(self.img, current_tab.filtered_image, self)
            self.manual_interpretation_window.show()
        else:
            QMessageBox.warning(self, "Warning", "Please load an image and apply a filter first before using manual interpretation.")

    def createMenus(self):
        menubar = self.menuBar()

        fileMenu = menubar.addMenu('File')
        newAction = QAction('New', self)
        newAction.triggered.connect(self.new_project)
        fileMenu.addAction(newAction)

        openAction = QAction('Open', self)
        openAction.triggered.connect(self.open_project)
        fileMenu.addAction(openAction)

        saveAction = QAction('Save', self)
        saveAction.triggered.connect(self.save_project)
        fileMenu.addAction(saveAction)

        exportMenu = fileMenu.addMenu('Export')

        exitAction = QAction('Exit', self)
        exitAction.triggered.connect(self.close)
        fileMenu.addAction(exitAction)

        exportShapefileAction = QAction('Export to Shapefile', self)
        exportShapefileAction.triggered.connect(self.export_to_shapefile)
        exportMenu.addAction(exportShapefileAction)

        exportVectorAction = QAction('Export to Vector File', self)
        exportVectorAction.triggered.connect(self.export_to_vector)
        exportMenu.addAction(exportVectorAction)

        exportPngAction = QAction('Export to PNG', self)
        exportPngAction.triggered.connect(self.export_to_png)
        exportMenu.addAction(exportPngAction)

        exportJpegAction = QAction('Export to JPEG', self)
        exportJpegAction.triggered.connect(self.export_to_jpeg)
        exportMenu.addAction(exportJpegAction)

        exportTiffAction = QAction('Export to TIFF', self)
        exportTiffAction.triggered.connect(self.export_to_tiff)
        exportMenu.addAction(exportTiffAction)

        toolsMenu = menubar.addMenu('Tools')
        imagePropertiesMenu = toolsMenu.addMenu('Image Properties')


        calculateStatsMenu = toolsMenu.addMenu('Calculate Statistics')
        calculateStatsAction = QAction('Precision and Recall', self)
        calculateStatsAction.triggered.connect(self.show_statistics_dialog)
        calculateStatsMenu.addAction(calculateStatsAction)

        shearWaveletMenu = toolsMenu.addMenu('Shear Wavelet')
        edgeMeasurementAction = QAction('Edge Measurement', self)
        edgeMeasurementAction.triggered.connect(self.show_edge_measurement)
        shearWaveletMenu.addAction(edgeMeasurementAction)

        thinEdgeMeasurementAction = QAction('Thin Edge Measurement', self)
        thinEdgeMeasurementAction.triggered.connect(self.show_thin_edge_measurement)
        shearWaveletMenu.addAction(thinEdgeMeasurementAction)

        orientationMeasurementAction = QAction('Orientation Measurement', self)
        orientationMeasurementAction.triggered.connect(self.show_orientation_measurement)
        shearWaveletMenu.addAction(orientationMeasurementAction)

        curvatureMeasurementAction = QAction('Curvature Measurement', self)
        curvatureMeasurementAction.triggered.connect(self.show_curvature_measurement)
        shearWaveletMenu.addAction(curvatureMeasurementAction)

        conversionMenu = toolsMenu.addMenu('Conversion')
        batchProcessAction = QAction('Create Batch Process', self)
        batchProcessAction.triggered.connect(self.open_batch_process_dialog)
        conversionMenu.addAction(batchProcessAction)

        plotHistogramAction = QAction('Plot Histogram', self)
        plotHistogramAction.triggered.connect(self.open_image_properties_dialog)
        imagePropertiesMenu.addAction(plotHistogramAction)

        plotIntensityProfileAction = QAction('Intensity Profile', self)
        plotIntensityProfileAction.triggered.connect(self.open_intensity_profile_dialog)
        imagePropertiesMenu.addAction(plotIntensityProfileAction)

        plotFourierTransformAction = QAction('Fourier Transform', self)
        plotFourierTransformAction.triggered.connect(self.open_fourier_transform_dialog)
        imagePropertiesMenu.addAction(plotFourierTransformAction)

        plotColorHistogramAction = QAction('Color Histogram', self)
        plotColorHistogramAction.triggered.connect(self.open_color_histogram_dialog)
        imagePropertiesMenu.addAction(plotColorHistogramAction)

        helpMenu = menubar.addMenu('Help')
        helpMenu.addAction('About')

    def open_batch_process_dialog(self):
        dialog = BatchProcessDialog(self)
        dialog.exec_()

    def show_statistics_dialog(self):
        if self.img is None:
            QMessageBox.warning(self, "Error", "Please load an image first.")
            return

        # Ask user to select a manual interpretation mask
        mask_path, _ = QFileDialog.getOpenFileName(self, "Select Manual Interpretation Mask", "",
                                                   "Image Files (*.png *.jpg *.bmp)")
        if not mask_path:
            return

        # Load and preprocess the mask
        manual_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        manual_mask = cv2.resize(manual_mask, (self.img.shape[1], self.img.shape[0]))
        manual_mask = (manual_mask > 128).astype(np.uint8)  # Binarize the mask

        # Calculate for each filter
        filter_data = {}
        for i in range(self.tab_widget.count() - 1):  # Exclude the "+" tab
            tab = self.tab_widget.widget(i)
            if isinstance(tab, FilterTab) and tab.filtered_image is not None:
                filter_name = self.tab_widget.tabText(i)

                # Resize the filtered image to match the manual mask
                resized_filtered_image = cv2.resize(tab.filtered_image, (manual_mask.shape[1], manual_mask.shape[0]))

                precision, recall = self.calculate_precision_recall(manual_mask, resized_filtered_image)
                filter_data[filter_name] = {
                    'precision': precision,
                    'recall': recall
                }

                # Add specific filter parameters
                if isinstance(tab, CannyFilterTab):
                    filter_data[filter_name]['low'] = tab.threshold1.value()
                    filter_data[filter_name]['high'] = tab.threshold2.value()
                elif isinstance(tab, SobelFilterTab):
                    filter_data[filter_name]['ksize'] = tab.ksize.value()
                elif isinstance(tab, ShearletFilterTab):
                    filter_data[filter_name]['min_contrast'] = tab.min_contrast.value()

        # Show results in the new dialog
        dialog = PrecisionRecallDialog(self, **filter_data)
        dialog.exec_()

    def calculate_precision_recall(self, ground_truth, prediction):
        # Ensure both inputs are binary
        ground_truth = (ground_truth > 0).astype(np.uint8)
        prediction = (prediction > 0).astype(np.uint8)

        true_positives = np.sum(np.logical_and(prediction == 1, ground_truth == 1))
        false_positives = np.sum(np.logical_and(prediction == 1, ground_truth == 0))
        false_negatives = np.sum(np.logical_and(prediction == 0, ground_truth == 1))

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

        return precision, recall

    def show_edge_measurement(self):
        if self.img is None or self.shearlet_system is None:
            QMessageBox.warning(self, "Error", "Please load an image first.")
            return

        edges, _ = self.shearlet_system.detect(self.img, min_contrast=self.shearletMinContrast.value())
        window = ShearWaveletWindow("Edge Measurement", edges, self)
        window.exec_()

    def show_thin_edge_measurement(self):
        if self.img is None or self.shearlet_system is None:
            QMessageBox.warning(self, "Error", "Please load an image first.")
            return

        edges, _ = self.shearlet_system.detect(self.img, min_contrast=self.shearletMinContrast.value())
        thinned_edges = mask(edges, thin_mask(edges))
        window = ShearWaveletWindow("Thin Edge Measurement", thinned_edges, self)
        window.exec_()

    def show_orientation_measurement(self):
        if self.img is None or self.shearlet_system is None:
            QMessageBox.warning(self, "Error", "Please load an image first.")
            return

        _, orientations = self.shearlet_system.detect(self.img, min_contrast=self.shearletMinContrast.value())
        window = ShearWaveletWindow("Orientation Measurement", orientations, self)
        window.exec_()

    def show_curvature_measurement(self):
        if self.img is None or self.shearlet_system is None:
            QMessageBox.warning(self, "Error", "Please load an image first.")
            return

        edges, orientations = self.shearlet_system.detect(self.img, min_contrast=self.shearletMinContrast.value())
        thinned_orientations = mask(edges, thin_mask(edges))
        curvature_map = curvature(thinned_orientations)
        rgb_curvature = curvature_rgb(curvature_map, max_curvature=3)
        window = ShearWaveletWindow("Curvature Measurement", rgb_curvature, self)
        window.exec_()


    def new_project(self):
        self.img = None
        self.mask = None
        self.filtered_img = None
        self.shearlet_system = None

        # Clear all filter tabs
        for i in range(self.tab_widget.count() - 1):  # Exclude the "+" tab
            tab = self.tab_widget.widget(i)
            if isinstance(tab, FilterTab):
                tab.input_image = None
                tab.filtered_image = None
                tab.show_input_image()
                tab.show_filtered_image()

        # Disable buttons that require an image
        self.manual_interpretation_button.setEnabled(False)
        self.clean_edges_button.setEnabled(False)

        QMessageBox.information(self, "New Project", "New project created. Please load an image.")

    def open_project(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Project File", "",
                                                   "Project Files (*.pkl);;All Files (*)")
        if file_path:
            try:
                with open(file_path, 'rb') as file:
                    project_data = pickle.load(file)
                    self.img = project_data['image']
                    self.mask = project_data.get('mask')
                    self.filtered_img = project_data.get('filtered_image')

                    # Update all filter tabs with the new image
                    for i in range(self.tab_widget.count() - 1):  # Exclude the "+" tab
                        tab = self.tab_widget.widget(i)
                        if isinstance(tab, FilterTab):
                            tab.set_input_image(self.img)
                            tab.update_filter()

                    # Enable buttons that require an image
                    self.manual_interpretation_button.setEnabled(True)
                    self.clean_edges_button.setEnabled(True)

                QMessageBox.information(self, "Project Opened", "Project loaded successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to open project: {str(e)}")

    def save_project(self):
        if self.img is None:
            QMessageBox.warning(self, "Error", "No image loaded. Cannot save project.")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Project File", "", "Project Files (*.pkl);;All Files (*)")
        if file_path:
            try:
                project_data = {
                    'image': self.img,
                    'mask': self.mask,
                    'filtered_image': self.filtered_img
                }

                # Save filter-specific data
                for i in range(self.tab_widget.count() - 1):  # Exclude the "+" tab
                    tab = self.tab_widget.widget(i)
                    if isinstance(tab, FilterTab):
                        filter_name = self.tab_widget.tabText(i)
                        project_data[f'{filter_name}_filtered'] = tab.filtered_image

                with open(file_path, 'wb') as file:
                    pickle.dump(project_data, file)

                QMessageBox.information(self, "Project Saved", "Project saved successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save project: {str(e)}")

    def open_image_properties_dialog(self):
        if self.img is not None:
            dialog = ImagePropertiesDialog(self.img)
            dialog.exec_()

    def open_intensity_profile_dialog(self):
        if self.img is not None:
            dialog = IntensityProfileDialog(self.img)
            dialog.exec_()

    def open_fourier_transform_dialog(self):
        if self.img is not None:
            dialog = FourierTransformDialog(self.img)
            dialog.exec_()

    def open_color_histogram_dialog(self):
        if self.img is not None and len(self.img.shape) == 3:
            dialog = ColorHistogramDialog(self.img)
            dialog.exec_()
    def convert_to_binary(self, image, method='otsu'):
        if method == 'otsu':
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == 'adaptive':
            binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        else:
            raise ValueError("Unsupported binarization method")
        return binary
    def invert_binary_image(image):
        return cv2.bitwise_not(image)

    def image_to_lines(self, binary_image, min_length=10):
        print("Starting image_to_lines function")
        print(f"Binary image shape: {binary_image.shape}")
        print(f"Binary image dtype: {binary_image.dtype}")
        print(f"Binary image min value: {np.min(binary_image)}")
        print(f"Binary image max value: {np.max(binary_image)}")

        # Ensure the image is binary
        _, binary_image = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)

        try:
            # Find contours
            print("Finding contours")
            contours, _ = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            print(f"Number of contours found: {len(contours)}")

            lines = []
            for i, contour in enumerate(contours):
                try:
                    print(f"Processing contour {i}, length: {len(contour)}")

                    # Skip contours that are too small
                    if len(contour) < 2:
                        print(f"Skipping contour {i}: too few points")
                        continue

                    # Simplify the contour
                    epsilon = 0.01 * cv2.arcLength(contour, False)
                    approx = cv2.approxPolyDP(contour, epsilon, False)

                    # Convert to LineString if it meets the minimum length criteria
                    contour_length = cv2.arcLength(approx, False)
                    if len(approx) >= 2 and contour_length > min_length:
                        line = LineString(approx.reshape(-1, 2))
                        lines.append(line)
                        print(f"Added line from contour {i}, length: {contour_length}")
                    else:
                        print(f"Skipping contour {i}: length {contour_length} < {min_length}")

                except Exception as e:
                    print(f"Error processing contour {i}: {str(e)}")
                    continue

            print(f"Number of lines detected: {len(lines)}")
            return lines

        except Exception as e:
            print(f"Error in image_to_lines: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    def export_to_shapefile(self):
        if self.img is None:
            QMessageBox.warning(self, "Error", "No image loaded.")
            return

        dialog = ExportDialog(self, "Shapefile")
        if dialog.exec_():
            selected_filter = dialog.get_selected_filter()
            filtered_image = self.get_filtered_image(selected_filter)

            if filtered_image is None:
                QMessageBox.warning(self, "Error", f"No filtered image available for {selected_filter}.")
                return

            # Convert to binary
            binary_method, ok = QInputDialog.getItem(self, "Binary Conversion",
                                                     "Select binary conversion method:",
                                                     ["Otsu", "Adaptive"], 0, False)
            if ok and binary_method:
                binary_image = self.convert_to_binary(filtered_image, method=binary_method.lower())
            else:
                return

            lines = self.image_to_lines(binary_image)
            if not lines:
                QMessageBox.warning(self, "Warning",
                                    "No lines were detected in the image. The shapefile will not be created.")
                return

            gdf = gpd.GeoDataFrame({'geometry': lines})
            if hasattr(self, 'geotiff_crs'):
                gdf.crs = self.geotiff_crs

            save_path, _ = QFileDialog.getSaveFileName(self, "Save Shapefile", "", "Shapefile (*.shp)")
            if save_path:
                try:
                    gdf.to_file(save_path)
                    QMessageBox.information(self, "Success", "Shapefile exported successfully")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to save shapefile: {str(e)}")

    def convert_edges_to_lines(self, edges, simplification_tolerance=1.0):
        debug_print("Converting edges to SVG paths")
        try:
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)

            paths = []
            for contour in contours:
                # Simplify the contour
                epsilon = simplification_tolerance * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # Create path data
                path_data = f"M {approx[0][0][0]} {approx[0][0][1]}"
                for point in approx[1:]:
                    path_data += f" L {point[0][0]} {point[0][1]}"
                path_data += " Z"
                paths.append(path_data)

            debug_print(f"Converted edges to {len(paths)} SVG paths")
            return paths
        except Exception as e:
            debug_print(f"Error in convert_edges_to_lines: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    def create_vector_lines(self, paths):
        debug_print("Creating SVG content")
        try:
            # Create SVG root element
            svg = Element('svg')
            svg.set('xmlns', 'http://www.w3.org/2000/svg')
            svg.set('width', str(self.img.shape[1]))
            svg.set('height', str(self.img.shape[0]))

            # Add a white background rectangle
            background = SubElement(svg, 'rect')
            background.set('width', '100%')
            background.set('height', '100%')
            background.set('fill', 'white')

            # Find the bounding box of all paths
            all_points = []
            for path_data in paths:
                all_points.extend(parse_path(path_data))

            if all_points:
                min_x, min_y = min(all_points[0::2]), min(all_points[1::2])
                max_x, max_y = max(all_points[0::2]), max(all_points[1::2])

                # Calculate scaling factor to fit all paths within the SVG
                width = max_x - min_x
                height = max_y - min_y
                scale = min(self.img.shape[1] / width, self.img.shape[0] / height) * 0.9  # 90% of the available space

                # Calculate translation to center the paths
                translate_x = (self.img.shape[1] - width * scale) / 2 - min_x * scale
                translate_y = (self.img.shape[0] - height * scale) / 2 - min_y * scale

                # Create a group for all paths with the calculated transform
                group = SubElement(svg, 'g')
                group.set('transform', f'translate({translate_x},{translate_y}) scale({scale})')

                for path_data in paths:
                    path = SubElement(group, 'path')
                    path.set('fill', 'none')
                    path.set('stroke', 'black')
                    path.set('stroke-width', '0.5')
                    path.set('d', path_data)
            else:
                debug_print("No valid points found in paths")

            # Convert to string
            rough_string = tostring(svg, 'utf-8')
            reparsed = minidom.parseString(rough_string)
            pretty_svg = reparsed.toprettyxml(indent="  ")

            debug_print(f"Created SVG content with {len(paths)} paths")
            return pretty_svg
        except Exception as e:
            debug_print(f"Error in create_vector_lines: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def export_to_vector(self):
        if self.img is None:
            QMessageBox.warning(self, "Error", "No image loaded.")
            return

        dialog = ExportDialog(self, "Vector")
        if dialog.exec_():
            selected_filter = dialog.get_selected_filter()
            filtered_image = self.get_filtered_image(selected_filter)

            if filtered_image is None:
                QMessageBox.warning(self, "Error", f"No filtered image available for {selected_filter}.")
                return

            paths = self.convert_edges_to_lines(filtered_image)
            if not paths:
                QMessageBox.warning(self, "Warning", "No valid paths were detected.")
                return

            svg_content = self.create_vector_lines(paths)
            if not svg_content:
                QMessageBox.warning(self, "Error", "Failed to create SVG content.")
                return

            save_path, _ = QFileDialog.getSaveFileName(self, "Save Vector", "", "SVG Files (*.svg)")
            if save_path:
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(svg_content)
                QMessageBox.information(self, "Success", "File saved as SVG")

    def export_to_png(self):
        if self.img is None:
            QMessageBox.warning(self, "Error", "No image loaded.")
            return

        dialog = ExportDialog(self, "PNG")
        if dialog.exec_():
            selected_filter = dialog.get_selected_filter()
            filtered_image = self.get_filtered_image(selected_filter)

            if filtered_image is None:
                QMessageBox.warning(self, "Error", f"No filtered image available for {selected_filter}.")
                return

            save_path, _ = QFileDialog.getSaveFileName(self, "Save PNG", "", "PNG Files (*.png)")
            if save_path:
                cv2.imwrite(save_path, filtered_image)
                QMessageBox.information(self, "Success", "Image saved as PNG")

    def export_to_jpeg(self):
        if self.img is None:
            QMessageBox.warning(self, "Error", "No image loaded.")
            return

        dialog = ExportDialog(self, "JPEG")
        if dialog.exec_():
            selected_filter = dialog.get_selected_filter()
            filtered_image = self.get_filtered_image(selected_filter)

            if filtered_image is None:
                QMessageBox.warning(self, "Error", f"No filtered image available for {selected_filter}.")
                return

            save_path, _ = QFileDialog.getSaveFileName(self, "Save JPEG", "", "JPEG Files (*.jpg *.jpeg)")
            if save_path:
                # JPEG doesn't support 16-bit depth, so we need to convert to 8-bit
                if filtered_image.dtype != np.uint8:
                    filtered_image = cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                # If the image is grayscale, convert to RGB
                if len(filtered_image.shape) == 2:
                    filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_GRAY2RGB)

                # Save the image with JPEG compression
                cv2.imwrite(save_path, filtered_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                QMessageBox.information(self, "Success", "Image saved as JPEG")

    def export_to_tiff(self):
        if self.img is None:
            QMessageBox.warning(self, "Error", "No image loaded.")
            return

        dialog = ExportDialog(self, "TIFF")
        if dialog.exec_():
            selected_filter = dialog.get_selected_filter()
            filtered_image = self.get_filtered_image(selected_filter)

            if filtered_image is None:
                QMessageBox.warning(self, "Error", f"No filtered image available for {selected_filter}.")
                return

            save_path, _ = QFileDialog.getSaveFileName(self, "Save TIFF", "", "TIFF Files (*.tif)")
            if save_path:
                if hasattr(self, 'geotiff_transform') and hasattr(self, 'geotiff_projection'):
                    # Save as GeoTIFF with original coordinates
                    driver = gdal.GetDriverByName('GTiff')
                    dataset = driver.Create(save_path, filtered_image.shape[1], filtered_image.shape[0], 1,
                                            gdal.GDT_Byte)
                    dataset.SetGeoTransform(self.geotiff_transform)
                    dataset.SetProjection(self.geotiff_projection)
                    dataset.GetRasterBand(1).WriteArray(filtered_image)
                    dataset.FlushCache()
                else:
                    # Save as regular TIFF
                    cv2.imwrite(save_path, filtered_image)
                QMessageBox.information(self, "Success", "Image saved as TIFF")

    def get_filtered_image(self, filter_name):
        for i in range(self.tab_widget.count() - 1):  # Exclude the "+" tab
            tab = self.tab_widget.widget(i)
            if isinstance(tab, FilterTab) and self.tab_widget.tabText(i) == filter_name:
                return tab.filtered_image
        return None

    def get_filter_param(self, filter_name, param_name, default_value):
        tab_index = self.tab_widget.indexOf(self.tab_widget.findChild(QWidget, filter_name))
        if tab_index == -1:
            return default_value

        tab = self.tab_widget.widget(tab_index)
        param_widget = tab.findChild(QSlider, param_name)
        return param_widget.value() if param_widget else default_value

    def image_to_polygons(self, image):
        # Ensure the image is binary
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        polygons = []
        for contour in contours:
            # Simplify the contour to reduce the number of points
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Convert the contour to a Shapely polygon
            if len(approx) >= 3:  # A polygon must have at least 3 points
                poly = Polygon(approx.reshape(-1, 2))
                if poly.is_valid:  # Ensure the polygon is valid
                    polygons.append(poly)

        return polygons

    def load_image(self):
        img_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "",
                                                "Image Files (*.png *.jpg *.bmp *.tif *.tiff)")
        if img_path:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Normalize the image
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                
                # Open preprocessing window
                preprocess_dialog = PreprocessingWindow(img, self)
                if preprocess_dialog.exec_() == QDialog.Accepted:
                    # Get the masked image
                    self.img = preprocess_dialog.get_masked_image()
                    self.img = cv2.resize(self.img, (1024, 1024))
                    self.mask = np.zeros(self.img.shape[:2], np.uint8)
                    self.filtered_img = self.img.copy()
                    self.shearlet_system = EdgeSystem(*self.img.shape)

                    # Update all filter tabs with the new image
                    for i in range(self.tab_widget.count() - 1):  # Exclude the "+" tab
                        tab = self.tab_widget.widget(i)
                        if isinstance(tab, FilterTab):
                            tab.set_input_image(self.img)

                    # Enable buttons that require an image
                    self.manual_interpretation_button.setEnabled(True)
                    self.clean_edges_button.setEnabled(True)

                    QMessageBox.information(self, "Image Loaded", 
                                        "Image loaded and preprocessed successfully.")
                else:
                    QMessageBox.warning(self, "Cancelled", 
                                    "Image preprocessing was cancelled.")
            else:
                QMessageBox.warning(self, "Error", "Failed to load image.")

    def save_mask(self):
        if self.img is None:
            return

        dialog = SaveMaskDialog(self.img)
        if dialog.exec_() == QDialog.Accepted:
            # Mask saved inside the dialog; no need to do anything here.
            pass
    def update_canny_filter(self):
        self.cannyThreshold1_label.setText(str(self.cannyThreshold1.value()))
        self.cannyThreshold2_label.setText(str(self.cannyThreshold2.value()))
        self.skeletonizeIterations_label.setText(str(self.skeletonizeIterations.value()))
        self.apply_canny_filter()

    def apply_canny_filter(self):
        if self.img is None:
            return
        threshold1 = self.cannyThreshold1.value()
        threshold2 = self.cannyThreshold2.value()

        edges = cv2.Canny(self.img, threshold1, threshold2)
        edges = self.apply_skeletonize(edges)
        self.show_canny_image(edges)
    def apply_sobel_filter(self):
        if self.img is None:
            return
        ksize = self.sobelKsize.value()
        if ksize % 2 == 0:
            ksize += 1  # Ensure ksize is odd
        grad_x = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=ksize)
        grad_y = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, ksize=ksize)
        sobel = np.sqrt(grad_x**2 + grad_y**2)
        sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        sobel = self.apply_skeletonize(sobel)
        self.show_sobel_image(sobel)

    def show_original_image(self):
        if self.img is not None:
            # Assuming you have a FigureCanvas for the original image
            fig = self.findChild(FigureCanvas, "original_image_canvas")
            if fig:
                fig.figure.clear()
                ax = fig.figure.add_subplot(111)
                ax.imshow(self.img, cmap='gray')
                ax.axis('off')
                fig.draw()



if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyWindow()
    ex.show()
    sys.exit(app.exec_())
