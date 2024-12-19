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
# import QLineF
from PyQt5.QtCore import QLineF
from PyQt5.QtGui import QTransform
from collections import deque
import torch
from PyQt5.QtCore import (Qt, pyqtSignal)
from PyQt5.QtWidgets import QSpinBox
from PyQt5.QtCore import QEvent
from PyQt5.QtCore import QEvent, Qt, QPointF
from PyQt5.QtGui import QPen, QColor, QPainterPath
from PyQt5.QtWidgets import QGraphicsPathItem, QMenu, QAction

 # Define a NodeItem representing control pointsA
# Define a NodeItem representing control points
from skimage import exposure
from skimage.util import img_as_float, img_as_ubyte
import scipy 
from utility import edgelink, seglist  # Add seglist to import

class PreprocessingWindow(QDialog):
    def __init__(self, image, parent=None):
        super().__init__(parent)
        self.original_image = image.copy()
        self.current_image = image.copy()
        self.mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
        self.drawing = False
        self.points = []
        
        # Expand state tracking
        self.effects_state = {
            'ahe': False,
            'background': False,
            'contrast': False,
            'clahe': False,
            'bilateral': False
        }
        
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

        self.background_button = QPushButton("Remove Background")
        self.background_button.setCheckable(True)
        self.background_button.clicked.connect(self.toggle_background)

        self.contrast_button = QPushButton("Enhance Contrast")
        self.contrast_button.setCheckable(True)
        self.contrast_button.clicked.connect(self.toggle_contrast)

        enhance_layout.addWidget(self.ahe_button)
        enhance_layout.addWidget(self.background_button)
        enhance_layout.addWidget(self.contrast_button)
        layout.addLayout(enhance_layout)

        # Add new enhancement controls
        advanced_layout = QHBoxLayout()
        
        # CLAHE controls
        clahe_group = QGroupBox("CLAHE")
        clahe_layout = QVBoxLayout()
        self.clahe_button = QPushButton("Apply CLAHE")
        self.clahe_button.setCheckable(True)
        self.clahe_button.clicked.connect(self.toggle_clahe)
        self.clahe_size = QSlider(Qt.Horizontal)
        self.clahe_size.setRange(2, 16)
        self.clahe_size.setValue(8)
        self.clahe_size.setEnabled(False)
        clahe_layout.addWidget(self.clahe_button)
        clahe_layout.addWidget(self.clahe_size)
        clahe_group.setLayout(clahe_layout)
        
        # Bilateral controls
        bilateral_group = QGroupBox("Bilateral Filter")
        bilateral_layout = QVBoxLayout()
        self.bilateral_button = QPushButton("Apply Bilateral")
        self.bilateral_button.setCheckable(True)
        self.bilateral_button.clicked.connect(self.toggle_bilateral)
        self.bilateral_sigma = QSlider(Qt.Horizontal)
        self.bilateral_sigma.setRange(10, 150)
        self.bilateral_sigma.setValue(75)
        self.bilateral_sigma.setEnabled(False)
        bilateral_layout.addWidget(self.bilateral_button)
        bilateral_layout.addWidget(self.bilateral_sigma)
        bilateral_group.setLayout(bilateral_layout)
        
        advanced_layout.addWidget(clahe_group)
        advanced_layout.addWidget(bilateral_group)
        layout.addLayout(advanced_layout)
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

    def toggle_clahe(self, checked):
        try:
            self.effects_state['clahe'] = checked
            self.clahe_size.setEnabled(checked)
            self.apply_active_effects()
            self.clahe_button.setText("Remove CLAHE" if checked else "Apply CLAHE")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to apply CLAHE: {str(e)}")
            self.clahe_button.setChecked(False)
            self.effects_state['clahe'] = False
            self.clahe_size.setEnabled(False)

    def toggle_bilateral(self, checked):
        try:
            self.effects_state['bilateral'] = checked
            self.bilateral_sigma.setEnabled(checked)
            self.apply_active_effects()
            self.bilateral_button.setText("Remove Bilateral" if checked else "Apply Bilateral")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to apply bilateral: {str(e)}")
            self.bilateral_button.setChecked(False)
            self.effects_state['bilateral'] = False
            self.bilateral_sigma.setEnabled(False)

    def apply_active_effects(self):
        # Start from original image
        self.current_image = self.original_image.copy()
        
        # Apply effects in specific order
        if self.effects_state['background']:
            selem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            background = cv2.morphologyEx(self.current_image, cv2.MORPH_OPEN, selem)
            self.current_image = cv2.subtract(self.current_image, background)
        
        if self.effects_state['contrast']:
            p2, p98 = np.percentile(self.current_image, (2, 98))
            self.current_image = img_as_ubyte(exposure.rescale_intensity(
                self.current_image, in_range=(p2, p98)))
        
        if self.effects_state['ahe']:
            img_float = img_as_float(self.current_image)
            img_eq = exposure.equalize_adapthist(img_float)
            self.current_image = img_as_ubyte(img_eq)

        # 1. Edge-preserving smoothing first
        if self.effects_state['bilateral']:
            sigma = self.bilateral_sigma.value()
            self.current_image = cv2.bilateralFilter(
                self.current_image, d=9, 
                sigmaColor=sigma, sigmaSpace=sigma
            )
        
        # 2. Background removal
        if self.effects_state['background']:
            selem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            background = cv2.morphologyEx(self.current_image, cv2.MORPH_OPEN, selem)
            self.current_image = cv2.subtract(self.current_image, background)
        
        # 3. Contrast enhancements
        if self.effects_state['contrast']:
            p2, p98 = np.percentile(self.current_image, (2, 98))
            self.current_image = img_as_ubyte(exposure.rescale_intensity(
                self.current_image, in_range=(p2, p98)))
        
        if self.effects_state['clahe']:
            grid_size = self.clahe_size.value()
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(grid_size, grid_size))
            self.current_image = clahe.apply(self.current_image)
        
        if self.effects_state['ahe']:
            img_float = img_as_float(self.current_image)
            img_eq = exposure.equalize_adapthist(img_float)
            self.current_image = img_as_ubyte(img_eq)
        
        self.update_display()
        
    def toggle_ahe(self, checked):
        try:
            self.effects_state['ahe'] = checked
            self.apply_active_effects()
            self.ahe_button.setText("Remove AHE" if checked else "Apply AHE")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to apply AHE: {str(e)}")
            self.ahe_button.setChecked(False)
            self.effects_state['ahe'] = False
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
    def toggle_background(self, checked):
        try:
            self.effects_state['background'] = checked
            self.apply_active_effects()
            self.background_button.setText("Restore Background" if checked else "Remove Background")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to remove background: {str(e)}")
            self.background_button.setChecked(False)
            self.effects_state['background'] = False

    def toggle_contrast(self, checked):
        try:
            self.effects_state['contrast'] = checked
            self.apply_active_effects()
            self.contrast_button.setText("Reset Contrast" if checked else "Enhance Contrast")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to enhance contrast: {str(e)}")
            self.contrast_button.setChecked(False)
            self.effects_state['contrast'] = False
class NodeItem(QGraphicsEllipseItem):
    def __init__(self, x, y, radius=3, parent=None):
        super().__init__(-radius, -radius, 2 * radius, 2 * radius, parent)
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
            # Notify all connected lines to update their paths
            for line in self.lines:
                line.updatePath()
        return super().itemChange(change, value)

    def contextMenuEvent(self, event):
        # Access the parent window (ManualInterpretationWindow)
        parent_window = self.scene().parent()
        if not hasattr(parent_window, 'delete_node'):
            return

        menu = QMenu()
        delete_action = QAction('Delete Node', self)
        delete_action.triggered.connect(lambda: parent_window.delete_node(self))
        menu.addAction(delete_action)
        menu.exec_(event.screenPos())
# Define a LineItem representing lines composed of nodes
class LineItem(QGraphicsPathItem):
    def __init__(self):
        super().__init__()
        self.nodes = []  # All nodes
        self.path_points = []  # All path points
        self.control_points = []  # User-added control nodes
        self.bezier_points = []  # Bezier curve control points
        
        pen = QPen(QColor('green'))
        pen.setWidth(2)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        self.setPen(pen)
        
        self.setFlags(QGraphicsItem.ItemIsSelectable)
        self.setAcceptHoverEvents(True)
        self.setZValue(1)


    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            menu = QMenu()
            add_control_point = menu.addAction("Add Control Point")
            delete_line = menu.addAction("Delete Line")
            
            action = menu.exec_(event.screenPos())
            if action == add_control_point:
                self.addNodeAtPosition(event.scenePos())
            elif action == delete_line:
                if self.scene() and self.scene().views():
                    parent_window = self.scene().views()[0].parent()
                    if hasattr(parent_window, 'delete_line'):
                        parent_window.delete_line(self)
        else:
            super().mousePressEvent(event)
    def addNodeAtPosition(self, pos):
        """Add a control point that affects the curve shape"""
        if len(self.path_points) < 2:
            return

        # Find closest segment
        min_dist = float('inf')
        insert_idx = 0
        proj_point = QPointF()

        for i in range(len(self.path_points) - 1):
            p1 = QPointF(self.path_points[i][0], self.path_points[i][1])
            p2 = QPointF(self.path_points[i+1][0], self.path_points[i+1][1])
            
            # Project point onto line segment
            line_vec = p2 - p1
            point_vec = pos - p1
            line_len = (line_vec.x() * line_vec.x() + line_vec.y() * line_vec.y())
            
            if line_len > 0:
                t = max(0, min(1, (point_vec.x() * line_vec.x() + point_vec.y() * line_vec.y()) / line_len))
                curr_proj = QPointF(p1.x() + t * line_vec.x(), p1.y() + t * line_vec.y())
                curr_dist = (pos - curr_proj).manhattanLength()
                
                if curr_dist < min_dist:
                    min_dist = curr_dist
                    insert_idx = i + 1
                    proj_point = curr_proj

        # Add control node
        node = NodeItem(proj_point.x(), proj_point.y())
        node.setZValue(2)
        node.setVisible(True)  # Make new control points visible
        self.scene().addItem(node)
        
        # Insert into nodes and control nodes
        self.nodes.insert(insert_idx, node)
        self.control_points.append(node)  # Add to control nodes list
        self.path_points.insert(insert_idx, (proj_point.x(), proj_point.y()))
        node.lines.append(self)
        
        # Update the curve
        self.updatePath()

        # Record action for undo
        if self.scene() and self.scene().views():
            parent_window = self.scene().views()[0].parent()
            if hasattr(parent_window, 'undo_stack'):
                parent_window.undo_stack.append(('add_control_node', node))
                parent_window.redo_stack.clear()

    def updateControlPoints(self):
        """Update Bezier control points when nodes change"""
        self.bezier_points = []
        
        if len(self.nodes) < 2:
            return
            
        for i in range(len(self.nodes) - 1):
            p1 = self.nodes[i].pos()
            p2 = self.nodes[i+1].pos()
            
            # Calculate control points at 1/3 and 2/3 distance
            dx = p2.x() - p1.x()
            dy = p2.y() - p1.y()
            
            ctrl1 = QPointF(p1.x() + dx/3, p1.y() + dy/3)
            ctrl2 = QPointF(p1.x() + dx*2/3, p1.y() + dy*2/3)
            
            self.bezier_points.append((ctrl1, ctrl2))

    def updatePath(self):
        """Update path using Bezier curves between control points"""
        if len(self.nodes) < 2:
            return
            
        self.updateControlPoints()
        
        path = QPainterPath()
        path.moveTo(self.nodes[0].pos())

        for i in range(len(self.nodes) - 1):
            p1 = self.nodes[i].pos()
            p2 = self.nodes[i+1].pos()
            ctrl1, ctrl2 = self.bezier_points[i]
            
            path.cubicTo(ctrl1, ctrl2, p2)

        self.setPath(path)

    def updateSimplePath(self):
        """Update path using straight lines between points"""
        if len(self.path_points) < 2:
            return
            
        path = QPainterPath()
        path.moveTo(*self.path_points[0])
        
        for point in self.path_points[1:]:
            path.lineTo(*point)
        
        self.setPath(path)
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
            QMessageBox.warning(self, "No Image", "Please apply a filter first.")
            return

        # Create an instance of EdgeLinkWindow
        self.edge_link_window = EdgeLinkWindow(self.filtered_image, parent=self)

        # Connect the edge_link_updated signal to update_filtered_image method
        self.edge_link_window.edge_link_updated.connect(self.update_filtered_image)

        # Show the EdgeLinkWindow modally
        self.edge_link_window.exec_()

    def update_filtered_image(self, updated_image):
        # Update the filtered_image with the updated image
        self.filtered_image = updated_image.copy()
        # Refresh the display
        self.show_filtered_image()

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
            # Apply global processing (from parent window)
            if isinstance(self.parent(), MyWindow):
                processed = self.parent().apply_global_processing(self.input_image)
            else:
                processed = self.input_image.copy()
                
            # Apply Gaussian filter if enabled
            if self.gaussian_checkbox.isChecked():
                sigma = self.gaussian_sigma.value() / 10.0
                processed = cv2.GaussianBlur(processed, (0, 0), sigma)

            # Apply the specific filter
            self.apply_filter(processed)

            # Apply skeletonization if enabled
            if self.skeletonize_checkbox.isChecked():
                self.filtered_image = self.apply_skeletonization(self.filtered_image)

            self.show_filtered_image()
    def get_main_window(self):
        """Helper method to get reference to main window"""
        parent = self.parent()
        while parent:
            if isinstance(parent, MyWindow):
                return parent
            parent = parent.parent()
        return None
    def prune_skeleton(self, skeleton, min_branch_length=10):
        """Remove small branches from skeleton while preserving main structure"""
        # Find endpoints and branch points
        kernel = np.array([[1, 1, 1],
                        [1, 10, 1],
                        [1, 1, 1]], dtype=np.uint8)
        
        skeleton_uint8 = skeleton.astype(np.uint8)
        filtered = cv2.filter2D(skeleton_uint8, -1, kernel)
        
        # Points with value > 11 are branch points (more than 1 neighbor)
        # Points with value == 11 are endpoints (exactly 1 neighbor)
        branch_points = filtered > 11
        endpoints = filtered == 11
        
        # Initialize output
        pruned = skeleton.copy()
        changes = True
        
        while changes:
            changes = False
            
            # Find current endpoints
            filtered = cv2.filter2D(pruned.astype(np.uint8), -1, kernel)
            current_endpoints = filtered == 11
            
            for y, x in zip(*np.where(current_endpoints)):
                # Check if this endpoint is part of a small branch
                visited = set()
                current = (y, x)
                branch = [current]
                
                while True:
                    # Get neighbors of current point
                    y, x = current
                    neighbors = []
                    
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            ny, nx = y + dy, x + dx
                            if (ny, nx) not in visited and 0 <= ny < pruned.shape[0] and 0 <= nx < pruned.shape[1]:
                                if pruned[ny, nx]:
                                    neighbors.append((ny, nx))
                    
                    if not neighbors:
                        break
                        
                    # If we hit a branch point or the branch is long enough, keep it
                    if len(neighbors) > 1 or len(branch) >= min_branch_length:
                        break
                        
                    visited.add(current)
                    current = neighbors[0]
                    branch.append(current)
                
                # Remove branch if it's too short and doesn't connect important points
                if len(branch) < min_branch_length:
                    for py, px in branch:
                        if not branch_points[py, px]:  # Don't remove branch points
                            pruned[py, px] = 0
                            changes = True
        
        return pruned

    def better_skeletonize(self, binary_image, distance_threshold=35):
        try:
            # Initial binary image processing - dark ridges are 0s
            _, binary_image = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)
            print(f"Initial binary sum: {np.sum(binary_image > 0)}")

            # Get dark ridges
            dark_ridges = cv2.bitwise_not(binary_image)
            
            # Distance transform for ridge detection
            dist_transform = cv2.distanceTransform(dark_ridges, cv2.DIST_L2, 5)
            max_width = np.max(dist_transform)
            
            # Normalize distance transform
            dist_transform = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # Threshold distance transform to get ridge cores
            _, ridge_cores = cv2.threshold(dist_transform, 50, 255, cv2.THRESH_BINARY)
            
            # Enhance ridges with morphology
            kernel_size = max(3, int(max_width * 0.2))
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            
            # Clean ridge cores
            ridge_cores = cv2.morphologyEx(ridge_cores, cv2.MORPH_CLOSE, kernel)
            ridge_cores = cv2.morphologyEx(ridge_cores, cv2.MORPH_OPEN, kernel)
            
            print(f"Ridge cores sum: {np.sum(ridge_cores > 0)}")
            
            # Skeletonize the ridge cores
            skeleton = thin(ridge_cores > 0)
            skeleton = skeleton.astype(np.uint8) * 255
            
            # Component filtering with reduced threshold
            retval, labels, stats, _ = cv2.connectedComponentsWithStats(skeleton, connectivity=8)
            final_skeleton = np.zeros_like(skeleton, dtype=np.uint8)
            
            # Very small minimum size to preserve detail
            min_size = max(2, int(max_width * 0.05))  # Reduced threshold
            
            for label in range(1, retval):
                if stats[label, cv2.CC_STAT_AREA] >= min_size:
                    final_skeleton[labels == label] = 255
                    
            print(f"Final skeleton sum: {np.sum(final_skeleton > 0)}")
            
            # Invert back for light ridges if needed
            if np.sum(binary_image > 0) > np.sum(binary_image == 0):
                final_skeleton = cv2.bitwise_not(final_skeleton)
                
            return final_skeleton

        except Exception as e:
            print(f"Better skeletonization error: {str(e)}")
            traceback.print_exc()
            return binary_image
    def apply_skeletonization(self, image):
        """Apply skeletonization with special handling for ridges"""
        try:
            # Get main window and current tab safely
            main_window = self.get_main_window()
            if main_window and hasattr(main_window, 'tab_widget'):
                current_tab = main_window.tab_widget.currentWidget()
                is_ridge_mode = isinstance(current_tab, FrangiFilterTab)
            else:
                is_ridge_mode = False

            if is_ridge_mode:
                # Ridge processing
                _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                is_dark_ridge = np.sum(binary == 0) > np.sum(binary == 255)
                
                if is_dark_ridge:
                    binary = cv2.bitwise_not(binary)
                
                # Use better skeletonization for ridges
                skeleton = self.better_skeletonize(binary, distance_threshold=35)
                
                if is_dark_ridge:
                    skeleton = cv2.bitwise_not(skeleton)
            else:
                # Standard edge processing
                skeleton = skeletonize(image > 0).astype(np.uint8) * 255
                
            return skeleton
            
        except Exception as e:
            print(f"Skeletonization error: {str(e)}")
            return image.copy()

    def open_edge_link_window(self):
        if self.filtered_image is None:
            QMessageBox.warning(self, "No Image", "Please apply a filter first.")
            return

        # Create an instance of EdgeLinkWindow
        self.edge_link_window = EdgeLinkWindow(self.filtered_image, parent=self)

        # Connect the edge_link_updated signal to update_filtered_image method
        self.edge_link_window.edge_link_updated.connect(self.update_filtered_image)

        # Show the EdgeLinkWindow modally
        self.edge_link_window.exec_()

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

        
        # Initialize eraser properties
        self.is_eraser_mode = False
        self.eraser_size = 10
        self.eraser_preview = None

        # Initialize Undo/Redo stacks
        self.undo_stack = deque()
        self.redo_stack = deque()
        # Add line width control
        self.line_width = 2  # Default width
        self.line_buffer = 0  # Default buffer
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

         # Add Hide Lines button after Hide Nodes button
        self.toggle_lines_button = QPushButton("Hide Lines")
        self.toggle_lines_button.setCheckable(True)
        self.toggle_lines_button.setChecked(True)  # Initially, lines are visible
        self.toggle_lines_button.clicked.connect(self.toggle_lines_visibility)
        button_layout.addWidget(self.toggle_lines_button)

        self.export_button = QPushButton("Export to Shapefile")
        self.export_button.clicked.connect(self.export_to_shapefile)
        layout.addWidget(self.export_button)
        # **Optional Enhancement: Undo and Redo Buttons**
        undo_redo_layout = QHBoxLayout()
        self.undo_button = QPushButton("Undo")
        self.undo_button.clicked.connect(self.undo_action)
        self.redo_button = QPushButton("Redo")
        self.redo_button.clicked.connect(self.redo_action)
        undo_redo_layout.addWidget(self.undo_button)
        undo_redo_layout.addWidget(self.redo_button)
        layout.addLayout(undo_redo_layout)

        # Create line width control layout
        line_control_layout = QHBoxLayout()
        
        # Width slider
        width_label = QLabel("Line Width:")
        self.width_slider = QSlider(Qt.Horizontal)
        self.width_slider.setMinimum(1)
        self.width_slider.setMaximum(10)
        self.width_slider.setValue(self.line_width)
        self.width_slider.valueChanged.connect(self.update_line_width)
        
        # Buffer slider
        buffer_label = QLabel("Line Buffer:")
        self.buffer_slider = QSlider(Qt.Horizontal)
        self.buffer_slider.setMinimum(0)
        self.buffer_slider.setMaximum(10)
        self.buffer_slider.setValue(self.line_buffer)
        self.buffer_slider.valueChanged.connect(self.update_line_buffer)
        
        # Add widgets to layout
        line_control_layout.addWidget(width_label)
        line_control_layout.addWidget(self.width_slider)
        line_control_layout.addWidget(buffer_label)
        line_control_layout.addWidget(self.buffer_slider)
        # Add to main layout
        layout.addLayout(line_control_layout)
        layout.addLayout(button_layout)
        # Add morphological controls layout
        morpho_layout = QHBoxLayout()
        
        # Merge parallel lines checkbox and slider
        self.merge_lines_checkbox = QCheckBox("Merge Parallel Lines")
        self.merge_lines_checkbox.stateChanged.connect(self.update_line_merging)
        
        merge_label = QLabel("Merge Distance:")
        self.merge_distance_slider = QSlider(Qt.Horizontal)
        self.merge_distance_slider.setMinimum(1)
        self.merge_distance_slider.setMaximum(50)
        self.merge_distance_slider.setValue(3)
        self.merge_distance_slider.setEnabled(False)  # Initially disabled
        self.merge_distance_slider.valueChanged.connect(self.update_line_merging)

        # Add Eraser button
        self.eraser_button = QPushButton("Enable Eraser")
        self.eraser_button.setCheckable(True)
        self.eraser_button.clicked.connect(self.toggle_eraser)
        button_layout.addWidget(self.eraser_button)

        # Add Eraser Size Slider
        eraser_control_layout = QHBoxLayout()
        eraser_label = QLabel("Eraser Size:")
        self.eraser_slider = QSlider(Qt.Horizontal)
        self.eraser_slider.setMinimum(5)
        self.eraser_slider.setMaximum(50)
        self.eraser_slider.setValue(self.eraser_size)
        self.eraser_slider.valueChanged.connect(self.update_eraser_size)
        
        eraser_control_layout.addWidget(eraser_label)
        eraser_control_layout.addWidget(self.eraser_slider)
        
        # Add to main layout before button_layout
        layout.addLayout(eraser_control_layout)

        
        morpho_layout.addWidget(self.merge_lines_checkbox)
        morpho_layout.addWidget(merge_label)
        morpho_layout.addWidget(self.merge_distance_slider)
        
        # Add to main layout before button_layout
        layout.addLayout(morpho_layout)
        self.setLayout(layout)

       # Install both event filters
        self.view.viewport().installEventFilter(self)
        self.view.viewport().setMouseTracking(True)
 
    # Add these new methods

    def toggle_eraser(self, checked):
        """Toggle eraser mode"""
        print("Eraser toggled:", checked)
        self.is_eraser_mode = checked
        self.is_drawing = False
        self.is_semi_auto = False
        self.is_edit_mode = False
        
        # Update button texts
        self.eraser_button.setText("Disable Eraser" if checked else "Enable Eraser")
        self.toggle_drawing_button.setText("Enable Manual Drawing")
        self.toggle_semi_auto_button.setText("Enable Semi-Auto Drawing")
        self.edit_mode_button.setText("Enter Edit Mode")
        
        # Update cursor
        if checked:
            self.view.viewport().setCursor(Qt.CrossCursor)
        else:
            self.view.viewport().setCursor(Qt.ArrowCursor)

    def update_eraser_size(self, value):
        """Update eraser size"""
        self.eraser_size = value

    def eraserEventFilter(self, source, event):
        """Handle eraser events with proper event type checking"""
        if source == self.view.viewport():
            # Get position based on event type
            if event.type() == QEvent.MouseMove:
                pos = event.pos()
                scene_pos = self.view.mapToScene(pos)
                self.update_eraser_preview(scene_pos.x(), scene_pos.y())
                
                if event.buttons() & Qt.LeftButton:
                    self.erase_at_point(scene_pos.x(), scene_pos.y())
                    return True
                    
            elif event.type() == QEvent.MouseButtonPress:
                if event.button() == Qt.LeftButton:
                    pos = event.pos()
                    scene_pos = self.view.mapToScene(pos)
                    self.erase_at_point(scene_pos.x(), scene_pos.y())
                    return True
                    
            elif event.type() == QEvent.MouseButtonRelease:
                return True
                
            elif event.type() == QEvent.Leave:
                if self.eraser_preview:
                    self.scene.removeItem(self.eraser_preview)
                    self.eraser_preview = None
                return True
                
        return False

    def erase_at_point(self, x, y):
        from PyQt5.QtGui import QPainterPathStroker
        eraser_region = QPainterPath()
        eraser_region.addEllipse(QPointF(x, y), self.eraser_size, self.eraser_size)

        lines_to_remove = []
        for line in self.lines[:]:
            if len(line.path_points) < 2:
                continue

            line_path = line.path()

            # Create a stroked path
            stroker = QPainterPathStroker()
            stroker.setWidth(self.line_width + self.line_buffer * 2)
            stroked_path = stroker.createStroke(line_path)

            # Print debug information
            print("Eraser center:", x, y)
            print("Line bounding rect:", line_path.boundingRect())
            print("Eraser bounding rect:", eraser_region.boundingRect())

            if eraser_region.intersects(stroked_path):
                print("Intersection detected with line:", line)
                lines_to_remove.append(line)
            else:
                print("No intersection for line:", line)

        for line in lines_to_remove:
            self.delete_line(line)

        if lines_to_remove:
            self.scene.update()


    def delete_line(self, line):
        """Delete a line and its nodes with proper cleanup"""
        try:
            # First remove all nodes
            for node in line.nodes[:]:
                self.scene.removeItem(node)
                node.lines.remove(line)
                
            # Remove line from scene and list
            self.scene.removeItem(line)
            if line in self.lines:
                self.lines.remove(line)
                
            # Add to undo stack
            self.undo_stack.append(('delete_line', line))
            self.redo_stack.clear()
            
        except Exception as e:
            print(f"Error in delete_line: {str(e)}")

    def update_eraser_preview(self, x, y):
        """Show preview of eraser area"""
        if not self.eraser_preview:
            # Create new preview
            self.eraser_preview = self.scene.addEllipse(
                x - self.eraser_size, 
                y - self.eraser_size,
                self.eraser_size * 2,
                self.eraser_size * 2,
                QPen(Qt.red)
            )
            self.eraser_preview.setZValue(1000)  # Keep on top
        else:
            # Update existing preview
            self.eraser_preview.setRect(
                x - self.eraser_size,
                y - self.eraser_size, 
                self.eraser_size * 2,
                self.eraser_size * 2
            )
    def update_line_width(self, value):
        """Update the width of all lines"""
        self.line_width = value
        self.update_all_lines()

    def update_line_buffer(self, value):
        """Update the buffer of all lines"""
        self.line_buffer = value
        self.update_all_lines()
    def export_to_shapefile(self):
        """Export the drawn lines to a shapefile with CRS and optionally create a geotiff copy"""
        try:
            # Create list of LineString geometries with y-coordinate flipped
            geometries = []
            for line in self.lines:
                if len(line.path_points) >= 2:
                    # Flip the y coordinates to match GeoTIFF orientation
                    height = self.parent().img.shape[0]
                    coords = [(float(x), float(height - y)) for x, y in line.path_points]
                    if len(coords) >= 2:
                        geometries.append(LineString(coords))

            if not geometries:
                QMessageBox.warning(self, "Export Error", "No valid lines to export!")
                return

            # Create GeoDataFrame
            gdf = gpd.GeoDataFrame(geometry=geometries)
            
            # Get parent window and image dimensions
            parent_window = self.parent()
            height, width = parent_window.img.shape[:2]

            if hasattr(parent_window, 'geotiff_crs') and parent_window.geotiff_crs:
                # Use original GeoTIFF CRS and transform
                gdf.set_crs(parent_window.geotiff_crs, inplace=True)
                used_crs = parent_window.geotiff_crs
                
                if hasattr(parent_window, 'geotiff_transform'):
                    def pixel_to_coords(geom):
                        if geom.is_empty:
                            return geom
                        coords = []
                        for x, y in geom.coords:
                            world_x, world_y = rasterio.transform.xy(
                                parent_window.geotiff_transform, y, x)
                            coords.append((world_x, world_y))
                        return LineString(coords)
                    
                    gdf['geometry'] = gdf['geometry'].apply(pixel_to_coords)
            else:
                # Create transform that preserves pixel coordinates
                transform = rasterio.transform.Affine(
                    1, 0, 0,    # x scaling, x shearing, x translation
                    0, -1, height    # y shearing, y scaling (negative), y translation
                )
                
                # Use simple projected CRS
                crs = rasterio.crs.CRS.from_string("+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")
                used_crs = crs
                
                # Set CRS for shapefile
                gdf.set_crs(crs, inplace=True)
                
                # Create a georeferenced copy of the original image
                if parent_window and hasattr(parent_window, 'img'):
                    try:
                        geotiff_path, _ = QFileDialog.getSaveFileName(
                            self,
                            "Save Georeferenced Image Copy",
                            "",
                            "GeoTIFF (*.tif)"
                        )
                        
                        if geotiff_path:
                            # Create GeoTIFF with same coordinate system
                            with rasterio.open(
                                geotiff_path,
                                'w',
                                driver='GTiff',
                                height=height,
                                width=width,
                                count=1,
                                dtype=parent_window.img.dtype,
                                crs=crs,
                                transform=transform,
                                nodata=0
                            ) as dst:
                                dst.write(parent_window.img, 1)
                                
                            QMessageBox.information(
                                self,
                                "Image Copy Created",
                                f"A georeferenced copy of the image has been saved to:\n{geotiff_path}"
                            )
                            
                    except Exception as e:
                        QMessageBox.warning(
                            self,
                            "Warning",
                            f"Failed to create georeferenced image copy: {str(e)}"
                        )

                # Save shapefile
                file_name, _ = QFileDialog.getSaveFileName(
                    self,
                    "Save Shapefile",
                    "",
                    "Shapefile (*.shp)"
                )

                if file_name:
                    if not file_name.lower().endswith('.shp'):
                        file_name += '.shp'

                    gdf.to_file(file_name)
                    
                    success_msg = f"Lines exported to {os.path.basename(file_name)}\nCRS: {used_crs}"
                    QMessageBox.information(
                        self,
                        "Export Successful",
                        success_msg
                    )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Export Error",
                f"Error exporting shapefile: {str(e)}"
            )
    def update_all_lines(self):
        """Update the appearance of all lines with current width and buffer settings"""
        for line in self.lines:
            # Update pen with new width
            pen = QPen(line.pen())
            pen.setWidth(self.line_width + (self.line_buffer * 2))
            line.setPen(pen)
            line.updatePath()
        self.scene.update()

    def update_line_merging(self):
        """Update line merging based on checkbox and slider"""
        if self.merge_lines_checkbox.isChecked():
            self.merge_distance_slider.setEnabled(True)
            self.merge_parallel_lines(self.merge_distance_slider.value())
        else:
            self.merge_distance_slider.setEnabled(False)
            self.display_filtered_lines()  # Reset to original lines

    def merge_parallel_lines(self, max_distance):
        """Merge smaller parallel lines into main fracture lines"""
        if not self.lines:
            return
                
        # Convert lines to numpy arrays and calculate lengths
        line_arrays = []
        for line in self.lines:
            points = np.array(line.path_points)
            if len(points) >= 2:
                # Calculate line length
                length = np.sum(np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1)))
                line_arrays.append((line, points, length))
        
        # Sort lines by length to identify main fracture lines
        line_arrays.sort(key=lambda x: x[2], reverse=True)
        
        # Separate main lines and subsidiary lines
        length_threshold = np.mean([x[2] for x in line_arrays])  # Use mean length as threshold
        main_lines = [(line, points) for line, points, length in line_arrays if length >= length_threshold]
        subsidiary_lines = [(line, points) for line, points, length in line_arrays if length < length_threshold]
        
        merged = []
        merged_lines = set()
        
        # For each main line, find and merge nearby subsidiary lines
        for main_line, main_points in main_lines:
            if main_line in merged_lines:
                continue
                
            parallel_group = [main_points]
            merged_subsidiaries = []
            
            # Find subsidiary lines to merge
            for sub_line, sub_points in subsidiary_lines:
                if sub_line in merged_lines:
                    continue
                    
                if self.are_lines_parallel(main_points, sub_points) and \
                self.get_line_distance(main_points, sub_points) < max_distance:
                    parallel_group.append(sub_points)
                    merged_subsidiaries.append(sub_line)
                    merged_lines.add(sub_line)
            
            if merged_subsidiaries:
                # Merge lines while preserving main line position
                merged_points = self.merge_preserving_main(main_points, parallel_group)
                merged_line = self.create_merged_line(merged_points)
                merged.append(merged_line)
                
                # Remove merged subsidiary lines
                for line in merged_subsidiaries:
                    if line in self.lines:
                        self.delete_line(line)
                
                merged_lines.add(main_line)
                if main_line in self.lines:
                    self.delete_line(main_line)
            else:
                merged.append(main_line)
        
        # Add remaining unmerged subsidiary lines
        for sub_line, _ in subsidiary_lines:
            if sub_line not in merged_lines:
                merged.append(sub_line)
        
        # Update lines list
        self.lines = merged
        self.show_overlay()
    def merge_preserving_main(self, main_points, all_points):
        """Merge lines while preserving main line position"""
        # Calculate weights based on distance from main line
        weights = []
        for points in all_points:
            dist = self.get_line_distance(main_points, points)
            weight = 1.0 / (1.0 + dist)  # Higher weight for closer lines
            weights.append(weight)
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # Create float array for calculations
        merged_points = np.zeros_like(main_points, dtype=np.float64)
        
        # Combine points with weighted averaging
        for i, points in enumerate(all_points):
            # Interpolate points to match main line length
            if len(points) != len(main_points):
                t = np.linspace(0, 1, len(points))
                t_new = np.linspace(0, 1, len(main_points))
                points = np.array([np.interp(t_new, t, points[:, 0]),
                                np.interp(t_new, t, points[:, 1])]).T
            merged_points += weights[i] * points.astype(np.float64)
        
        # Convert back to integers at the end
        return np.round(merged_points).astype(np.int32)
    def are_lines_parallel(self, points1, points2, angle_threshold=20):
        """Check if two lines are roughly parallel with safety checks"""
        # Get primary directions
        dir1 = points1[-1] - points1[0]
        dir2 = points2[-1] - points2[0]
        
        # Calculate vector lengths
        norm1 = np.linalg.norm(dir1)
        norm2 = np.linalg.norm(dir2)
        
        # Check for zero-length vectors
        if norm1 < 1e-6 or norm2 < 1e-6:
            return False
        
        # Safely normalize directions
        dir1_normalized = dir1 / norm1
        dir2_normalized = dir2 / norm2
        
        # Calculate angle between directions with numerical stability
        dot_product = np.clip(np.dot(dir1_normalized, dir2_normalized), -1.0, 1.0)
        angle = np.abs(np.arccos(dot_product))
        angle_deg = np.degrees(angle)
        
        # Check if lines are parallel within threshold
        return angle_deg < angle_threshold or angle_deg > (180 - angle_threshold)


    def get_line_distance(self, points1, points2):
        """Get minimum distance between two line segments with validation"""
        if len(points1) < 2 or len(points2) < 2:
            return float('inf')
            
        min_dist = float('inf')
        
        # Sample points along both lines
        for p1 in points1:
            for p2 in points2:
                if np.any(np.isnan(p1)) or np.any(np.isnan(p2)):
                    continue
                dist = np.linalg.norm(p1 - p2)
                if not np.isnan(dist):
                    min_dist = min(min_dist, dist)
        
        return min_dist

    def merge_point_sets(self, points1, points2):
        """Merge two sets of points into a single line"""
        # Combine points
        combined = np.vstack((points1, points2))
        
        # Fit line to combined points
        vx, vy, x0, y0 = cv2.fitLine(combined, cv2.DIST_L2, 0, 0.01, 0.01)
        
        # Project points onto fitted line
        direction = np.array([vx[0], vy[0]])
        point = np.array([x0[0], y0[0]])
        
        # Get extent of line
        dots = np.dot(combined - point, direction)
        min_t = np.min(dots)
        max_t = np.max(dots)
        
        # Create new line points
        num_points = max(len(points1), len(points2))
        t = np.linspace(min_t, max_t, num_points)
        merged_points = point + direction * t.reshape(-1, 1)
        
        return merged_points.astype(np.int32)

    def create_merged_line(self, points):
        """Create a new LineItem from merged points"""
        line = LineItem()
        line.setZValue(1)
        pen = QPen(QColor('green'), self.line_width + (self.line_buffer * 2))
        line.setPen(pen)
        
        # Add points
        for x, y in points:
            line.path_points.append((int(x), int(y)))
        
        # Update path
        line.updateSimplePath()
        self.scene.addItem(line)
        
        return line
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
        """Display filtered lines preserving exact edge positions"""
        # Clear existing lines
        for line in self.lines:
            for node in line.nodes:
                self.scene.removeItem(node)
            self.scene.removeItem(line)
        self.lines.clear()

        # Create binary image
        _, binary = cv2.threshold(self.filtered_image, 127, 255, cv2.THRESH_BINARY)

        if hasattr(self.parent(), 'tab_widget'):
            current_tab = self.parent().tab_widget.currentWidget()
            is_dog_filter = isinstance(current_tab, FrangiFilterTab)
        else:
            is_dog_filter = False

        # Extract and convert edges/ridges to lines
        if is_dog_filter:
            lines = self.extract_ridge_lines(binary)
        else:
            lines = self.extract_edge_lines(binary)

        # Add lines to scene
        for line_points in lines:
            if len(line_points) >= 2:
                self.add_line(line_points, is_ridge=is_dog_filter)

        self.edge_map = binary
    def toggle_lines_visibility(self, checked):
        """Toggle visibility of all lines"""
        if checked:
            self.toggle_lines_button.setText("Hide Lines")
        else:
            self.toggle_lines_button.setText("Show Lines")

        for line in self.lines:
            line.setVisible(checked)
    def extract_edge_lines(self, binary_image):
        """Extract edges preserving exact shape and position"""
        try:
            # Find contours with more precise method
            contours, _ = cv2.findContours(
                binary_image, 
                cv2.RETR_LIST,
                cv2.CHAIN_APPROX_NONE  # Get ALL contour points
            )
            
            lines = []
            min_length = 1  # Minimum line length to keep
            
            for contour in contours:
                # Convert contour to points
                points = [tuple(point[0]) for point in contour]
                
                if len(points) >= min_length:
                    # Remove duplicate points while preserving shape
                    points = self.remove_duplicates(points)
                    # Add the entire contour as a single line
                    lines.append(points)

            return lines
            
        except Exception as e:
            print(f"Error in edge extraction: {str(e)}")
            return []
    def extract_ridge_lines(self, binary_image):
        """Extract ridge lines preserving exact shape and position for DoG filter output"""
        try:
            # Use the same precise method as in extract_edge_lines
            contours, _ = cv2.findContours(
                binary_image,
                cv2.RETR_LIST,
                cv2.CHAIN_APPROX_NONE  # Get all contour points
            )

            lines = []
            min_length = 1  # Set minimum length to 1 to include all lines

            for contour in contours:
                # Convert contour to points
                points = [tuple(point[0]) for point in contour]

                if len(points) >= min_length:
                    # Remove duplicate points while preserving shape
                    points = self.remove_duplicates(points)
                    # Add the entire contour as a single line
                    lines.append(points)

            return lines

        except Exception as e:
            print(f"Error in ridge line extraction: {str(e)}")
            return []
    def simplify_line(self, points, tolerance=5.0):
        """Simplify line using Douglas-Peucker algorithm"""
        if len(points) < 3:
            return points

        # Convert to numpy array
        points = np.array(points)

        # Recursively simplify the line
        simplified_points = self.douglas_peucker(points, tolerance)
        return simplified_points.tolist()
    def douglas_peucker(self, points, tolerance):
        """Recursive implementation of the Douglas-Peucker algorithm"""
        # Douglas-Peucker algorithm is an algorithm for reducing the number of points in a curve, useful for simplification
        if len(points) < 3:
            return points

        # Line between first and last point
        start, end = points[0], points[-1]
        line_vec = end - start
        line_len_sq = np.sum(line_vec ** 2)

        # Find the point with the maximum distance from the line
        distances = np.abs(np.cross(line_vec, points - start)) / np.sqrt(line_len_sq)
        max_dist_idx = np.argmax(distances)
        max_dist = distances[max_dist_idx]

        if max_dist > tolerance:
            # Recursive call
            left = self.douglas_peucker(points[:max_dist_idx+1], tolerance)
            right = self.douglas_peucker(points[max_dist_idx:], tolerance)
            return np.vstack((left[:-1], right))
        else:
            return np.vstack((start, end))
    def remove_duplicates(self, points):
        """Remove duplicate points while preserving line shape"""
        if len(points) < 2:
            return points
                
        result = [points[0]]
        for i in range(1, len(points)):
            # Compare points using numpy's array equality check
            if not np.array_equal(points[i], points[i-1]):
                result.append(points[i])
                
        return np.array(result)
    def find_next_ridge_point(self, image, current, visited):
        """Find next connected ridge point"""
        y, x = current
        
        # Check 8-connected neighbors in order of priority
        neighbors = [
            (-1,0), (1,0), (0,-1), (0,1),  # Direct neighbors first
            (-1,-1), (-1,1), (1,-1), (1,1)  # Diagonals second
        ]
        
        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            if ((ny, nx) not in visited and 
                0 <= ny < image.shape[0] and 
                0 <= nx < image.shape[1] and
                image[ny, nx] > 0):
                return (ny, nx)
                
        return None
    def split_at_corners(self, points, angle_threshold=45):
        """Split line at sharp corners"""
        if len(points) < 3:
            return [points]
            
        segments = []
        current_segment = [points[0]]
        
        for i in range(1, len(points)-1):
            current_segment.append(points[i])
            
            # Calculate angle
            v1 = np.array(points[i]) - np.array(points[i-1])
            v2 = np.array(points[i+1]) - np.array(points[i])
            
            if np.any(v1) and np.any(v2):
                angle = np.abs(np.degrees(
                    np.arctan2(np.cross(v1, v2), np.dot(v1, v2))
                ))
                
                # Split at sharp corners
                if angle < angle_threshold:
                    if len(current_segment) >= 2:
                        segments.append(current_segment)
                    current_segment = [points[i]]
                    
        if len(current_segment) >= 2:
            segments.append(current_segment)
            
        return segments
    def split_into_segments(self, points, angle_threshold=60):
        """Split polyline into segments at sharp angles."""
        if len(points) < 3:
            return [points]
            
        segments = []
        current_segment = [points[0]]
        
        for i in range(1, len(points) - 1):
            current_segment.append(points[i])
            
            # Calculate angle between segments
            v1 = np.array(points[i]) - np.array(points[i-1])
            v2 = np.array(points[i+1]) - np.array(points[i])
            
            if np.any(v1) and np.any(v2):  # Avoid zero vectors
                angle = np.abs(np.degrees(np.arctan2(np.cross(v1, v2), np.dot(v1, v2))))
                
                # Split at sharp angles
                if angle < angle_threshold and len(current_segment) > 2:
                    segments.append(current_segment)
                    current_segment = [points[i]]
                    
        current_segment.append(points[-1])
        if len(current_segment) > 2:
            segments.append(current_segment)
            
        return segments
    def smooth_line(self, points, num_points=None):
        """Smooth a line using cubic spline interpolation."""
        if len(points) < 3:
            return points
            
        points = np.array(points)
        
        # Calculate path length for each point
        t = np.zeros(len(points))
        for i in range(1, len(points)):
            t[i] = t[i-1] + np.linalg.norm(points[i] - points[i-1])
        
        if num_points is None:
            num_points = len(points)
        
        # Create smooth interpolated curve
        t_new = np.linspace(0, t[-1], num_points)
        
        # Fit cubic spline
        cs = scipy.interpolate.CubicSpline(t, points)
        smooth_points = cs(t_new)
        
        return [(int(x), int(y)) for x, y in smooth_points]
    def split_at_high_curvature(self, points, angle_threshold=45):
        """Split line at points of high curvature."""
        if len(points) < 3:
            return [points]

        segments = []
        current_segment = [points[0]]
        
        for i in range(1, len(points) - 1):
            current_segment.append(points[i])
            
            # Calculate angle between segments
            v1 = np.array(points[i]) - np.array(points[i-1])
            v2 = np.array(points[i+1]) - np.array(points[i])
            
            if len(v1) > 0 and len(v2) > 0:
                angle = np.abs(np.degrees(
                    np.arctan2(np.cross(v1, v2), np.dot(v1, v2))
                ))
                
                # Split at high curvature points
                if angle > angle_threshold and len(current_segment) > 2:
                    segments.append(current_segment)
                    current_segment = [points[i]]

        current_segment.append(points[-1])
        if len(current_segment) > 2:
            segments.append(current_segment)

        return segments
    def add_line(self, points, is_ridge=False):
        """Add a line with initial points but no visible control points"""
        if is_ridge:
            reduced_points = self.remove_duplicates(points)
        else:
            simplified_points = self.simplify_line(points, tolerance=0.5)
            cleaned_points = self.merge_close_points(simplified_points, threshold=2)
            reduced_points = self.get_control_points(cleaned_points)

         # Create the line with current width settings
        line = LineItem()
        line.setZValue(1)
        pen = QPen(QColor('green'), self.line_width + (self.line_buffer * 2))
        line.setPen(pen)
        self.scene.addItem(line)
        
        # Add initial points (not as control points)
        for x, y in reduced_points:
            node = NodeItem(x, y)
            node.setZValue(2)
            node.setVisible(False)  # Initially hidden
            self.scene.addItem(node)
            line.nodes.append(node)  # Add to all nodes
            node.lines.append(line)
            line.path_points.append((x, y))
        
        # Update path
        line.updatePath()
        self.lines.append(line)

        # Record action for undo
        self.undo_stack.append(('add_line', line))
        self.redo_stack.clear()

    def get_control_points(self, points, min_angle=30, min_dist=10):
        """
        Get strategic control points that define the line's shape:
        - Start and end points
        - Points of significant direction change
        - Points at regular intervals for long straight segments
        """
        if len(points) < 3:
            return points
            
        control_points = [points[0]]  # Always include start point
        current_angle = 0
        last_control = 0
        
        for i in range(1, len(points) - 1):
            # Calculate angle change
            v1 = np.array(points[i]) - np.array(points[i-1])
            v2 = np.array(points[i+1]) - np.array(points[i])
            
            if np.any(v1) and np.any(v2):
                # Calculate angle between vectors
                dot = np.dot(v1, v2)
                norm = np.linalg.norm(v1) * np.linalg.norm(v2)
                if norm != 0:
                    cos_angle = dot / norm
                    cos_angle = min(1.0, max(-1.0, cos_angle))
                    angle = np.degrees(np.arccos(cos_angle))
                    
                    # Add point if:
                    # 1. Significant angle change
                    # 2. Long distance from last control point
                    dist_from_last = np.linalg.norm(np.array(points[i]) - np.array(points[last_control]))
                    
                    if angle > min_angle or dist_from_last > min_dist:
                        control_points.append(points[i])
                        last_control = i
                        current_angle = 0
                    else:
                        current_angle += angle

        control_points.append(points[-1])  # Always include end point
        
        # Add intermediate points for long segments
        final_points = []
        for i in range(len(control_points) - 1):
            final_points.append(control_points[i])
            # If segment is too long, add intermediate control point
            dist = np.linalg.norm(np.array(control_points[i+1]) - np.array(control_points[i]))
            if dist > min_dist * 3:
                # Add midpoint
                mid = (np.array(control_points[i]) + np.array(control_points[i+1])) / 2
                final_points.append(tuple(mid.astype(int)))
                
        final_points.append(control_points[-1])
        
        return final_points

    def merge_close_points(self, points, threshold=2):
        """Merge points that are very close to each other."""
        if len(points) < 2:
            return points

        result = [points[0]]
        for i in range(1, len(points)):
            if euclidean_dist(points[i], points[i-1]) > threshold:
                result.append(points[i])
        return result
    def reduce_points(self, points):
        """
        Reduce the number of points to only include significant ones:
        - Start point
        - End point
        - Points where direction changes significantly
        """
        if len(points) <= 2:
            return points

        result = [points[0]]  # Always include start point
        
        # Angle threshold for determining significant direction changes (in degrees)
        angle_threshold = 45
        
        for i in range(1, len(points) - 1):
            # Calculate vectors
            v1 = np.array(points[i]) - np.array(points[i-1])
            v2 = np.array(points[i+1]) - np.array(points[i])
            
            # Calculate angle between vectors
            dot_product = np.dot(v1, v2)
            norms = np.linalg.norm(v1) * np.linalg.norm(v2)
            
            if norms != 0:  # Avoid division by zero
                cos_angle = dot_product / norms
                cos_angle = min(1.0, max(-1.0, cos_angle))  # Ensure value is in [-1, 1]
                angle = np.degrees(np.arccos(cos_angle))
                
                # If angle is significant, include this point
                if angle < (180 - angle_threshold) or angle > (180 + angle_threshold):
                    result.append(points[i])
        
        result.append(points[-1])  # Always include end point
        return result
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
        self.display_filtered_lines()

    # **Existing Method: Toggle Nodes Visibility**
    def toggle_nodes_visibility(self, checked):
        """Toggle visibility only for manually added control points"""
        if checked:
            self.toggle_nodes_button.setText("Hide Control Points")
        else:
            self.toggle_nodes_button.setText("Show Control Points")

        for line in self.lines:
            # Only toggle visibility for manually added control points
            for node in line.control_points:
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
            # Make sure lines are selectable
            line.setFlag(QGraphicsItem.ItemIsSelectable, True)
            
            # Update line appearance
            if line.isSelected():
                line.setPen(QPen(QColor('red'), 3))  # Highlight selected lines
            else:
                line.setPen(QPen(QColor('green'), 2))  # Normal appearance

            # Update nodes
            for node in line.nodes:
                if self.is_edit_mode:
                    node.setFlags(
                        QGraphicsItem.ItemIsMovable |
                        QGraphicsItem.ItemSendsGeometryChanges |
                        QGraphicsItem.ItemIsSelectable
                    )
                else:
                    node.setFlags(QGraphicsItem.ItemIsSelectable)
                
                # Update node appearance
                if node.isSelected():
                    node.setBrush(QColor('yellow'))
                else:
                    node.setBrush(QColor('blue'))

        # Refresh the scene
        self.scene.update()

    def eventFilter(self, source, event):
        """Handle drawing and semi-auto events"""
        # Handle eraser events first if eraser mode is active
        if self.is_eraser_mode:
            return self.eraserEventFilter(source, event)
            
        # Original drawing/semi-auto event handling
        if not (self.is_drawing or self.is_semi_auto):
            return super().eventFilter(source, event)
            
        if event.type() == QEvent.MouseButtonPress:
            scene_pos = self.view.mapToScene(event.pos())
            x, y = int(scene_pos.x()), int(scene_pos.y())
            
            # Handle manual drawing
            if self.is_drawing:
                if event.button() == Qt.LeftButton:
                    if not self.current_line:
                        self.current_line = [(x, y)]
                        self.add_manual_line(x, y)
                    else:
                        self.current_line.append((x, y))
                        self.add_manual_line(x, y)
                elif event.button() == Qt.RightButton:
                    if self.current_line:
                        self.current_line = []
                        self.current_line_item = None
                        
            # Handle semi-auto drawing
            elif self.is_semi_auto:
                if event.button() == Qt.LeftButton:
                    if not self.semi_auto_start_point:
                        self.semi_auto_start_point = (x, y)
                        QMessageBox.information(self, "Semi-Auto Drawing",
                                            "Start point set. Click again to set end point.")
                    else:
                        end_point = (x, y)
                        self.semi_automatic_tracking(self.semi_auto_start_point, end_point)
                        self.semi_auto_start_point = None
            return True
            
        return super().eventFilter(source, event)


    def add_manual_line(self, x, y):
        """Improved manual line drawing with continuous path"""
        x, y = int(x), int(y)  # Ensure integer coordinates
        
        if len(self.current_line) == 1:
            # Create new line on first click
            self.current_line_item = LineItem()
            self.current_line_item.setZValue(1)
            pen = QPen(QColor('green'), self.line_width + (self.line_buffer * 2))
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            self.current_line_item.setPen(pen)
            self.scene.addItem(self.current_line_item)
            
            # Store first point
            self.current_line_item.path_points = [(x, y)]
            
            # Add to lines list immediately
            self.lines.append(self.current_line_item)
            
            # Update path
            self.current_line_item.updateSimplePath()
            
            # Record for undo
            self.undo_stack.append(('add_line', self.current_line_item))
            self.redo_stack.clear()
            
        else:
            # Add new point
            self.current_line_item.path_points.append((x, y))
            
            # Update path
            self.current_line_item.updateSimplePath()
            
            # Record for undo
            self.undo_stack.append(('update_line', (self.current_line_item, [(x, y)])))
            self.redo_stack.clear()

        # Ensure the line is selectable and can show context menu
        self.current_line_item.setFlag(QGraphicsItem.ItemIsSelectable)
        self.current_line_item.setAcceptHoverEvents(True)
    def semi_automatic_tracking(self, start_point, end_point):
        """Semi-automatic line tracking along edges using A* pathfinding"""
        # Create cost map from edge_map - inverse the values so edges have lower cost
        cost_map = np.where(self.edge_map > 0, 1, 255)  # Edges = low cost (1), non-edges = high cost (255)
        
        # Swap x,y coordinates since the image coordinates are (y,x)
        start = (int(start_point[1]), int(start_point[0]))
        goal = (int(end_point[1]), int(end_point[0]))
        
        # Find path using modified A* that prefers edge pixels
        path = self.a_star(cost_map, start, goal)

        if path:
            # Convert path back to (x,y) coordinates and create line
            tracked_points = [(int(point[1]), int(point[0])) for point in path]
            
            # Create new line item with current width settings
            line = LineItem()
            line.setZValue(1)
            pen = QPen(QColor('green'), self.line_width + (self.line_buffer * 2))
            line.setPen(pen)
            self.scene.addItem(line)
            
            # Create path for the line
            path = QPainterPath()
            first_point = tracked_points[0]
            path.moveTo(first_point[0], first_point[1])
            
            # Add all points to the path
            for point in tracked_points[1:]:
                path.lineTo(point[0], point[1])
            
            # Set the path to the line
            line.setPath(path)
            
            # Store path points for future reference
            line.path_points = tracked_points
            
            # Add to lines list and record for undo
            self.lines.append(line)
            self.undo_stack.append(('add_line', line))
            self.redo_stack.clear()
        else:
            QMessageBox.warning(self, "Path Not Found",
                            "Could not find a valid path between points.")

    def a_star(self, cost_map, start, goal):
        """Modified A* pathfinding that prefers edge pixels"""
        def heuristic(a, b):
            return np.hypot(b[0] - a[0], b[1] - a[1])

        def get_neighbors(point):
            y, x = point
            # Check 8-connected neighbors
            for dy, dx in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < cost_map.shape[0] and 0 <= nx < cost_map.shape[1]:
                    yield (ny, nx)

        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}

        while frontier:
            current = heapq.heappop(frontier)[1]
            
            if current == goal:
                break

            for next_pos in get_neighbors(current):
                # Add edge preference to movement cost
                edge_cost = cost_map[next_pos]
                new_cost = cost_so_far[current] + edge_cost
                
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + heuristic(goal, next_pos)
                    heapq.heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = current

        # Reconstruct path
        if goal not in came_from:
            return None
            
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path


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
        try:
            # First remove all nodes
            for node in line.nodes[:]:  # Use a copy of the list
                if node in line.nodes:  # Check if node is still in the list
                    node.lines.remove(line)  # Remove line reference from node
                    if not node.lines:  # If node has no more lines, remove it
                        self.scene.removeItem(node)

            # Then remove the line itself
            if line in self.lines:
                self.lines.remove(line)
                self.scene.removeItem(line)

            # Record action for undo
            self.undo_stack.append(('delete_line', line))
            self.redo_stack.clear()

            # Update display
            self.scene.update()
            self.show_overlay()

        except Exception as e:
            print(f"Error in delete_line: {str(e)}")
            traceback.print_exc()
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
from coshrem.shearletsystem import RidgeSystem
class ShearletFilterTab(FilterTab):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.shearlet_system = None  # Will hold RidgeSystem instance
        self.setWindowTitle("Ridge (Shearlet-based)")
        self.setup_controls()

    def setup_controls(self):
        # Minimum contrast slider
        self.min_contrast = QSlider(Qt.Horizontal)
        self.min_contrast.setRange(0, 100)
        self.min_contrast.setValue(1)
        self.min_contrast.valueChanged.connect(self.update_filter)
        self.controls_layout.addWidget(QLabel("Min Contrast"))
        self.controls_layout.addWidget(self.min_contrast)

        # Threshold slider for ridgeness map
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 255)
        self.threshold_slider.setValue(10)  # Default threshold
        self.threshold_slider.valueChanged.connect(self.update_filter)
        self.controls_layout.addWidget(QLabel("Binary Threshold"))
        self.controls_layout.addWidget(self.threshold_slider)

        # Minimum fracture size slider
        self.min_size_slider = QSlider(Qt.Horizontal)
        self.min_size_slider.setRange(1, 2000)
        self.min_size_slider.setValue(500)
        self.min_size_slider.valueChanged.connect(self.update_filter)
        self.controls_layout.addWidget(QLabel("Min Fracture Size (px)"))
        self.controls_layout.addWidget(self.min_size_slider)

        # Pruning checkbox and slider
        self.prune_checkbox = QCheckBox("Apply Pruning")
        self.prune_checkbox.stateChanged.connect(self.update_filter)
        self.controls_layout.addWidget(self.prune_checkbox)

        self.prune_slider = QSlider(Qt.Horizontal)
        self.prune_slider.setRange(1, 100)  # Adjust range as needed # higher value means 
        self.prune_slider.setValue(10)  # Default minimum branch length
        self.prune_slider.valueChanged.connect(self.update_filter)
        self.controls_layout.addWidget(QLabel("Prune Branch Length"))
        self.controls_layout.addWidget(self.prune_slider)

    def set_input_image(self, image):
        super().set_input_image(image)
        # Create a RidgeSystem with similar parameters as the original image size
        rows, cols = image.shape
        self.shearlet_system = RidgeSystem(rows, cols)  # Use defaults or modify as needed

    def apply_filter(self, image):
        if self.shearlet_system is None:
            self.filtered_image = image.copy()
            return

        # Get parameters
        min_contrast_val = self.min_contrast.value()
        binary_thresh = self.threshold_slider.value()
        min_size = self.min_size_slider.value()
        prune_length = self.prune_slider.value()
        do_prune = self.prune_checkbox.isChecked()

        # Detect ridges using RidgeSystem
        ridgeness, orientations = self.shearlet_system.detect(
            image,
            min_contrast=min_contrast_val,
            positive_only=False,
            negative_only=False,
            pivoting_scales='lowest'  # 'all', 'highest' can also be tried
        )

        # Normalize ridgeness to 0-255
        ridgeness_norm = cv2.normalize(ridgeness, None, 0, 255, cv2.NORM_MINMAX)
        ridgeness_uint8 = ridgeness_norm.astype(np.uint8)

        # Apply binary threshold
        _, binary = cv2.threshold(ridgeness_uint8, binary_thresh, 255, cv2.THRESH_BINARY)

        # Remove small objects that are not significant fractures
        binary = self.remove_small_objects(binary, min_size=min_size)

        if do_prune:
            # Skeletonize
            skeleton = skeletonize(binary > 0).astype(np.uint8) * 255
            # Prune skeleton
            skeleton_pruned = self.prune_skeleton(skeleton, min_branch_length=prune_length)
            self.filtered_image = skeleton_pruned
        else:
            # No pruning, just show the binary result
            self.filtered_image = binary


    def remove_small_objects(self, binary_image, min_size=500):
        """
        Remove small connected components from binary_image.
        Components smaller than min_size pixels are removed.
        """
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, 8, cv2.CV_32S)
        output_mask = np.zeros_like(binary_image)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_size:
                output_mask[labels == i] = 255
        return output_mask

    def prune_skeleton(self, skeleton, min_branch_length=10):
        """
        Prune small skeleton components by removing any connected component 
        smaller than 'min_branch_length' pixels.
        """
        # skeleton is assumed to be a binary image (0 or 255)
        # Ensure binary format
        _, sk_bin = cv2.threshold(skeleton, 127, 255, cv2.THRESH_BINARY)

        # Label connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(sk_bin, connectivity=8)

        # Create output mask initialized to zeros
        pruned = np.zeros_like(sk_bin, dtype=np.uint8)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            # Keep only components that are at least min_branch_length pixels long
            if area >= min_branch_length:
                pruned[labels == i] = 255

        return pruned

    def open_manual_interpretation(self):
        pass


class RidgeEnsembleGenerator:
    """Class for generating ridge ensembles using DoG filter"""
    def __init__(self):
        self.params = {
            'sigma1_range': (1.0, 5.0, 0.5),  # (min, max, step) for smaller sigma
            'sigma2_range': (5.0, 15.0, 1.0),  # (min, max, step) for larger sigma
            'threshold_range': (0.1, 0.9, 0.1)  # (min, max, step) for thresholding
        }

    def generate_ensemble(self, image):
        ensemble = np.zeros_like(image, dtype=float)
        count = 0
        
        # Iterate through all parameter combinations
        for sigma1 in np.arange(*self.params['sigma1_range']):
            for sigma2 in np.arange(*self.params['sigma2_range']):
                for thresh in np.arange(*self.params['threshold_range']):
                    # Apply DoG filter
                    g1 = cv2.GaussianBlur(image, (0,0), sigma1)
                    g2 = cv2.GaussianBlur(image, (0,0), sigma2)
                    dog = cv2.subtract(g1, g2)
                    
                    # Detect ridges
                    ridge_response = self.compute_ridge_measure(dog)
                    
                    # Threshold ridge response
                    binary_response = (ridge_response > thresh * np.max(ridge_response)).astype(float)
                    
                    # Add to ensemble
                    ensemble += binary_response
                    count += 1
                    
        # Normalize ensemble
        return (ensemble / count * 255).astype(np.uint8)

    def compute_ridge_measure(self, image):
        """Compute ridge measure using Hessian matrix analysis"""
        try:
            # Convert image to float64
            img = image.astype(np.float64)
            
            # Compute Hessian matrix elements using correct Sobel parameters
            # First derivatives
            Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
            Iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
            
            # Second derivatives
            Ixx = cv2.Sobel(Ix, cv2.CV_64F, 1, 0, ksize=3)
            Iyy = cv2.Sobel(Iy, cv2.CV_64F, 0, 1, ksize=3)
            Ixy = cv2.Sobel(Ix, cv2.CV_64F, 0, 1, ksize=3)
            
            # Compute eigenvalues
            # For ridges: |λ2| >> |λ1| ≈ 0
            delta = np.sqrt((Ixx - Iyy)**2 + 4*Ixy**2)
            lambda1 = (Ixx + Iyy + delta) / 2
            lambda2 = (Ixx + Iyy - delta) / 2
            
            # Ridge measure (Frangi's vesselness measure)
            ridge_measure = np.zeros_like(img)
            
            # Parameters for ridge measure
            beta = 0.5
            c = 0.5 * np.max(np.abs(lambda2))
            
            # Ridge criteria and measure
            mask = np.abs(lambda2) > np.abs(lambda1)
            if c > 0:  # Prevent division by zero
                ridge_measure[mask] = np.exp(-np.abs(lambda1[mask])/(beta*np.abs(lambda2[mask]))) * \
                                    (1 - np.exp(-lambda2[mask]**2/(2*c**2)))
            
            # Normalize output
            ridge_measure = cv2.normalize(ridge_measure, None, 0, 1, cv2.NORM_MINMAX)
            
            return ridge_measure
            
        except Exception as e:
            print(f"Error in compute_ridge_measure: {str(e)}")
            return np.zeros_like(image, dtype=np.float64)

class FrangiFilterTab(FilterTab):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ridge Detection (Frangi)")
        self.ensemble_generator = RidgeEnsembleGenerator()
        # self.create_filter_controls()

    def create_filter_controls(self):
        # Group box for Frangi parameters
        ridge_group = QGroupBox("Frangi Filter Parameters")
        ridge_layout = QVBoxLayout()

        # Scale Range Min
        scale_min_layout = QHBoxLayout()
        scale_min_label = QLabel("Scale Min:")
        self.scale_min_slider = QSlider(Qt.Horizontal)
        self.scale_min_slider.setRange(1, 50)  # Adjust as needed
        self.scale_min_slider.setValue(1)  # Default
        self.scale_min_slider.valueChanged.connect(self.update_filter)
        self.scale_min_value_label = QLabel("1")
        scale_min_layout.addWidget(scale_min_label)
        scale_min_layout.addWidget(self.scale_min_slider)
        scale_min_layout.addWidget(self.scale_min_value_label)
        ridge_layout.addLayout(scale_min_layout)

        # Scale Range Max
        scale_max_layout = QHBoxLayout()
        scale_max_label = QLabel("Scale Max:")
        self.scale_max_slider = QSlider(Qt.Horizontal)
        self.scale_max_slider.setRange(2, 100)  # Adjust as needed
        self.scale_max_slider.setValue(20)  # Default
        self.scale_max_slider.valueChanged.connect(self.update_filter)
        self.scale_max_value_label = QLabel("20")
        scale_max_layout.addWidget(scale_max_label)
        scale_max_layout.addWidget(self.scale_max_slider)
        scale_max_layout.addWidget(self.scale_max_value_label)
        ridge_layout.addLayout(scale_max_layout)

        # Scale Step
        scale_step_layout = QHBoxLayout()
        scale_step_label = QLabel("Scale Step:")
        self.scale_step_slider = QSlider(Qt.Horizontal)
        self.scale_step_slider.setRange(1, 10)  # Adjust as needed
        self.scale_step_slider.setValue(1)  # Default
        self.scale_step_slider.valueChanged.connect(self.update_filter)
        self.scale_step_value_label = QLabel("1")
        scale_step_layout.addWidget(scale_step_label)
        scale_step_layout.addWidget(self.scale_step_slider)
        scale_step_layout.addWidget(self.scale_step_value_label)
        ridge_layout.addLayout(scale_step_layout)

        # Alpha
        alpha_layout = QHBoxLayout()
        alpha_label = QLabel("Alpha:")
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(1, 50)  # Scale 0.1 to 5.0
        self.alpha_slider.setValue(5)  # Corresponds to 0.5 if scaled by 10
        self.alpha_slider.valueChanged.connect(self.update_filter)
        self.alpha_value_label = QLabel("0.5")
        alpha_layout.addWidget(alpha_label)
        alpha_layout.addWidget(self.alpha_slider)
        alpha_layout.addWidget(self.alpha_value_label)
        ridge_layout.addLayout(alpha_layout)

        # Beta
        beta_layout = QHBoxLayout()
        beta_label = QLabel("Beta:")
        self.beta_slider = QSlider(Qt.Horizontal)
        self.beta_slider.setRange(1, 50)  # Scale similarly as alpha
        self.beta_slider.setValue(5)  # Default 0.5
        self.beta_slider.valueChanged.connect(self.update_filter)
        self.beta_value_label = QLabel("0.5")
        beta_layout.addWidget(beta_label)
        beta_layout.addWidget(self.beta_slider)
        beta_layout.addWidget(self.beta_value_label)
        ridge_layout.addLayout(beta_layout)

        # Gamma
        gamma_layout = QHBoxLayout()
        gamma_label = QLabel("Gamma:")
        self.gamma_slider = QSlider(Qt.Horizontal)
        self.gamma_slider.setRange(1, 50)  # from 1 to 50
        self.gamma_slider.setValue(15)  # Default
        self.gamma_slider.valueChanged.connect(self.update_filter)
        self.gamma_value_label = QLabel("15")
        gamma_layout.addWidget(gamma_label)
        gamma_layout.addWidget(self.gamma_slider)
        gamma_layout.addWidget(self.gamma_value_label)
        ridge_layout.addLayout(gamma_layout)

        # Ensemble generation button (as before)
        self.ensemble_button = QPushButton("Generate Ridge Ensemble")
        self.ensemble_button.clicked.connect(self.generate_ridge_ensemble)
        ridge_layout.addWidget(self.ensemble_button)

        ridge_group.setLayout(ridge_layout)
        self.controls_layout.addWidget(ridge_group)

    def apply_filter(self, image):
        if image is None:
            return

        from skimage.filters import frangi, frangi
        
        # Read parameters from sliders
        scale_min = self.scale_min_slider.value()
        scale_max = self.scale_max_slider.value()
        # Ensure scale_min < scale_max
        if scale_min >= scale_max:
            scale_max = scale_min + 1

        scale_step = self.scale_step_slider.value()
        alpha = self.alpha_slider.value() / 10.0  # converting to float, e.g., slider=5 -> alpha=0.5
        beta = self.beta_slider.value() / 10.0
        gamma = float(self.gamma_slider.value())

        # Update labels
        self.scale_min_value_label.setText(str(scale_min))
        self.scale_max_value_label.setText(str(scale_max))
        self.scale_step_value_label.setText(str(scale_step))
        self.alpha_value_label.setText(f"{alpha:.1f}")
        self.beta_value_label.setText(f"{beta:.1f}")
        self.gamma_value_label.setText(str(int(gamma)))

        # Convert image to float for Frangi
        img_float = img_as_float(image)

        # Apply Frangi filter with user-selected parameters
        frangi_response = frangi(
            img_float, 
            scale_range=(scale_min, scale_max),
            scale_step=scale_step,
            alpha=alpha,
            beta=beta,
            gamma=gamma
        )

        # Rescale the Frangi output to 0-255 for easier thresholding
        from skimage.exposure import rescale_intensity
        frangi_norm = rescale_intensity(frangi_response, out_range=(0,255)).astype(np.uint8)

        # Perform Otsu thresholding to separate ridges from background
        ret, otsu_img = cv2.threshold(frangi_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # If Otsu is too strict, lower the threshold slightly
        threshold_value = max(ret - 10, 0)
        _, binary = cv2.threshold(frangi_norm, threshold_value, 255, cv2.THRESH_BINARY)

        # Mild morphological closing to connect nearby ridge segments
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, morph_kernel, iterations=1)

        # Remove very small objects if needed
        binary = self.remove_small_objects(binary, min_size=10)

        self.filtered_image = binary
        self.show_filtered_image()

    def remove_small_objects(self, binary_image, min_size=500):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, 8, cv2.CV_32S)
        output_mask = np.zeros_like(binary_image)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_size:
                output_mask[labels == i] = 255
        return output_mask

    def generate_ridge_ensemble(self):
        if self.input_image is None:
            return
            
        # Adjust parameters or keep as is
        self.ensemble_generator.params = {
            'sigma1_range': (1.5, 2.5, 0.5),
            'sigma2_range': (8.0, 12.0, 2.0),
            'threshold_range': (0.1, 0.9, 0.2)
        }

        ensemble = self.ensemble_generator.generate_ensemble(self.input_image)
        self.filtered_image = ensemble
        self.show_filtered_image()
    
class EdgeLinkWindow(QDialog):
    edge_link_updated = pyqtSignal(np.ndarray)
    
    def __init__(self, image, parent=None):
        super().__init__(parent)
        self.image = image
        self.edge_linked_image = None
        self.parent = parent
        self.edge_linker = None
        self.processed_edges = [] 
        self.segment_list = []
        self.initUI()
        
    def create_slider(self, name, min_val, max_val, default_val, tooltip_text):
        group = QGroupBox(name)
        layout = QVBoxLayout()
        
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(default_val)
        slider.setToolTip(tooltip_text)  # Set tooltip for the slider
        
        value_label = QLabel(str(default_val))
        value_label.setToolTip(tooltip_text)  # Set tooltip for the label
        slider.valueChanged.connect(lambda v: value_label.setText(str(v)))
        slider.valueChanged.connect(self.process_and_display)
        
        layout.addWidget(slider)
        layout.addWidget(value_label)
        group.setLayout(layout)
        group.setToolTip(tooltip_text)  # Set tooltip for the group box
        
        return group

    def initUI(self):
        self.setWindowTitle('Geological Fracture Edge Link')
        self.setGeometry(100, 100, 800, 800)
        layout = QVBoxLayout()

        # Create matplotlib figure
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Controls group
        controls_group = QGroupBox("Edge Detection Parameters")
        controls_layout = QVBoxLayout()

        # Sliders with tooltips
        self.min_length_slider = self.create_slider(
            "Minimum Fracture Length", 5, 100, 20,
            "Sets the minimum length for detected fractures. Increasing this value filters out shorter fractures."
        )
        self.max_gap_slider = self.create_slider(
            "Maximum Fracture Gap", 1, 30, 5,
            "Defines the maximum allowed gap between fracture segments when merging. Larger values allow larger gaps."
        )
        self.min_angle_slider = self.create_slider(
            "Minimum Junction Angle", 15, 165, 45,
            "Specifies the minimum angle at junctions between fracture segments. Adjust to control merging based on angle."
        )
        self.straightness_slider = self.create_slider(
            "Fracture Straightness", 1, 20, 5,
            "Controls the straightness threshold for filtering fractures. Lower values keep straighter fractures."
        )
        self.segment_tol_slider = self.create_slider(
            "Segmentation Tolerance", 1, 50, 10,
            "Sets the tolerance for segmenting fractures. Lower values result in more segments."
        )
        # Add sliders to layout
        for control in [self.min_length_slider, self.max_gap_slider, 
                       self.min_angle_slider, self.straightness_slider, 
                       self.segment_tol_slider]:
            controls_layout.addWidget(control)

        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)

        # Buttons
        button_layout = QHBoxLayout()
        self.segment_button = QPushButton("Segment Fractures")
        self.segment_button.clicked.connect(self.segment_fractures)
        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self.apply_to_main)
        
        button_layout.addWidget(self.segment_button)
        button_layout.addWidget(self.apply_button)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        self.process_and_display()

    def process_and_display(self):
        if self.image is None:
            return

        # Get slider values
        min_length = self.min_length_slider.findChild(QSlider).value()
        max_gap = self.max_gap_slider.findChild(QSlider).value()
        min_angle = self.min_angle_slider.findChild(QSlider).value()
        straightness = self.straightness_slider.findChild(QSlider).value()

        # Process image without transposing
        self.edge_linker = edgelink(self.image, min_length)
        self.edge_linker.get_edgelist()
        
        # Process edges with original orientation
        edge_lists = [np.array(edge) for edge in self.edge_linker.edgelist if len(edge) > 0]
        filtered_edges = self.filter_by_straightness(edge_lists, straightness)
        processed_edges = self.process_geological_edges(filtered_edges, max_gap, min_angle)
        
        # Save for later use
        self.processed_edges = processed_edges
        
        # Create and display results
        self.edge_linked_image = self.create_edge_image(processed_edges)
        self.visualize_edges(processed_edges)

    def filter_by_straightness(self, edge_lists, straightness_threshold):
        filtered_edges = []
        for edge in edge_lists:
            if len(edge) >= 3:
                total_dev = 0
                for i in range(1, len(edge)-1):
                    dev = self.point_to_line_distance(edge[i], edge[0], edge[-1])
                    total_dev += dev
                avg_dev = total_dev / (len(edge) - 2)
                if avg_dev < straightness_threshold:
                    filtered_edges.append(edge)
        return filtered_edges

    def point_to_line_distance(self, point, line_start, line_end):
        if np.array_equal(line_start, line_end):
            return np.linalg.norm(point - line_start)
            
        line_vec = line_end - line_start
        point_vec = point - line_start
        line_len = np.linalg.norm(line_vec)
        line_unitvec = line_vec / line_len
        point_proj_len = np.dot(point_vec, line_unitvec)
        
        if point_proj_len < 0:
            return np.linalg.norm(point_vec)
        elif point_proj_len > line_len:
            return np.linalg.norm(point - line_end)
        else:
            point_proj = line_start + line_unitvec * point_proj_len
            return np.linalg.norm(point - point_proj)

    def process_geological_edges(self, edge_lists, max_gap, min_angle):
        processed_edges = []
        
        for edge in edge_lists:
            if len(edge) < 3:
                continue
                
            # Split into segments
            segments = []
            current_segment = [edge[0]]
            
            for i in range(1, len(edge) - 1):
                current_segment.append(edge[i])
                v1 = edge[i] - edge[i-1]
                v2 = edge[i+1] - edge[i]
                
                if np.any(v1) and np.any(v2):
                    # Convert 2D vectors to 3D for cross product
                    v1_3d = np.array([v1[0], v1[1], 0])
                    v2_3d = np.array([v2[0], v2[1], 0])
                    angle = np.abs(np.degrees(
                        np.arctan2(np.linalg.norm(np.cross(v1_3d, v2_3d)), np.dot(v1, v2))
                    ))
                    if angle < min_angle:
                        if len(current_segment) >= 2:
                            segments.append(np.array(current_segment))
                        current_segment = [edge[i]]
            
            current_segment.append(edge[-1])
            if len(current_segment) >= 2:
                segments.append(np.array(current_segment))
                
            # Merge segments
            i = 0
            while i < len(segments):
                j = i + 1
                while j < len(segments):
                    seg1 = segments[i]
                    seg2 = segments[j]
                    dist = np.linalg.norm(seg1[-1] - seg2[0])
                    
                    if dist <= max_gap:
                        v1 = seg1[-1] - seg1[-2]
                        v2 = seg2[1] - seg2[0]
                        angle = np.abs(np.degrees(
                            np.arctan2(np.cross(v1, v2), np.dot(v1, v2))
                        ))
                        if angle >= min_angle:
                            segments[i] = np.vstack((seg1, seg2))
                            segments.pop(j)
                            continue
                    j += 1
                i += 1
                
            processed_edges.extend(segments)
            
        return processed_edges

    def create_edge_image(self, edges):
        height, width = self.image.shape[:2]
        edge_image = np.zeros((height, width), dtype=np.uint8)
        
        for edge in edges:
            points = edge.astype(np.int32)
            for i in range(len(points)-1):
                # Swap x,y coordinates for OpenCV line drawing
                p1 = (int(points[i][1]), int(points[i][0]))  # Swap coordinates
                p2 = (int(points[i+1][1]), int(points[i+1][0]))  # Swap coordinates
                cv2.line(edge_image, p1, p2, 255, 1)
        
        return edge_image

    
    def visualize_edges(self, edges):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Display image with correct orientation
        ax.imshow(self.image, cmap='gray', alpha=0.5, origin='upper')
        
        # Use a colormap to represent some property, e.g., edge length
        lengths = [len(edge) for edge in edges]
        norm = plt.Normalize(vmin=min(lengths), vmax=max(lengths))
        cmap = plt.cm.jet
        
        for edge in edges:
            length = len(edge)
            color = cmap(norm(length))
            ax.plot(edge[:, 1], edge[:, 0], '-', color=color, linewidth=2)
            
        ax.axis('off')
        # Add a colorbar to represent edge lengths
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = self.figure.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Edge Length')
        self.canvas.draw()

    def segment_fractures(self):
        if self.processed_edges:
            tol = self.segment_tol_slider.findChild(QSlider).value()
            # Use the existing 'seglist' function
            self.segment_list = seglist(self.processed_edges, tol)
            self.visualize_segments()
            # Update the edge-linked image to include segments
            self.edge_linked_image = self.create_edge_image(self.segment_list)

    def visualize_segments(self):
        if not self.segment_list:
            return
                
        self.figure.clear()
        ax = self.figure.add_subplot(111)
            
        # Display image with correct orientation
        ax.imshow(self.image, cmap='gray', alpha=0.5, origin='upper')
            
        # Use a colormap to represent some property, e.g., segment length
        lengths = [len(segment) for segment in self.segment_list]
        norm = plt.Normalize(vmin=min(lengths), vmax=max(lengths))
        cmap = plt.cm.jet
        
        for segment in self.segment_list:
            length = len(segment)
            color = cmap(norm(length))
            ax.plot(segment[:, 1], segment[:, 0], '-', color=color, linewidth=2)
                
        ax.axis('off')
        # Add a colorbar to represent segment lengths
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = self.figure.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Segment Length')
        self.canvas.draw()

    def apply_to_main(self):
        if self.segment_list:
            # Use the segmented edges to create the final image
            self.edge_linked_image = self.create_edge_image(self.segment_list)
        else:
            # Ensure the image is processed with the latest parameters
            self.process_and_display()
            # edge_linked_image is already updated in process_and_display()

        if self.edge_linked_image is not None:
            # Emit the signal with the updated image
            self.edge_link_updated.emit(self.edge_linked_image)
            self.accept()  # Close the dialog

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
        self.add_filter_tab("Frangi", FrangiFilterTab)  # Add Frangi tab
        # Add binarization controls below clean edges layout
        binarize_layout = QHBoxLayout()
        
        # Binarization checkbox
        self.binarize_checkbox = QCheckBox("Apply Binarization")
        self.binarize_checkbox.stateChanged.connect(self.update_all_filters)
        binarize_layout.addWidget(self.binarize_checkbox)
        
        # Threshold slider
        self.binary_threshold = QSlider(Qt.Horizontal)
        self.binary_threshold.setRange(0, 255)
        self.binary_threshold.setValue(127)
        self.binary_threshold.valueChanged.connect(self.update_all_filters)
        
        # Threshold value label
        self.threshold_label = QLabel("127")
        
        binarize_layout.addWidget(QLabel("Threshold:"))
        binarize_layout.addWidget(self.binary_threshold)
        binarize_layout.addWidget(self.threshold_label)
        
        main_layout.addLayout(binarize_layout)
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
    def update_all_filters(self):
        """Update all filter tabs when global settings change"""
        if self.binarize_checkbox.isChecked():
            threshold = self.binary_threshold.value()
            self.threshold_label.setText(str(threshold))
        
        # Update all filter tabs
        for i in range(self.tab_widget.count() - 1):  # Exclude the "+" tab
            tab = self.tab_widget.widget(i)
            if isinstance(tab, FilterTab):
                tab.update_filter()

    def apply_global_processing(self, image):
        """Apply global image processing settings before filter-specific processing"""
        processed = image.copy()
        
        # Apply binarization if enabled
        if self.binarize_checkbox.isChecked():
            threshold = self.binary_threshold.value()
            _, processed = cv2.threshold(processed, threshold, 255, cv2.THRESH_BINARY)
        
        return processed
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
                "HED",  # Add HED to the list
                "Frangi"  # Add Frangi to the list
            ]
            filter_name, ok = QInputDialog.getItem(self, "Select Filter", "Choose a filter:", filters, 0, False)
            if ok and filter_name:
                filter_classes = {
                    "Canny": CannyFilterTab,
                    "Sobel": SobelFilterTab,
                    "Shearlet": ShearletFilterTab,
                    "Laplacian": LaplacianFilterTab,
                    "Roberts": RobertsFilterTab,
                    "HED": HEDFilterTab,  # Map "HED" to HEDFilterTab
                    "Frangi": FrangiFilterTab  # Map "Frangi" to Frangi
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
            
            # Ensure binary image without skeletonization
            _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
            
            # Label connected components
            labeled, num_features = measure.label(binary_image > 0, return_num=True, connectivity=2)
            
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
            # Initialize geospatial attributes
            self.geotiff_crs = None
            self.geotiff_transform = None
            
            # Check if file is a GeoTIFF
            if img_path.lower().endswith(('.tif', '.tiff')):
                try:
                    # Open with rasterio to get geospatial information
                    with rasterio.open(img_path) as dataset:
                        # Read image data
                        img = dataset.read(1)  # Read first band
                        
                        # Store geospatial information
                        self.geotiff_crs = dataset.crs
                        self.geotiff_transform = dataset.transform
                        
                        print(f"Loaded GeoTIFF with CRS: {self.geotiff_crs}")
                        
                except Exception as e:
                    print(f"Failed to read GeoTIFF: {str(e)}")
                    # Fallback to regular image reading
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            else:
                # Regular image reading for non-TIFF formats
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

                    # Update all filter tabs
                    for i in range(self.tab_widget.count() - 1):
                        tab = self.tab_widget.widget(i)
                        if isinstance(tab, FilterTab):
                            tab.set_input_image(self.img)

                    # Enable buttons
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
