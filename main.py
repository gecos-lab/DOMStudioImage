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

    def open_manual_interpretation(self):
        if self.filtered_image is not None:
            self.manual_interpretation_window = ManualInterpretationWindow(self.filtered_image, self.parent())
            self.manual_interpretation_window.show()

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
            return
        self.edge_link_window = EdgeLinkWindow(self.filtered_image, self.parent())
        self.edge_link_window.show()

    def apply_filter(self, image):
        # To be implemented in subclasses
        pass

    def apply_skeletonization(self, image):
        return skeletonize(image > 0).astype(np.uint8) * 255
class ManualInterpretationWindow(QDialog):
    def __init__(self, original_image, filtered_image, parent=None):
        super().__init__(parent)
        self.original_image = original_image
        self.filtered_image = filtered_image
        self.lines = []
        self.current_line = []
        self.is_drawing = False
        self.is_semi_auto = False
        self.is_edit_mode = False
        self.semi_auto_start_point = None
        self.edge_map = self.create_edge_map()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Manual Interpretation')
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self.toggle_drawing_button = QPushButton("Enable Manual Drawing")
        self.toggle_drawing_button.clicked.connect(self.toggle_drawing)
        layout.addWidget(self.toggle_drawing_button)

        self.toggle_semi_auto_button = QPushButton("Enable Semi-Auto Drawing")
        self.toggle_semi_auto_button.clicked.connect(self.toggle_semi_auto)
        layout.addWidget(self.toggle_semi_auto_button)

        self.edit_mode_button = QPushButton("Enter Edit Mode")
        self.edit_mode_button.clicked.connect(self.toggle_edit_mode)
        layout.addWidget(self.edit_mode_button)

        self.setLayout(layout)

        self.canvas.mpl_connect("button_press_event", self.on_canvas_click)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.canvas.mpl_connect("button_release_event", self.on_mouse_release)

        self.show_image()

    def show_image(self):
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.ax.imshow(self.image, cmap='gray')
        self.ax.axis('off')
        self.canvas.draw()

    def toggle_drawing(self):
        self.is_drawing = not self.is_drawing
        self.is_semi_auto = False
        self.is_edit_mode = False
        self.semi_auto_start_point = None
        self.toggle_drawing_button.setText("Disable Manual Drawing" if self.is_drawing else "Enable Manual Drawing")
        self.toggle_semi_auto_button.setText("Enable Semi-Auto Drawing")
        self.edit_mode_button.setText("Enter Edit Mode")

    def toggle_semi_auto(self):
        self.is_semi_auto = not self.is_semi_auto
        self.is_drawing = False
        self.is_edit_mode = False
        self.semi_auto_start_point = None
        self.toggle_semi_auto_button.setText("Disable Semi-Auto Drawing" if self.is_semi_auto else "Enable Semi-Auto Drawing")
        self.toggle_drawing_button.setText("Enable Manual Drawing")
        self.edit_mode_button.setText("Enter Edit Mode")

    def toggle_edit_mode(self):
        self.is_edit_mode = not self.is_edit_mode
        self.is_drawing = False
        self.is_semi_auto = False
        if self.is_edit_mode:
            self.toggle_drawing_button.setEnabled(False)
            self.toggle_semi_auto_button.setEnabled(False)
            self.edit_mode_button.setText("Exit Edit Mode")
            QMessageBox.information(self, "Edit Mode", "Click near a line point to edit. Click 'Exit Edit Mode' when done.")
        else:
            self.toggle_drawing_button.setEnabled(True)
            self.toggle_semi_auto_button.setEnabled(True)
            self.edit_mode_button.setText("Enter Edit Mode")

    def on_canvas_click(self, event):
        if event.inaxes != self.ax:
            return

        x, y = int(event.xdata), int(event.ydata)

        if self.is_drawing:
            if event.button == 1:  # Left click
                if not self.current_line:
                    self.current_line = [(x, y)]
                else:
                    self.current_line.append((x, y))
                    self.draw_lines()
            elif event.button == 3:  # Right click
                if self.current_line:
                    self.lines.append(self.current_line)
                    self.current_line = []
                    self.draw_lines()
        elif self.is_semi_auto:
            if not self.semi_auto_start_point:
                self.semi_auto_start_point = (x, y)
                QMessageBox.information(self, "Semi-Auto Drawing", "Start point set. Click again to set end point.")
            else:
                end_point = (x, y)
                self.semi_automatic_tracking(self.semi_auto_start_point, end_point)
                self.semi_auto_start_point = None
        elif self.is_edit_mode:
            self.edit_nearest_point(x, y)

    def on_mouse_move(self, event):
        if not self.is_drawing or not self.current_line or event.inaxes != self.ax:
            return

        x, y = int(event.xdata), int(event.ydata)
        temp_line = self.current_line + [(x, y)]
        self.draw_lines(temp_line)

    def on_mouse_release(self, event):
        if self.is_drawing and event.button == 1 and self.current_line:  # Left click release
            x, y = int(event.xdata), int(event.ydata)
            self.current_line.append((x, y))
            self.draw_lines()

    def draw_lines(self, temp_line=None):
        self.show_image()
        for line in self.lines:
            if line and len(line) > 1:
                x, y = zip(*line)
                self.ax.plot(x, y, 'r-')
        if self.current_line and len(self.current_line) > 1:
            x, y = zip(*self.current_line)
            self.ax.plot(x, y, 'r-')
        if temp_line and len(temp_line) > 1:
            x, y = zip(*temp_line)
            self.ax.plot(x, y, 'r--')
        self.canvas.draw()

    def semi_automatic_tracking(self, start_point, end_point):
        # Convert points to image coordinates
        start = (int(start_point[1]), int(start_point[0]))
        end = (int(end_point[1]), int(end_point[0]))

        # Use A* algorithm to find the path
        path = self.a_star(self.edge_map, start, end)

        if path:
            # Convert path back to display coordinates
            tracked_line = [(x, y) for y, x in path]
            self.lines.append(tracked_line)
            self.draw_lines()
        else:
            QMessageBox.warning(self, "Path Not Found", "Could not find a path between the selected points. Try selecting closer points or adjusting the filter settings.")
    def create_edge_map(self):
        # Use the filtered image to create the edge map
        blurred_image = gaussian_filter(self.filtered_image, sigma=2)
        edges = feature.canny(blurred_image, sigma=2)
        return 1 - edges

    def a_star(self, cost_map, start, goal):
        def heuristic(a, b):
            return np.hypot(b[0] - a[0], b[1] - a[1])

        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

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
                if 0 <= neighbor[0] < cost_map.shape[0] and 0 <= neighbor[1] < cost_map.shape[1]:
                    if neighbor in close_set:
                        continue
                    tentative_g_score = gscore[current] + cost_map[neighbor]
                    if tentative_g_score < gscore.get(neighbor, np.inf):
                        came_from[neighbor] = current
                        gscore[neighbor] = tentative_g_score
                        fscore[neighbor] = gscore[neighbor] + heuristic(neighbor, goal)
                        heapq.heappush(oheap, (fscore[neighbor], neighbor))

        return None
    def edit_nearest_point(self, x, y):
        min_distance = float('inf')
        nearest_line = None
        nearest_point_index = None

        for i, line in enumerate(self.lines):
            for j, point in enumerate(line):
                distance = np.sqrt((x - point[0])**2 + (y - point[1])**2)
                if distance < min_distance:
                    min_distance = distance
                    nearest_line = i
                    nearest_point_index = j

        if nearest_line is not None and min_distance < 10:  # Threshold for selection
            self.lines[nearest_line][nearest_point_index] = (x, y)
            self.draw_lines()

    def draw_lines(self, temp_line=None):
        self.show_image()
        for line in self.lines:
            if line and len(line) > 1:
                x, y = zip(*line)
                self.ax.plot(x, y, 'r-')
        if self.current_line and len(self.current_line) > 1:
            x, y = zip(*self.current_line)
            self.ax.plot(x, y, 'r-')
        if temp_line and len(temp_line) > 1:
            x, y = zip(*temp_line)
            self.ax.plot(x, y, 'r--')
        self.canvas.draw()

    def show_image(self):
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.ax.imshow(self.original_image, cmap='gray')
        self.ax.imshow(self.edge_map, cmap='jet', alpha=0.3)  # Overlay edge map
        self.ax.axis('off')
        self.canvas.draw()
class CannyFilterTab(FilterTab):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_controls()

    def setup_controls(self):
        # Threshold sliders
        self.threshold1 = QSlider(Qt.Horizontal)
        self.threshold1.setRange(0, 255)
        self.threshold1.setValue(50)
        self.threshold1.valueChanged.connect(self.update_filter)

        self.threshold2 = QSlider(Qt.Horizontal)
        self.threshold2.setRange(0, 255)
        self.threshold2.setValue(150)
        self.threshold2.valueChanged.connect(self.update_filter)

        self.controls_layout.addWidget(QLabel("Threshold 1"))
        self.controls_layout.addWidget(self.threshold1)
        self.controls_layout.addWidget(QLabel("Threshold 2"))
        self.controls_layout.addWidget(self.threshold2)

        # Manual #

    def apply_filter(self, image):
        self.filtered_image = cv2.Canny(image, self.threshold1.value(), self.threshold2.value())

    def open_manual_interpretation(self):
        if self.filtered_image is not None:
            self.manual_interpretation_window = ManualInterpretationWindow(self.filtered_image, self.parent())
            self.manual_interpretation_window.show()

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
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
        self.filtered_image = cv2.addWeighted(cv2.convertScaleAbs(grad_x), 0.5,
                                              cv2.convertScaleAbs(grad_y), 0.5, 0)

    def open_manual_interpretation(self):
        if self.filtered_image is not None:
            self.manual_interpretation_window = ManualInterpretationWindow(self.filtered_image, self.parent())
            self.manual_interpretation_window.show()

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
        if self.filtered_image is not None:
            self.manual_interpretation_window = ManualInterpretationWindow(self.filtered_image, self.parent())
            self.manual_interpretation_window.show()
class EdgeLinkWindow(QDialog):
    def __init__(self, image, parent=None):
        super().__init__(parent)
        self.image = image
        self.edge_linked_image = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Edge Link Visualization')
        self.setGeometry(100, 100, 800, 800)

        layout = QVBoxLayout()

        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Add sliders for edgelink parameters
        self.min_length_slider = self.create_slider("Minimum Edge Length", 1, 50, 10)
        layout.addWidget(self.min_length_slider)

        self.max_gap_slider = self.create_slider("Maximum Gap", 1, 20, 2)
        layout.addWidget(self.max_gap_slider)

        self.min_angle_slider = self.create_slider("Minimum Angle", 0, 90, 20)
        layout.addWidget(self.min_angle_slider)

        # Add Apply button
        self.apply_button = QPushButton("Apply Edge Link")
        self.apply_button.clicked.connect(self.apply_edge_link)
        layout.addWidget(self.apply_button)

        self.setLayout(layout)

    def create_slider(self, name, min_val, max_val, default_val):
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel(name))
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default_val)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval((max_val - min_val) // 10)
        slider_layout.addWidget(slider)
        value_label = QLabel(str(default_val))
        slider_layout.addWidget(value_label)
        slider.valueChanged.connect(lambda v: value_label.setText(str(v)))

        widget = QWidget()
        widget.setLayout(slider_layout)
        return widget

    def apply_edge_link(self):
        minilength = self.min_length_slider.findChild(QSlider).value()
        max_gap = self.max_gap_slider.findChild(QSlider).value()
        min_angle = self.min_angle_slider.findChild(QSlider).value()

        edge_linker = edgelink(self.image, minilength)
        edge_linker.get_edgelist()
        edge_lists = [np.array(edge) for edge in edge_linker.edgelist if len(edge) > 0]

        # Post-process edge lists based on max_gap and min_angle
        processed_edge_lists = self.post_process_edges(edge_lists, max_gap, min_angle)

        self.visualize_edge_lists(processed_edge_lists)

    def post_process_edges(self, edge_lists, max_gap, min_angle):
        # Implement post-processing logic here
        return edge_lists

    def visualize_edge_lists(self, edge_lists):
        self.edge_linked_image = np.zeros(self.image.shape, dtype=np.uint8)
        for edge in edge_lists:
            for point in edge:
                if 0 <= point[0] < self.edge_linked_image.shape[0] and 0 <= point[1] < self.edge_linked_image.shape[1]:
                    self.edge_linked_image[int(point[0]), int(point[1])] = 255

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.imshow(self.edge_linked_image, cmap='gray')
        ax.axis('off')
        self.canvas.draw()



class EdgeLinkWindow(QDialog):
    def __init__(self, image, parent=None):
        super().__init__(parent)
        self.image = image
        self.edge_linked_image = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Edge Link Visualization')
        self.setGeometry(100, 100, 800, 800)

        layout = QVBoxLayout()

        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Add sliders for edgelink parameters
        self.min_length_slider = self.create_slider("Minimum Edge Length", 1, 50, 10)
        layout.addWidget(self.min_length_slider)

        self.max_gap_slider = self.create_slider("Maximum Gap", 1, 20, 2)
        layout.addWidget(self.max_gap_slider)

        self.min_angle_slider = self.create_slider("Minimum Angle", 0, 90, 20)
        layout.addWidget(self.min_angle_slider)

        # Add Apply button
        self.apply_button = QPushButton("Apply Edge Link")
        self.apply_button.clicked.connect(self.apply_edge_link)
        layout.addWidget(self.apply_button)

        self.setLayout(layout)

    def create_slider(self, name, min_val, max_val, default_val):
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel(name))
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default_val)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval((max_val - min_val) // 10)
        slider_layout.addWidget(slider)
        value_label = QLabel(str(default_val))
        slider_layout.addWidget(value_label)
        slider.valueChanged.connect(lambda v: value_label.setText(str(v)))

        widget = QWidget()
        widget.setLayout(slider_layout)
        return widget

    def apply_edge_link(self):
        minilength = self.min_length_slider.findChild(QSlider).value()
        max_gap = self.max_gap_slider.findChild(QSlider).value()
        min_angle = self.min_angle_slider.findChild(QSlider).value()

        edge_linker = edgelink(self.image, minilength)
        edge_linker.get_edgelist()
        edge_lists = [np.array(edge) for edge in edge_linker.edgelist if len(edge) > 0]

        # Post-process edge lists based on max_gap and min_angle
        processed_edge_lists = self.post_process_edges(edge_lists, max_gap, min_angle)

        self.visualize_edge_lists(processed_edge_lists)

    def post_process_edges(self, edge_lists, max_gap, min_angle):
        # Implement post-processing logic here
        return edge_lists

    def visualize_edge_lists(self, edge_lists):
        self.edge_linked_image = np.zeros(self.image.shape, dtype=np.uint8)
        for edge in edge_lists:
            for point in edge:
                if 0 <= point[0] < self.edge_linked_image.shape[0] and 0 <= point[1] < self.edge_linked_image.shape[1]:
                    self.edge_linked_image[int(point[0]), int(point[1])] = 255

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.imshow(self.edge_linked_image, cmap='gray')
        ax.axis('off')
        self.canvas.draw()

    def clean_short_edges(self):
        if self.edge_lists is None:
            QMessageBox.warning(self, "Error", "Please apply edge link first.")
            return

        min_length = self.clean_edge_length_slider.findChild(QSlider).value()

        # Convert edge lists to the format expected by cleanedgelist
        converted_edge_lists = []
        for edge in self.edge_lists:
            converted_edge = np.array(edge)
            if converted_edge.shape[0] > 0:
                converted_edge_lists.append(converted_edge)

        try:
            cleaned_edge_lists = cleanedgelist(converted_edge_lists, min_length)

            # Convert cleaned edge lists back to our format
            final_edge_lists = [edge.tolist() for edge in cleaned_edge_lists if edge.shape[0] > 0]

            self.visualize_edge_lists(final_edge_lists)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"An error occurred while cleaning edges: {str(e)}")
            print(f"Error details: {traceback.format_exc()}")
    def reset_view(self):
        # Reset the view to show original edges when sliders change
        self.edge_linked_image = self.original_edges.copy()
        self.update_view()

    def post_process_edges(self, edge_lists, max_gap, min_angle):
        processed_edges = []
        for edge in edge_lists:
            new_edge = [edge[0]]
            for i in range(1, len(edge)):
                if self.point_distance(new_edge[-1], edge[i]) <= max_gap:
                    if len(new_edge) < 2 or self.angle_between_points(new_edge[-2], new_edge[-1], edge[i]) >= min_angle:
                        new_edge.append(edge[i])
                    else:
                        processed_edges.append(new_edge)
                        new_edge = [edge[i]]
                else:
                    processed_edges.append(new_edge)
                    new_edge = [edge[i]]
            processed_edges.append(new_edge)
        return processed_edges

    def point_distance(self, p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def angle_between_points(self, p1, p2, p3):
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        return np.degrees(angle)
    def visualize_edge_lists(self, edge_lists):
        self.edge_linked_image = np.zeros(self.image.shape, dtype=np.uint8)
        for edge in edge_lists:
            for point in edge:
                if 0 <= point[0] < self.edge_linked_image.shape[0] and 0 <= point[1] < self.edge_linked_image.shape[1]:
                    self.edge_linked_image[int(point[0]), int(point[1])] = 255

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.imshow(self.edge_linked_image, cmap='gray')
        ax.axis('off')
        self.canvas.draw()
    def update_view(self):
        if self.edge_linked_image is None:
            return
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.imshow(self.edge_linked_image, cmap='gray')
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
                    else:
                        processed_edges.append(np.array(new_edge))
                        new_edge = [edge[i]]
                else:
                    processed_edges.append(np.array(new_edge))
                    new_edge = [edge[i]]
            processed_edges.append(np.array(new_edge))
        return processed_edges
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


# class FullViewWindow(QDialog):
#     def __init__(self, image, title, original_shape, cmap=None, is_manual_interpretation=False, filter_type='canny', shearlet_system=None,
#                  canny_low=50, canny_high=150, sobel_ksize=3, shearlet_min_contrast=10):
#         super().__init__()
#         self.setWindowTitle(title)
#         self.setGeometry(100, 100, 800, 600)
#
#         self.original_image = image
#         self.display_image = image.copy()
#         self.filtered_image = None
#         self.original_shape = original_shape
#         self.cmap = cmap
#         self.is_manual_interpretation = is_manual_interpretation
#         self.filter_type = filter_type
#         self.shearlet_system = shearlet_system
#
#         self.canny_low = canny_low
#         self.canny_high = canny_high
#         self.sobel_ksize = sobel_ksize
#         self.shearlet_min_contrast = shearlet_min_contrast
#
#         self.lines = []
#         self.filtered_lines = []
#         self.current_line = []
#         self.is_drawing = False
#         self.is_semi_auto = False
#         self.is_edit_mode = False
#         self.semi_auto_start_point = None
#         self.max_path_length = 1000
#
#         self.setup_ui()
#         self.show_original_image()
#         self.update_edge_map()  # Generate initial edge map for semi-auto drawing
#
#     def setup_ui(self):
#         layout = QVBoxLayout()
#
#         self.figure = Figure(figsize=(8, 6))
#         self.canvas = FigureCanvas(self.figure)
#         self.toolbar = NavigationToolbar(self.canvas, self)
#
#         layout.addWidget(self.toolbar)
#         layout.addWidget(self.canvas)
#
#         if self.is_manual_interpretation:
#             self.setup_manual_interpretation_ui(layout)
#
#         self.setLayout(layout)
#
#         self.canvas.mpl_connect("button_press_event", self.on_canvas_click)
#         self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
#         self.canvas.mpl_connect("button_release_event", self.on_mouse_release)
#
#     def setup_manual_interpretation_ui(self, layout):
#         self.toggle_drawing_button = QPushButton("Enable Manual Drawing")
#         self.toggle_drawing_button.clicked.connect(self.toggle_drawing)
#         layout.addWidget(self.toggle_drawing_button)
#
#         self.toggle_semi_auto_button = QPushButton("Enable Semi-Auto Drawing")
#         self.toggle_semi_auto_button.clicked.connect(self.toggle_semi_auto)
#         layout.addWidget(self.toggle_semi_auto_button)
#
#         self.apply_filter_button = QPushButton("Apply Filter (Compare)")
#         self.apply_filter_button.clicked.connect(self.apply_filter)
#         layout.addWidget(self.apply_filter_button)
#
#         self.edit_mode_button = QPushButton("Enter Edit Mode")
#         self.edit_mode_button.clicked.connect(self.toggle_edit_mode)
#         layout.addWidget(self.edit_mode_button)
#
#     # def show_original_image(self):
#     #     self.figure.clear()
#     #     self.ax = self.figure.add_subplot(111)
#     #     self.ax.imshow(self.display_image, cmap=self.cmap, aspect='auto')
#     #     self.ax.axis('off')
#     #     self.figure.tight_layout(pad=0)
#     #     self.canvas.draw()
#
#     def update_edge_map(self):
#         self.edge_map = self.get_edge_map()
#
#     def apply_filter(self):
#         if self.filtered_image is None:
#             self.filtered_image = cv2.addWeighted(self.original_image, 0.7, self.edge_map, 0.3, 0)
#         self.filtered_lines = self.convert_edges_to_lines(self.edge_map)
#         self.show_filtered_image()
#         self.draw_lines(include_filtered=True)
#
#     def show_filtered_image(self):
#         self.figure.clear()
#         self.ax = self.figure.add_subplot(111)
#         self.ax.imshow(self.filtered_image, cmap=self.cmap, aspect='auto')
#         self.ax.axis('off')
#         self.figure.tight_layout(pad=0)
#         self.canvas.draw()
#
#     def show_image(self):
#         self.figure.clear()
#         self.ax = self.figure.add_subplot(111)
#         self.ax.imshow(self.display_image, cmap=self.cmap, aspect='auto')
#         self.ax.axis('off')
#         self.figure.tight_layout(pad=0)
#         self.canvas.draw()
#
#     def convert_edges_to_lines(self, edges):
#         contours = measure.find_contours(edges, 0.5)
#         lines = []
#         for contour in contours:
#             simplified = measure.approximate_polygon(contour, tolerance=2)
#             if len(simplified) > 1:
#                 lines.append([(int(x), int(y)) for y, x in simplified])
#         return lines
#
#     def on_canvas_click(self, event):
#         if not self.is_manual_interpretation or event.inaxes != self.ax:
#             return
#
#         x, y = int(event.xdata), int(event.ydata)
#
#         if self.is_drawing:
#             if event.button == 1:  # Left click
#                 if not self.current_line:
#                     self.current_line = [(x, y)]
#                 else:
#                     self.current_line.append((x, y))
#                     self.draw_lines()
#             elif event.button == 3:  # Right click
#                 if self.current_line:
#                     self.lines.append(self.current_line)
#                     self.current_line = []
#                     self.draw_lines()
#         elif self.is_semi_auto:
#             if not self.semi_auto_start_point:
#                 self.semi_auto_start_point = (x, y)
#                 QMessageBox.information(self, "Semi-Auto Drawing", "Start point set. Click again to set end point.")
#             else:
#                 end_point = (x, y)
#                 tracked_line = self.semi_automatic_tracking(self.semi_auto_start_point, end_point)
#                 self.lines.append(tracked_line)
#                 self.semi_auto_start_point = None
#                 self.draw_lines()
#                 QMessageBox.information(self, "Semi-Auto Drawing", "Line drawn. You can start a new line.")
#         elif self.is_edit_mode:
#             self.edit_nearest_point(x, y)
#
#     def on_mouse_move(self, event):
#         if not self.is_manual_interpretation or not self.is_drawing or not self.current_line or event.inaxes != self.ax:
#             return
#
#         x, y = int(event.xdata), int(event.ydata)
#         temp_line = self.current_line + [(x, y)]
#         self.draw_lines(temp_line)
#
#     def toggle_drawing(self):
#         self.is_drawing = not self.is_drawing
#         self.is_semi_auto = False
#         self.is_edit_mode = False
#         self.semi_auto_start_point = None
#         self.toggle_drawing_button.setText("Disable Manual Drawing" if self.is_drawing else "Enable Manual Drawing")
#         self.toggle_semi_auto_button.setText("Enable Semi-Auto Drawing")
#         self.edit_mode_button.setText("Enter Edit Mode")
#
#     def start_autotrack(self):
#         if not self.start_point or not self.end_point:
#             QMessageBox.warning(self, "Error", "Start or end point not set")
#             return
#
#         blurred_image = gaussian_filter(self.image, sigma=6)
#
#         if self.filter_type == 'canny':
#             edges = feature.canny(blurred_image, sigma=2)
#         elif self.filter_type == 'sobel':
#             sobel_x = sobel(blurred_image, axis=0)
#             sobel_y = sobel(blurred_image, axis=1)
#             grad_magnitude = np.hypot(sobel_x, sobel_y)
#             large_edges = grad_magnitude > np.percentile(grad_magnitude, 90)
#             edges = np.where(large_edges, grad_magnitude, 0)
#         elif self.filter_type == 'shearlet':
#             shearlet_system = EdgeSystem(*self.image.shape)
#             edges, _ = shearlet_system.detect(blurred_image, min_contrast=10)
#             edges = mask(edges, thin_mask(edges))
#
#         costs = np.where(edges, 1 / (edges + 1e-6), 1)
#         costs /= np.max(costs)
#
#         # Convert points from display coordinates to original image coordinates
#         start_point_original = (self.start_point[0] * self.original_shape[0] // self.image.shape[0],
#                                 self.start_point[1] * self.original_shape[1] // self.image.shape[1])
#         end_point_original = (self.end_point[0] * self.original_shape[0] // self.image.shape[0],
#                               self.end_point[1] * self.original_shape[1] // self.image.shape[1])
#
#         #print(f"Start point in original image coordinates: {start_point_original}")
#         #print(f"End point in original image coordinates: {end_point_original}")
#
#         # Execute pathfinding using A* algorithm
#         path = a_star(costs, start_point_original, end_point_original)
#         self.draw_path(path, start_point_original, end_point_original)
#
#     def draw_path(self, path, start_point_original, end_point_original):
#         ax = self.canvas.figure.gca()
#         # Convert path coordinates back to display coordinates
#         path = [(y * self.image.shape[0] // self.original_shape[0], x * self.image.shape[1] // self.original_shape[1]) for y, x in path]
#         path_x, path_y = zip(*[(x, y) for x, y in path])
#         ax.plot(path_y, path_x, 'k-', linewidth=2)
#         ax.scatter(start_point_original[1] * self.image.shape[1] // self.original_shape[1],
#                    start_point_original[0] * self.image.shape[0] // self.original_shape[0],
#                    color='green', s=100)
#         ax.scatter(end_point_original[1] * self.image.shape[1] // self.original_shape[1],
#                    end_point_original[0] * self.image.shape[0] // self.original_shape[0],
#                    color='blue', s=100)
#         self.canvas.draw()
#         #print(f"Path coordinates in display coordinates: {list(zip(path_x, path_y))}")
#         self.start_point = None
#         self.end_point = None
#
#     def on_mouse_release(self, event):
#         if self.is_manual_interpretation and self.is_drawing and event.button == 1 and self.current_line:  # Left click release
#             x, y = int(event.xdata), int(event.ydata)
#             self.current_line.append((x, y))
#             self.draw_lines()
#
#     def draw_lines(self, temp_line=None, include_filtered=False):
#         self.show_original_image()  # Always draw on the original image
#         for line in self.lines:
#             if line and len(line) > 1:
#                 x, y = zip(*line)
#                 self.ax.plot(x, y, 'r-')
#         if include_filtered:
#             for line in self.filtered_lines:
#                 if line and len(line) > 1:
#                     x, y = zip(*line)
#                     self.ax.plot(x, y, 'b-')  # Draw filtered lines in blue
#         if self.current_line and len(self.current_line) > 1:
#             x, y = zip(*self.current_line)
#             self.ax.plot(x, y, 'r-')
#         if temp_line and len(temp_line) > 1:
#             x, y = zip(*temp_line)
#             self.ax.plot(x, y, 'r--')
#         self.canvas.draw()
#
#     def semi_automatic_tracking(self, start_point, end_point):
#         # Convert points to image coordinates
#         start = (int(start_point[1]), int(start_point[0]))
#         end = (int(end_point[1]), int(end_point[0]))
#
#         # Create a cost map from the edge map
#         cost_map = 1 - self.edge_map / 255.0  # Invert so edges have low cost
#
#         # Use A* algorithm to find the path
#         path = self.a_star(cost_map, start, end)
#
#         if path:
#             # Convert path back to display coordinates
#             tracked_line = [(x, y) for y, x in path]
#             return tracked_line
#         else:
#             QMessageBox.warning(self, "Path Not Found", "Could not find a path between the selected points. Try selecting closer points or adjusting the filter settings.")
#             return None
#
#     def get_edge_map(self):
#         if self.filter_type == 'canny':
#             return cv2.Canny(self.original_image, self.canny_low, self.canny_high)
#         elif self.filter_type == 'sobel':
#             grad_x = cv2.Sobel(self.original_image, cv2.CV_64F, 1, 0, ksize=self.sobel_ksize)
#             grad_y = cv2.Sobel(self.original_image, cv2.CV_64F, 0, 1, ksize=self.sobel_ksize)
#             edges = np.sqrt(grad_x**2 + grad_y**2)
#             return cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#         elif self.filter_type == 'shearlet':
#             if self.shearlet_system is not None:
#                 edges, _ = self.shearlet_system.detect(self.original_image, min_contrast=self.shearlet_min_contrast)
#                 edges = mask(edges, thin_mask(edges))
#                 return (edges * 255).astype(np.uint8)
#             else:
#                 return np.zeros_like(self.original_image)
#         else:
#             return np.zeros_like(self.original_image)
#
#     def a_star(self, cost_map, start, goal):
#         def heuristic(a, b):
#             return np.hypot(b[0] - a[0], b[1] - a[1])
#
#         neighbors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
#
#         close_set = set()
#         came_from = {}
#         gscore = {start: 0}
#         fscore = {start: heuristic(start, goal)}
#         oheap = []
#
#         heapq.heappush(oheap, (fscore[start], start))
#
#         while oheap:
#             current = heapq.heappop(oheap)[1]
#
#             if current == goal:
#                 path = []
#                 while current in came_from:
#                     path.append(current)
#                     current = came_from[current]
#                 path.append(start)
#                 path.reverse()
#                 return path
#
#             close_set.add(current)
#
#             if len(close_set) > self.max_path_length:
#                 return None  # Path is too long, abort search
#
#             for i, j in neighbors:
#                 neighbor = current[0] + i, current[1] + j
#                 if 0 <= neighbor[0] < cost_map.shape[0] and 0 <= neighbor[1] < cost_map.shape[1]:
#                     if neighbor in close_set:
#                         continue
#                     tentative_g_score = gscore[current] + cost_map[neighbor[0]][neighbor[1]]
#                     if tentative_g_score < gscore.get(neighbor, np.inf):
#                         came_from[neighbor] = current
#                         gscore[neighbor] = tentative_g_score
#                         fscore[neighbor] = gscore[neighbor] + heuristic(neighbor, goal)
#                         heapq.heappush(oheap, (fscore[neighbor], neighbor))
#
#         return None
#
#     def keyPressEvent(self, event):
#         if self.is_manual_interpretation and event.key() == Qt.Key_E:  # 'E' for edit mode
#             self.toggle_edit_mode()
#         else:
#             super().keyPressEvent(event)
#
#     def toggle_edit_mode(self):
#         if not self.is_manual_interpretation:
#             return
#
#         self.is_edit_mode = not self.is_edit_mode
#         self.is_drawing = False
#         self.is_semi_auto = False
#         if self.is_edit_mode:
#             self.toggle_drawing_button.setEnabled(False)
#             self.toggle_semi_auto_button.setEnabled(False)
#             self.edit_mode_button.setText("Exit Edit Mode")
#             QMessageBox.information(self, "Edit Mode", "Click near a line point to edit. Click 'Exit Edit Mode' when done.")
#         else:
#             self.toggle_drawing_button.setEnabled(True)
#             self.toggle_semi_auto_button.setEnabled(True)
#             self.edit_mode_button.setText("Enter Edit Mode")
#
#     def toggle_semi_auto(self):
#         self.is_semi_auto = not self.is_semi_auto
#         self.is_drawing = False
#         self.is_edit_mode = False
#         self.semi_auto_start_point = None
#         self.toggle_semi_auto_button.setText("Disable Semi-Auto Drawing" if self.is_semi_auto else "Enable Semi-Auto Drawing")
#         self.toggle_drawing_button.setText("Enable Manual Drawing")
#         self.edit_mode_button.setText("Enter Edit Mode")
#
#     def edit_nearest_point(self, x, y):
#         min_distance = float('inf')
#         nearest_line = None
#         nearest_point_index = None
#         is_filtered_line = False
#
#         # Check manually drawn lines
#         for i, line in enumerate(self.lines):
#             for j, point in enumerate(line):
#                 distance = np.sqrt((x - point[0])**2 + (y - point[1])**2)
#                 if distance < min_distance:
#                     min_distance = distance
#                     nearest_line = i
#                     nearest_point_index = j
#                     is_filtered_line = False
#
#         # Check filtered lines
#         for i, line in enumerate(self.filtered_lines):
#             for j, point in enumerate(line):
#                 distance = np.sqrt((x - point[0])**2 + (y - point[1])**2)
#                 if distance < min_distance:
#                     min_distance = distance
#                     nearest_line = i
#                     nearest_point_index = j
#                     is_filtered_line = True
#
#         if nearest_line is not None and min_distance < 10:  # Threshold for selection
#             if is_filtered_line:
#                 self.filtered_lines[nearest_line][nearest_point_index] = (x, y)
#             else:
#                 self.lines[nearest_line][nearest_point_index] = (x, y)
#             self.draw_lines()
#
#     def on_edit_click(self, event):
#         if event.inaxes != self.ax:
#             return
#
#         x, y = event.xdata, event.ydata
#         min_distance = float('inf')
#         selected_line = None
#         selected_point_index = None
#
#         for i, line in enumerate(self.lines):
#             for j, point in enumerate(line):
#                 distance = np.sqrt((x - point[0])**2 + (y - point[1])**2)
#                 if distance < min_distance:
#                     min_distance = distance
#                     selected_line = i
#                     selected_point_index = j
#
#         if selected_line is not None and min_distance < 10:  # Threshold for selection
#             self.edit_line(selected_line, selected_point_index, (x, y))
#
#     def edit_line(self, line_index, point_index, new_position):
#         self.lines[line_index][point_index] = new_position
#         self.draw_lines()
#
#     def update_thresholds(self, canny_low=None, canny_high=None, sobel_ksize=None, shearlet_min_contrast=None):
#         if canny_low is not None:
#             self.canny_low = canny_low
#         if canny_high is not None:
#             self.canny_high = canny_high
#         if sobel_ksize is not None:
#             self.sobel_ksize = sobel_ksize
#         if shearlet_min_contrast is not None:
#             self.shearlet_min_contrast = shearlet_min_contrast
#         self.update_edge_map()  # Update edge map when thresholds change
#
#     def handle_semi_auto(self, x, y):
#         if not self.semi_auto_start_point:
#             self.semi_auto_start_point = (x, y)
#             QMessageBox.information(self, "Semi-Auto Drawing", "Start point set. Click again to set end point.")
#         else:
#             end_point = (x, y)
#             QApplication.setOverrideCursor(Qt.WaitCursor)
#             try:
#                 tracked_line = self.semi_automatic_tracking(self.semi_auto_start_point, end_point)
#                 if tracked_line and len(tracked_line) > 1:
#                     self.lines.append(tracked_line)
#                     self.draw_lines()
#                     QMessageBox.information(self, "Semi-Auto Drawing", "Line drawn. You can start a new line.")
#                 else:
#                     QMessageBox.warning(self, "Semi-Auto Drawing", "Unable to find a path. Try different points or adjust the filter.")
#             except Exception as e:
#                 QMessageBox.critical(self, "Error", f"An error occurred during semi-automatic tracking: {str(e)}")
#             finally:
#                 QApplication.restoreOverrideCursor()
#             self.semi_auto_start_point = None

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
        self.shearletMinContrast = QSlider(Qt.Horizontal)
        self.shearletMinContrast.setRange(0, 100)
        self.shearletMinContrast.setValue(10)

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
        self.add_filter_tab("Canny")
        self.add_filter_tab("Sobel")
        self.add_filter_tab("Shearlet")

        # Add "+" tab for creating new tabs
        self.tab_widget.addTab(QWidget(), "+")
        self.tab_widget.tabBarClicked.connect(self.handle_tab_click)

        # Load image button
        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.load_image)
        main_layout.addWidget(self.load_button)

        # Clean Short Edges button
        self.clean_edges_button = QPushButton("Clean Short Edges")
        self.clean_edges_button.clicked.connect(self.clean_short_edges)
        main_layout.addWidget(self.clean_edges_button)

        # Manual Interpretation button
        self.manual_interpretation_button = QPushButton("Manual Interpretation")
        self.manual_interpretation_button.clicked.connect(self.open_manual_interpretation)
        main_layout.addWidget(self.manual_interpretation_button)

        self.createMenus()

    def add_filter_tab(self, filter_name):
        if filter_name == "Canny":
            tab = CannyFilterTab(self)
        elif filter_name == "Sobel":
            tab = SobelFilterTab(self)
        elif filter_name == "Shearlet":
            tab = ShearletFilterTab(self)
        else:
            tab = FilterTab(self)

        self.tab_widget.insertTab(self.tab_widget.count() - 1, tab, filter_name)

    def open_manual_interpretation(self):
        current_tab = self.tab_widget.currentWidget()
        if isinstance(current_tab, FilterTab) and self.img is not None:
            self.manual_interpretation_window = ManualInterpretationWindow(self.img, current_tab.filtered_image, self)
            self.manual_interpretation_window.show()
        else:
            QMessageBox.warning(self, "Warning", "Please load an image and apply a filter first before using manual interpretation.")
    def handle_tab_click(self, index):
        if self.tab_widget.tabText(index) == "+":
            filter_name, ok = QInputDialog.getText(self, "New Filter", "Enter filter name:")
            if ok and filter_name:
                self.add_filter_tab(filter_name)
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_image_sizes()

    def show_image(self, figure, image, cmap='gray'):
        figure.clear()
        ax = figure.add_subplot(111)
        ax.imshow(image, cmap=cmap, aspect='equal')
        ax.axis('off')
        figure.tight_layout(pad=0)
        figure.canvas.draw()

    def update_image_sizes(self):
        if self.img is not None:
            self.show_original_image()
            self.apply_canny_filter()
            self.apply_sobel_filter()
            self.apply_shearlet_filter()
            self.show_manual_interpretation()

    def process_batch_files(self, input_files, output_dir, filter_type):
        # Find the correct filter tab
        filter_tab = None
        for i in range(self.tab_widget.count() - 1):  # Exclude the "+" tab
            tab = self.tab_widget.widget(i)
            if isinstance(tab, FilterTab) and self.tab_widget.tabText(i) == filter_type:
                filter_tab = tab
                break

        if filter_tab is None:
            QMessageBox.warning(self, "Error", f"Filter {filter_type} not found.")
            return

        for input_file in input_files:
            img = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # Apply the filter
            filter_tab.set_input_image(img)
            filter_tab.update_filter()
            processed = filter_tab.filtered_image

            output_filename = os.path.join(output_dir,
                                           f"{os.path.splitext(os.path.basename(input_file))[0]}_{filter_type}.png")
            cv2.imwrite(output_filename, processed)

        QMessageBox.information(self, "Batch Process Complete",
                                f"Processed {len(input_files)} images with {filter_type} filter.")

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
        edgeLinkAction = QAction('Edge Link', self)
        # edgeLinkAction.triggered.connect(self.open_edge_link_window)
        toolsMenu.addAction(edgeLinkAction)

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
            self.img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if self.img is not None:
                self.img = cv2.normalize(self.img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                self.img = cv2.resize(self.img, (256, 256))
                self.mask = np.zeros(self.img.shape[:2], np.uint8)
                self.filtered_img = self.img.copy()
                self.shearlet_system = EdgeSystem(*self.img.shape)

                # Update all tabs with the new image
                for i in range(self.tab_widget.count() - 1):  # Exclude the "+" tab
                    tab = self.tab_widget.widget(i)
                    if isinstance(tab, FilterTab):
                        tab.set_input_image(self.img)

                # Enable buttons that require an image
                self.manual_interpretation_button.setEnabled(True)
                self.clean_edges_button.setEnabled(True)

                QMessageBox.information(self, "Image Loaded", "Image loaded successfully.")
            else:
                QMessageBox.warning(self, "Error", "Failed to load image.")

    def clean_short_edges(self):
        current_tab = self.tab_widget.currentWidget()
        if isinstance(current_tab, FilterTab) and current_tab.filtered_image is not None:
            # Convert the filtered image to a binary image
            _, binary_image = cv2.threshold(current_tab.filtered_image, 127, 255, cv2.THRESH_BINARY)

            # Find contours
            contours, _ = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            # Filter out short edges
            min_length = 10  # You can adjust this value
            long_contours = [cnt for cnt in contours if cv2.arcLength(cnt, False) > min_length]

            # Create a blank image and draw the long contours
            cleaned_image = np.zeros_like(current_tab.filtered_image)
            cv2.drawContours(cleaned_image, long_contours, -1, (255, 255, 255), 1)

            # Update the filtered image in the tab
            current_tab.filtered_image = cleaned_image
            current_tab.show_filtered_image()

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

    # def apply_canny_filter(self):
    #     if self.img is None:
    #         return
    #     threshold1 = self.cannyThreshold1.value()
    #     threshold2 = self.cannyThreshold2.value()
    #
    #     edges = cv2.Canny(self.img, threshold1, threshold2)
    #     edges = self.apply_skeletonize(edges)
    #     self.show_canny_image(edges)
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
    #
    # def show_canny_image(self, edges):
    #     self.show_image(self.figure_canny, edges)
    #
    # def show_sobel_image(self, sobel):
    #     self.show_image(self.figure_sobel, sobel)
    #
    # def show_manual_interpretation(self):
    #     self.show_image(self.figure_manual, self.img)
    #
    # def on_click_before(self, event):
    #     if event.button == 1:  # Left click
    #         self.open_full_view(self.img, "Original Image")
    #
    # def on_click_canny(self, event):
    #     if event.button == 1:  # Left click
    #         threshold1 = self.cannyThreshold1.value()
    #         threshold2 = self.cannyThreshold2.value()
    #         edges = cv2.Canny(self.img, threshold1, threshold2)
    #         edges = self.apply_skeletonize(edges)
    #         self.open_full_view(edges, "Canny Filtered Image", cmap='gray')
    #
    # def on_click_sobel(self, event):
    #     if event.button == 1:  # Left click
    #         ksize = self.sobelKsize.value()
    #         if ksize % 2 == 0:
    #             ksize += 1
    #         grad_x = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=ksize)
    #         grad_y = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, ksize=ksize)
    #         sobel = np.sqrt(grad_x**2 + grad_y**2)
    #         sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    #         sobel = self.apply_skeletonize(sobel)
    #         self.open_full_view(sobel, "Sobel Filtered Image", cmap='gray')
    #
    # def on_click_manual(self, event):
    #     if event.button == 1:  # Left click
    #         selected_filter = self.filter_type_combo.currentText()
    #         self.open_full_view(self.img, "Manual Interpretation", is_manual_interpretation=True, filter_type=selected_filter)

    # def open_full_view(self, image, title, cmap='gray', is_manual_interpretation=False, filter_type='canny'):
    #     self.full_view_window = FullViewWindow(
    #         image, title, self.img.shape, cmap, is_manual_interpretation, filter_type, self.shearlet_system,
    #         canny_low=self.cannyThreshold1.value(),
    #         canny_high=self.cannyThreshold2.value(),
    #         sobel_ksize=self.sobelKsize.value(),
    #         shearlet_min_contrast=self.shearletMinContrast.value()
    #     )
    #     self.full_view_window.exec_()

    # def update_canny_label(self):
    #     self.cannyThreshold1_label.setText(str(self.cannyThreshold1.value()))
    #     self.cannyThreshold2_label.setText(str(self.cannyThreshold2.value()))
    #     self.apply_canny_filter()
    #     self.update_full_view_window()
    #
    # def update_sobel_label(self):
    #     self.sobelKsize_label.setText(str(self.sobelKsize.value()))
    #     self.apply_sobel_filter()
    #     self.update_full_view_window()
    #
    # def apply_shearlet_filter(self):
    #     if self.img is None or self.shearlet_system is None:
    #         return
    #     min_contrast = self.shearletMinContrast.value()
    #     edges, orientations = self.shearlet_system.detect(self.img, min_contrast=min_contrast)
    #     edges = (edges * 255).astype(np.uint8)
    #     edges = self.apply_skeletonize(edges)
    #     thinned_edges = mask(edges, thin_mask(edges))
    #     edge_overlay = overlay(self.img, thinned_edges)
    #     self.show_shearlet_image(edge_overlay)
    #
    # def show_shearlet_image(self, edges):
    #     self.show_image(self.figure_shearlet, edges, cmap='jet')
    #
    # def on_click_shearlet(self, event):
    #     if event.button == 1:  # Left click
    #         min_contrast = self.shearletMinContrast.value()
    #         edges, orientations = self.shearlet_system.detect(self.img, min_contrast=min_contrast)
    #         edges = (edges * 255).astype(np.uint8)
    #         edges = self.apply_skeletonize(edges)
    #         thinned_edges = mask(edges, thin_mask(edges))
    #         edge_overlay = overlay(self.img, thinned_edges)
    #         self.open_full_view(edge_overlay, "Shearlet Filtered Image", cmap=None)
    #
    # def update_shearlet_label(self):
    #     self.shearletMinContrast_label.setText(str(self.shearletMinContrast.value()))
    #     self.apply_shearlet_filter()
    #     self.update_full_view_window()
    #
    # def update_full_view_window(self):
    #     if self.full_view_window and self.full_view_window.is_manual_interpretation:
    #         self.full_view_window.update_thresholds(
    #             canny_low=self.cannyThreshold1.value(),
    #             canny_high=self.cannyThreshold2.value(),
    #             sobel_ksize=self.sobelKsize.value(),
    #             shearlet_min_contrast=self.shearletMinContrast.value()
    #         )



if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyWindow()
    ex.show()
    sys.exit(app.exec_())
