import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget,
                             QPushButton, QGroupBox, QGridLayout, QSlider, QFileDialog, QDialog, QScrollArea, QComboBox, QMessageBox, QMenu, QAction, QInputDialog)
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

    def apply_canny_filter(self):
        threshold1 = self.threshold1_slider.value()
        threshold2 = self.threshold2_slider.value()
        self.filtered_image = cv2.Canny(self.image, threshold1, threshold2)
        self.threshold1_label.setText(str(threshold1))
        self.threshold2_label.setText(str(threshold2))

    def apply_sobel_filter(self):
        ksize = self.ksize_slider.value()
        if ksize % 2 == 0:
            ksize += 1  # Ensure ksize is odd
        grad_x = cv2.Sobel(self.image, cv2.CV_64F, 1, 0, ksize=ksize)
        grad_y = cv2.Sobel(self.image, cv2.CV_64F, 0, 1, ksize=ksize)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        self.filtered_image = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        _, self.filtered_image = cv2.threshold(self.filtered_image, 0, 255, cv2.THRESH_BINARY_INV)
        self.ksize_label.setText(str(ksize))

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


class FullViewWindow(QDialog):
    def __init__(self, image, title, original_shape, cmap=None, is_manual_interpretation=False, filter_type='canny'):
        super().__init__()
        self.setWindowTitle(title)
        self.setGeometry(100, 100, 800, 600)
        
        self.image = image
        self.original_shape = original_shape
        self.cmap = cmap
        self.is_manual_interpretation = is_manual_interpretation
        self.filter_type = filter_type

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)

        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.scroll_area)

        container = QWidget()
        container_layout = QVBoxLayout()
        container_layout.addWidget(self.canvas)
        container.setLayout(container_layout)

        self.scroll_area.setWidget(container)
        self.setLayout(layout)

        self.drawing_enabled = False
        
        if is_manual_interpretation:
            self.start_point = None
            self.end_point = None
            self.canvas.mpl_connect("button_press_event", self.on_canvas_click)
            self.toggle_drawing_button = QPushButton("Enable Semi Autotracking")
            self.toggle_drawing_button.clicked.connect(self.toggle_drawing)
            layout.addWidget(self.toggle_drawing_button)

        self.show_image(image, cmap)

    def show_image(self, image, cmap):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.imshow(image, cmap=cmap, aspect='auto')
        ax.axis('off')
        self.figure.tight_layout(pad=0)
        self.canvas.draw()

    def on_canvas_click(self, event):
        if not self.drawing_enabled:
            return

        coords = (int(event.ydata), int(event.xdata))
        print(f"Clicked coordinates: {coords}")
        if not self.start_point:
            self.start_point = coords
            self.canvas.figure.gca().scatter(coords[1], coords[0], color='green', s=100)
            self.canvas.draw()
            print(f"Start point set at: {self.start_point}")
        else:
            self.end_point = coords
            self.canvas.figure.gca().scatter(coords[1], coords[0], color='blue', s=100)
            self.canvas.draw()
            print(f"End point set at: {self.end_point}")
            self.start_autotrack()

    def toggle_drawing(self):
        self.drawing_enabled = not self.drawing_enabled
        if self.drawing_enabled:
            self.toggle_drawing_button.setText("Disable Semi Autotracking")
        else:
            self.toggle_drawing_button.setText("Enable Semi Autotracking")

    def start_autotrack(self):
        if not self.start_point or not self.end_point:
            QMessageBox.warning(self, "Error", "Start or end point not set")
            return

        blurred_image = gaussian_filter(self.image, sigma=6)

        if self.filter_type == 'canny':
            edges = feature.canny(blurred_image, sigma=2)
        elif self.filter_type == 'sobel':
            sobel_x = sobel(blurred_image, axis=0)
            sobel_y = sobel(blurred_image, axis=1)
            grad_magnitude = np.hypot(sobel_x, sobel_y)
            large_edges = grad_magnitude > np.percentile(grad_magnitude, 90)
            edges = np.where(large_edges, grad_magnitude, 0)
        elif self.filter_type == 'shearlet':
            shearlet_system = EdgeSystem(*self.image.shape)
            edges, _ = shearlet_system.detect(blurred_image, min_contrast=40)
            edges = mask(edges, thin_mask(edges))

        costs = np.where(edges, 1 / (edges + 1e-6), 1)
        costs /= np.max(costs)

        # Convert points from display coordinates to original image coordinates
        start_point_original = (self.start_point[0] * self.original_shape[0] // self.image.shape[0],
                                self.start_point[1] * self.original_shape[1] // self.image.shape[1])
        end_point_original = (self.end_point[0] * self.original_shape[0] // self.image.shape[0],
                              self.end_point[1] * self.original_shape[1] // self.image.shape[1])

        #print(f"Start point in original image coordinates: {start_point_original}")
        #print(f"End point in original image coordinates: {end_point_original}")

        # Execute pathfinding using A* algorithm
        path = a_star(costs, start_point_original, end_point_original)
        self.draw_path(path, start_point_original, end_point_original)

    def draw_path(self, path, start_point_original, end_point_original):
        ax = self.canvas.figure.gca()
        # Convert path coordinates back to display coordinates
        path = [(y * self.image.shape[0] // self.original_shape[0], x * self.image.shape[1] // self.original_shape[1]) for y, x in path]
        path_x, path_y = zip(*[(x, y) for x, y in path])
        ax.plot(path_y, path_x, 'k-', linewidth=2)
        ax.scatter(start_point_original[1] * self.image.shape[1] // self.original_shape[1], 
                   start_point_original[0] * self.image.shape[0] // self.original_shape[0], 
                   color='green', s=100)
        ax.scatter(end_point_original[1] * self.image.shape[1] // self.original_shape[1], 
                   end_point_original[0] * self.image.shape[0] // self.original_shape[0], 
                   color='blue', s=100)
        self.canvas.draw()
        #print(f"Path coordinates in display coordinates: {list(zip(path_x, path_y))}")
        self.start_point = None
        self.end_point = None

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
        self.setGeometry(100, 100, 1200, 800)  # Increase initial window size
        self.initUI()

    def initUI(self):
        self.setWindowTitle('PZeroStudioImage')
        self.setGeometry(100, 100, 1200, 800)

        mainWidget = QWidget(self)
        self.setCentralWidget(mainWidget)

        mainLayout = QHBoxLayout(mainWidget)

        leftPanel = QVBoxLayout()
        mainLayout.addLayout(leftPanel)

        # Image Selection
        imageSelectionGroup = QGroupBox("Image Selection")
        imageSelectionLayout = QVBoxLayout()
        self.loadButton = QPushButton("Load Image")
        self.loadButton.clicked.connect(self.load_image)
        imageSelectionLayout.addWidget(self.loadButton)
        self.saveButton = QPushButton("Save Mask")
        self.saveButton.setEnabled(False)
        self.saveButton.clicked.connect(self.save_mask)
        imageSelectionLayout.addWidget(self.saveButton)
        imageSelectionGroup.setLayout(imageSelectionLayout)
        leftPanel.addWidget(imageSelectionGroup)

        rightPanel = QVBoxLayout()
        mainLayout.addLayout(rightPanel)

        # Image Displays
        displayGroup = QGroupBox("Image Displays")
        displayLayout = QGridLayout()
        displayLayout.setSpacing(10)  # Add some spacing between images


        self.figure_before = Figure(figsize=(5, 5))  # Increase figure size
        self.canvas_before = FigureCanvas(self.figure_before)
        self.canvas_before.setMinimumSize(300, 300)  # Set minimum size
        self.canvas_before.mpl_connect("button_press_event", self.on_click_before)
        displayLayout.addWidget(QLabel("Original Image"), 0, 0)
        displayLayout.addWidget(self.canvas_before, 1, 0)

        self.figure_canny = Figure(figsize=(5, 5))
        self.canvas_canny = FigureCanvas(self.figure_canny)
        self.canvas_canny.setMinimumSize(300, 300)
        self.canvas_canny.mpl_connect("button_press_event", self.on_click_canny)
        displayLayout.addWidget(QLabel("Canny Filtered Image"), 0, 1)
        displayLayout.addWidget(self.canvas_canny, 1, 1)

        self.figure_sobel = Figure(figsize=(5, 5))
        self.canvas_sobel = FigureCanvas(self.figure_sobel)
        self.canvas_sobel.setMinimumSize(300, 300)
        self.canvas_sobel.mpl_connect("button_press_event", self.on_click_sobel)
        displayLayout.addWidget(QLabel("Sobel Filtered Image"), 2, 0)
        displayLayout.addWidget(self.canvas_sobel, 3, 0)

        self.figure_manual = Figure(figsize=(5, 5))
        self.canvas_manual = FigureCanvas(self.figure_manual)
        self.canvas_manual.setMinimumSize(300, 300)
        self.canvas_manual.mpl_connect("button_press_event", self.on_click_manual)
        displayLayout.addWidget(QLabel("Manual Interpretation"), 4, 0)
        displayLayout.addWidget(self.canvas_manual, 5, 0)

        displayGroup.setLayout(displayLayout)
        rightPanel.addWidget(displayGroup)

        # Controls for Canny Filter
        cannyControlsGroup = QGroupBox("Canny Filter with Sliders")
        cannyLayout = QVBoxLayout()
        self.cannyThreshold1 = QSlider(Qt.Horizontal)
        self.cannyThreshold1.setRange(0, 255)
        self.cannyThreshold1.setValue(50)
        self.cannyThreshold1.valueChanged.connect(self.update_canny_label)
        cannyLayout.addWidget(QLabel("Threshold 1"))
        cannyLayout.addWidget(self.cannyThreshold1)
        self.cannyThreshold1_label = QLabel("50")
        cannyLayout.addWidget(self.cannyThreshold1_label)

        self.cannyThreshold2 = QSlider(Qt.Horizontal)
        self.cannyThreshold2.setRange(0, 255)
        self.cannyThreshold2.setValue(150)
        self.cannyThreshold2.valueChanged.connect(self.update_canny_label)
        cannyLayout.addWidget(QLabel("Threshold 2"))
        cannyLayout.addWidget(self.cannyThreshold2)
        self.cannyThreshold2_label = QLabel("150")
        cannyLayout.addWidget(self.cannyThreshold2_label)

        cannyControlsGroup.setLayout(cannyLayout)
        rightPanel.addWidget(cannyControlsGroup)

        # Controls for Sobel Filter
        sobelControlsGroup = QGroupBox("Sobel Filter with Sliders")
        sobelLayout = QVBoxLayout()
        self.sobelKsize = QSlider(Qt.Horizontal)
        self.sobelKsize.setRange(1, 31)
        self.sobelKsize.setValue(3)
        self.sobelKsize.setSingleStep(2)
        self.sobelKsize.setTickInterval(2)
        self.sobelKsize.setTickPosition(QSlider.TicksBelow)
        self.sobelKsize.valueChanged.connect(self.update_sobel_label)
        sobelLayout.addWidget(QLabel("Kernel Size"))
        sobelLayout.addWidget(self.sobelKsize)
        self.sobelKsize_label = QLabel("3")
        sobelLayout.addWidget(self.sobelKsize_label)
        sobelControlsGroup.setLayout(sobelLayout)
        rightPanel.addWidget(sobelControlsGroup)

        self.filter_type_combo = QComboBox()
        self.filter_type_combo.addItems(['canny', 'sobel'])
        rightPanel.addWidget(QLabel("Manual Interpretation Filter"))
        rightPanel.addWidget(self.filter_type_combo)

        self.createMenus()

        self.figure_shearlet = Figure(figsize=(5, 5))
        self.canvas_shearlet = FigureCanvas(self.figure_shearlet)
        self.canvas_shearlet.setMinimumSize(300, 300)
        self.canvas_shearlet.mpl_connect("button_press_event", self.on_click_shearlet)
        displayLayout.addWidget(QLabel("Shearlet Filtered Image"), 2, 1)
        displayLayout.addWidget(self.canvas_shearlet, 3, 1)

        # Controls for Shearlet Filter
        shearletControlsGroup = QGroupBox("Shearlet Filter with Sliders")
        shearletLayout = QVBoxLayout()
        self.shearletMinContrast = QSlider(Qt.Horizontal)
        self.shearletMinContrast.setRange(0, 100)
        self.shearletMinContrast.setValue(40)
        self.shearletMinContrast.valueChanged.connect(self.update_shearlet_label)
        shearletLayout.addWidget(QLabel("Min Contrast"))
        shearletLayout.addWidget(self.shearletMinContrast)
        self.shearletMinContrast_label = QLabel("40")
        shearletLayout.addWidget(self.shearletMinContrast_label)
        shearletControlsGroup.setLayout(shearletLayout)
        rightPanel.addWidget(shearletControlsGroup)

        self.filter_type_combo.addItem('shearlet')

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
        for input_file in input_files:
            img = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            if filter_type == "Canny":
                processed = cv2.Canny(img, self.cannyThreshold1.value(), self.cannyThreshold2.value())
            elif filter_type == "Sobel":
                ksize = self.sobelKsize.value()
                if ksize % 2 == 0:
                    ksize += 1
                grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
                grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
                processed = np.sqrt(grad_x**2 + grad_y**2)
                processed = (processed / processed.max() * 255).astype(np.uint8)
            elif filter_type == "Shearlet":
                edges, _ = self.shearlet_system.detect(img, min_contrast=self.shearletMinContrast.value())
                processed = (edges * 255).astype(np.uint8)

            output_filename = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(input_file))[0]}_{filter_type}.png")
            cv2.imwrite(output_filename, processed)

        QMessageBox.information(self, "Batch Process Complete", f"Processed {len(input_files)} images with {filter_type} filter.")

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
        mask_path, _ = QFileDialog.getOpenFileName(self, "Select Manual Interpretation Mask", "", "Image Files (*.png *.jpg *.bmp)")
        if not mask_path:
            return
        
        # Load and preprocess the mask
        manual_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        manual_mask = cv2.resize(manual_mask, (self.img.shape[1], self.img.shape[0]))
        manual_mask = (manual_mask > 128).astype(np.uint8)  # Binarize the mask
        
        # Calculate for Canny
        canny_low = self.cannyThreshold1.value()
        canny_high = self.cannyThreshold2.value()
        canny_edges = cv2.Canny(self.img, canny_low, canny_high)
        canny_precision, canny_recall = self.calculate_precision_recall(manual_mask, canny_edges)
        
        # Calculate for Sobel
        ksize = self.sobelKsize.value()
        if ksize % 2 == 0:
            ksize += 1
        grad_x = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=ksize)
        grad_y = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, ksize=ksize)
        sobel = np.sqrt(grad_x**2 + grad_y**2)
        sobel = (sobel / sobel.max() * 255).astype(np.uint8)
        _, sobel_binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        sobel_precision, sobel_recall = self.calculate_precision_recall(manual_mask, sobel_binary)
        
        # Calculate for Shearlet
        edges, _ = self.shearlet_system.detect(self.img, min_contrast=self.shearletMinContrast.value())
        shearlet_edges = (edges * 255).astype(np.uint8)
        _, shearlet_binary = cv2.threshold(shearlet_edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        shearlet_precision, shearlet_recall = self.calculate_precision_recall(manual_mask, shearlet_binary)
        
        # Show results in a message box
        msg = f"Canny Filter (low={canny_low}, high={canny_high}):\nPrecision: {canny_precision:.4f}\nRecall: {canny_recall:.4f}\n\n"
        msg += f"Sobel Filter (ksize={ksize}):\nPrecision: {sobel_precision:.4f}\nRecall: {sobel_recall:.4f}\n\n"
        msg += f"Shearlet Filter (min_contrast={self.shearletMinContrast.value()}):\nPrecision: {shearlet_precision:.4f}\nRecall: {shearlet_recall:.4f}"
        
        QMessageBox.information(self, "Edge Detection Metrics", msg)

        # Debug: Save intermediate results
        cv2.imwrite('debug_canny.png', canny_edges)
        cv2.imwrite('debug_sobel.png', sobel_binary)
        cv2.imwrite('debug_shearlet.png', shearlet_binary)
        cv2.imwrite('debug_manual_mask.png', manual_mask * 255)

    def calculate_precision_recall(self, ground_truth, prediction):
        true_positives = np.sum(np.logical_and(prediction == 255, ground_truth == 1))
        false_positives = np.sum(np.logical_and(prediction == 255, ground_truth == 0))
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
        self.figure_before.clear()
        self.canvas_before.draw()
        self.figure_canny.clear()
        self.canvas_canny.draw()
        self.figure_sobel.clear()
        self.canvas_sobel.draw()
        self.figure_manual.clear()
        self.canvas_manual.draw()
        self.saveButton.setEnabled(False)

    def open_project(self):
        file_path = QFileDialog.getOpenFileName(self, "Open Project File", "", "Project Files (*.pkl);;All Files (*)")[0]
        if file_path:
            with open(file_path, 'rb') as file:
                project_data = pickle.load(file)
                self.img = project_data['image']
                self.mask = project_data['mask']
                self.filtered_img = project_data['filtered_image']
                self.show_original_image()
                self.apply_canny_filter()
                self.apply_sobel_filter()
                self.show_manual_interpretation()
                self.saveButton.setEnabled(True)

    def save_project(self):
        if self.img is not None:
            file_path = QFileDialog.getSaveFileName(self, "Save Project File", "", "Project Files (*.pkl);;All Files (*)")[0]
            if file_path:
                project_data = {
                    'image': self.img,
                    'mask': self.mask,
                    'filtered_image': self.filtered_img
                }
                with open(file_path, 'wb') as file:
                    pickle.dump(project_data, file)

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
        print("Starting export_to_shapefile function")
        if self.img is None:
            print("No image loaded")
            QMessageBox.warning(self, "Error", "No image loaded.")
            return

        print("Opening ExportDialog")
        dialog = ExportDialog(self, "Shapefile")
        if dialog.exec_():
            print("ExportDialog executed successfully")
            selected_filter = dialog.get_selected_filter()
            print(f"Selected filter: {selected_filter}")
            
            print("Getting filtered image")
            filtered_image = self.get_filtered_image(selected_filter)
            print(f"Filtered image shape: {filtered_image.shape}")

            # Convert to binary
            print("Converting to binary")
            binary_method, ok = QInputDialog.getItem(self, "Binary Conversion", 
                                                    "Select binary conversion method:",
                                                    ["Otsu", "Adaptive"], 0, False)
            if ok and binary_method:
                binary_image = self.convert_to_binary(filtered_image, method=binary_method.lower())
                print(f"Binary conversion completed using {binary_method} method")
                
                # Save binary image for inspection
                binary_save_path, _ = QFileDialog.getSaveFileName(self, "Save Binary Image", "", "PNG Files (*.png)")
                if binary_save_path:
                    cv2.imwrite(binary_save_path, binary_image)
                    print(f"Binary image saved to: {binary_save_path}")
            else:
                print("Binary conversion cancelled")
                return

            print("Converting binary image to lines")
            lines = self.image_to_lines(binary_image)
            print(f"Number of lines: {len(lines)}")

            if not lines:
                print("No lines detected")
                QMessageBox.warning(self, "Warning", "No lines were detected in the image. The shapefile will not be created.")
                return

            print("Creating GeoDataFrame")
            gdf = gpd.GeoDataFrame({'geometry': lines})
            print(f"GeoDataFrame created with {len(gdf)} rows")
            if hasattr(self, 'geotiff_crs'):
                print(f"Setting CRS: {self.geotiff_crs}")
                gdf.crs = self.geotiff_crs
            else:
                print("No CRS information available")

            print("Opening save file dialog")
            save_path, _ = QFileDialog.getSaveFileName(self, "Save Shapefile", "", "Shapefile (*.shp)")
            if save_path:
                print(f"Save path selected: {save_path}")
                try:
                    print("Attempting to save shapefile")
                    gdf.to_file(save_path)
                    print("Shapefile saved successfully")
                    QMessageBox.information(self, "Success", "Shapefile exported successfully")
                except Exception as e:
                    print(f"Error saving shapefile: {str(e)}")
                    QMessageBox.critical(self, "Error", f"Failed to save shapefile: {str(e)}")
            else:
                print("Save operation cancelled")

        print("export_to_shapefile function completed")

    def export_to_vector(self):
        print("Starting export_to_vector function")
        if self.img is None:
            print("No image loaded")
            QMessageBox.warning(self, "Error", "No image loaded.")
            return

        print("Opening ExportDialog")
        dialog = ExportDialog(self, "Vector")
        if dialog.exec_():
            print("ExportDialog executed successfully")
            selected_filter = dialog.get_selected_filter()
            print(f"Selected filter: {selected_filter}")
            
            print("Getting filtered image")
            filtered_image = self.get_filtered_image(selected_filter)
            print(f"Filtered image shape: {filtered_image.shape}")

            # Convert to binary
            print("Converting to binary")
            binary_method, ok = QInputDialog.getItem(self, "Binary Conversion", 
                                                    "Select binary conversion method:",
                                                    ["Otsu", "Adaptive"], 0, False)
            if ok and binary_method:
                binary_image = self.convert_to_binary(filtered_image, method=binary_method.lower())
                print(f"Binary conversion completed using {binary_method} method")
            else:
                print("Binary conversion cancelled")
                return

            print("Converting binary image to lines")
            lines = self.image_to_lines(binary_image)
            print(f"Number of lines: {len(lines)}")

            print("Creating GeoDataFrame")
            gdf = gpd.GeoDataFrame({'geometry': lines})
            print(f"GeoDataFrame created with {len(gdf)} rows")

            if hasattr(self, 'geotiff_crs'):
                print(f"Setting CRS: {self.geotiff_crs}")
                gdf.crs = self.geotiff_crs
            else:
                print("No CRS information available")

            # Save to various vector formats
            formats = {
                "GeoJSON": ("*.geojson", gdf.to_file),
                "KML": ("*.kml", gdf.to_file),
                "GeoPackage": ("*.gpkg", gdf.to_file),
            }

            format_choice, ok = QInputDialog.getItem(self, "Choose Vector Format", 
                                                    "Select output format:", 
                                                    list(formats.keys()), 0, False)
            
            if ok and format_choice:
                file_extension, save_function = formats[format_choice]
                save_path, _ = QFileDialog.getSaveFileName(self, f"Save {format_choice}", "", f"{format_choice} Files ({file_extension})")
                if save_path:
                    print(f"Save path selected: {save_path}")
                    try:
                        print(f"Attempting to save as {format_choice}")
                        save_function(save_path, driver=format_choice)
                        print(f"{format_choice} saved successfully")
                        QMessageBox.information(self, "Success", f"File saved as {format_choice}")
                    except Exception as e:
                        print(f"Error saving {format_choice}: {str(e)}")
                        QMessageBox.critical(self, "Error", f"Failed to save {format_choice}: {str(e)}")
                else:
                    print("Save operation cancelled")
            else:
                print("Format selection cancelled")

        print("export_to_vector function completed")

    def export_to_png(self):
        if self.img is None:
            QMessageBox.warning(self, "Error", "No image loaded.")
            return

        dialog = ExportDialog(self, "PNG")
        if dialog.exec_():
            selected_filter = dialog.get_selected_filter()
            filtered_image = self.get_filtered_image(selected_filter)

            save_path, _ = QFileDialog.getSaveFileName(self, "Save PNG", "", "PNG Files (*.png)")
            if save_path:
                cv2.imwrite(save_path, filtered_image)

    def export_to_jpeg(self):
        if self.img is None:
            QMessageBox.warning(self, "Error", "No image loaded.")
            return

        dialog = ExportDialog(self, "JPEG")
        if dialog.exec_():
            selected_filter = dialog.get_selected_filter()
            filtered_image = self.get_filtered_image(selected_filter)

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

            save_path, _ = QFileDialog.getSaveFileName(self, "Save TIFF", "", "TIFF Files (*.tif)")
            if save_path:
                if hasattr(self, 'geotiff_transform') and hasattr(self, 'geotiff_projection'):
                    # Save as GeoTIFF with original coordinates
                    driver = gdal.GetDriverByName('GTiff')
                    dataset = driver.Create(save_path, filtered_image.shape[1], filtered_image.shape[0], 1, gdal.GDT_Byte)
                    dataset.SetGeoTransform(self.geotiff_transform)
                    dataset.SetProjection(self.geotiff_projection)
                    dataset.GetRasterBand(1).WriteArray(filtered_image)
                    dataset.FlushCache()
                else:
                    # Save as regular TIFF
                    cv2.imwrite(save_path, filtered_image)

    def get_filtered_image(self, filter_type):
        if filter_type == "Original":
            return self.img
        elif filter_type == "Canny":
            return cv2.Canny(self.img, self.cannyThreshold1.value(), self.cannyThreshold2.value())
        elif filter_type == "Sobel":
            ksize = self.sobelKsize.value()
            if ksize % 2 == 0:
                ksize += 1
            grad_x = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=ksize)
            grad_y = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, ksize=ksize)
            return cv2.addWeighted(cv2.convertScaleAbs(grad_x), 0.5, cv2.convertScaleAbs(grad_y), 0.5, 0)
        elif filter_type == "Shearlet":
            edges, _ = self.shearlet_system.detect(self.img, min_contrast=self.shearletMinContrast.value())
            return (edges * 255).astype(np.uint8)

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
        img_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image Files (*.png *.jpg *.bmp *.tif *.tiff)")
        if img_path:
            if img_path.lower().endswith(('.tif', '.tiff')):
                # Load GeoTIFF
                with rasterio.open(img_path) as src:
                    self.img = src.read(1)
                    self.geotiff_transform = src.transform
                    self.geotiff_crs = src.crs
                    self.geotiff_projection = src.crs.to_wkt()
            else:
                self.img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if self.img is not None:
                self.img = cv2.resize(self.img, (256, 256))
                self.mask = np.zeros(self.img.shape[:2], np.uint8)
                self.filtered_img = self.img.copy()
                self.shearlet_system = EdgeSystem(*self.img.shape)
                self.show_original_image()
                self.apply_canny_filter()
                self.apply_sobel_filter()
                self.apply_shearlet_filter()
                self.show_manual_interpretation()
                self.saveButton.setEnabled(True)

    def save_mask(self):
        if self.img is None:
            return

        dialog = SaveMaskDialog(self.img)
        if dialog.exec_() == QDialog.Accepted:
            # Mask saved inside the dialog; no need to do anything here.
            pass

    def apply_canny_filter(self):
        if self.img is None:
            return
        threshold1 = self.cannyThreshold1.value()
        threshold2 = self.cannyThreshold2.value()
        edges = cv2.Canny(self.img, threshold1, threshold2)
        self.show_canny_image(edges)

    def apply_sobel_filter(self):
        if self.img is None:
            return
        ksize = self.sobelKsize.value()
        if ksize % 2 == 0:
            ksize += 1  # Ensure ksize is odd
        grad_x = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=ksize)
        grad_y = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, ksize=ksize)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        sobel = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        self.show_sobel_image(sobel)

    def show_original_image(self):
        self.show_image(self.figure_before, self.img)

    def show_canny_image(self, edges):
        self.show_image(self.figure_canny, edges)

    def show_sobel_image(self, sobel):
        self.show_image(self.figure_sobel, sobel)

    def show_manual_interpretation(self):
        self.show_image(self.figure_manual, self.img)

    def on_click_before(self, event):
        if event.button == 1:  # Left click
            self.open_full_view(self.img, "Original Image")

    def on_click_canny(self, event):
        if event.button == 1:  # Left click
            edges = cv2.Canny(self.img, self.cannyThreshold1.value(), self.cannyThreshold2.value())
            self.open_full_view(edges, "Canny Filtered Image", cmap='gray')

    def on_click_sobel(self, event):
        if event.button == 1:  # Left click
            ksize = self.sobelKsize.value()
            if ksize % 2 == 0:
                ksize += 1
            grad_x = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=ksize)
            grad_y = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, ksize=ksize)
            abs_grad_x = cv2.convertScaleAbs(grad_x)
            abs_grad_y = cv2.convertScaleAbs(grad_y)
            sobel = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
            self.open_full_view(sobel, "Sobel Filtered Image", cmap='gray')

    def on_click_manual(self, event):
        if event.button == 1:  # Left click
            self.open_full_view(self.img, "Manual Interpretation", is_manual_interpretation=True, filter_type=self.filter_type_combo.currentText())

    def open_full_view(self, image, title, cmap='gray', is_manual_interpretation=False, filter_type='canny'):
        self.full_view_window = FullViewWindow(image, title, self.img.shape, cmap, is_manual_interpretation, filter_type)
        self.full_view_window.exec_()

    def update_canny_label(self):
        self.cannyThreshold1_label.setText(str(self.cannyThreshold1.value()))
        self.cannyThreshold2_label.setText(str(self.cannyThreshold2.value()))
        self.apply_canny_filter()

    def update_sobel_label(self):
        self.sobelKsize_label.setText(str(self.sobelKsize.value()))
        self.apply_sobel_filter()

    def apply_shearlet_filter(self):
        if self.img is None or self.shearlet_system is None:
            return
        min_contrast = self.shearletMinContrast.value()
        edges, orientations = self.shearlet_system.detect(self.img, min_contrast=min_contrast)
        thinned_edges = mask(edges, thin_mask(edges))
        edge_overlay = overlay(self.img, thinned_edges)
        self.show_shearlet_image(edge_overlay)

    def show_shearlet_image(self, edges):
        self.show_image(self.figure_shearlet, edges, cmap='jet')

    def on_click_shearlet(self, event):
        if event.button == 1:  # Left click
            min_contrast = self.shearletMinContrast.value()
            edges, orientations = self.shearlet_system.detect(self.img, min_contrast=min_contrast)
            thinned_edges = mask(edges, thin_mask(edges))
            edge_overlay = overlay(self.img, thinned_edges)
            self.open_full_view(edge_overlay, "Shearlet Filtered Image", cmap=None)

    def update_shearlet_label(self):
        self.shearletMinContrast_label.setText(str(self.shearletMinContrast.value()))
        self.apply_shearlet_filter()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyWindow()
    ex.show()
    sys.exit(app.exec_())
