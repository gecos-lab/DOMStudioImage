# DOMStudioImage: Comprehensive Lineament Detection Tool for Geological Images

DOMStudioImage is a robust and user-friendly application designed for the detection and analysis of lineaments in geological images. Leveraging advanced edge detection algorithms and providing extensive customization options, DOMStudioImage serves as an indispensable tool for geologists, researchers, and image processing enthusiasts.

## Table of Contents

- [Features](#features)
  - [Load and Display Geological Images](#load-and-display-geological-images)
  - [Advanced Edge Detection Filters](#advanced-edge-detection-filters)
    - [Canny Edge Detection](#canny-edge-detection)
    - [Sobel Edge Detection](#sobel-edge-detection)
    - [Shearlet Edge Detection](#shearlet-edge-detection)
    - [Laplacian Edge Detection](#laplacian-edge-detection)
    - [Roberts Edge Detection](#roberts-edge-detection)
    - [HED Edge Detection](#hed-edge-detection)
  - [Manual Interpretation Tools](#manual-interpretation-tools)
    - [Manual Drawing](#manual-drawing)
    - [Semi-Automatic Tracking](#semi-automatic-tracking)
    - [Edit Mode](#edit-mode)
    - [Undo/Redo Functionality](#undoredo-functionality)
  - [Edge Linking and Visualization](#edge-linking-and-visualization)
  - [Batch Processing](#batch-processing)
  - [Exporting Results](#exporting-results)
  - [Image Analysis Tools](#image-analysis-tools)
    - [Histogram](#histogram)
    - [Fourier Transform](#fourier-transform)
    - [Intensity Profile](#intensity-profile)
    - [Color Histogram](#color-histogram)
  - [Project Management](#project-management)
  - [TabView Flexibility](#tabview-flexibility)
- [Installation](#installation)
  - [Using Conda (Recommended)](#using-conda-recommended)
  - [Manual Installation (Alternative)](#manual-installation-alternative)
- [Usage](#usage)
  - [Loading and Displaying Images](#loading-and-displaying-images)
  - [Applying Edge Detection Filters](#applying-edge-detection-filters)
  - [Manual Interpretation Workflow](#manual-interpretation-workflow)
  - [Edge Linking and Visualization](#edge-linking-and-visualization-1)
  - [Batch Processing Images](#batch-processing-images)
  - [Exporting Results](#exporting-results-1)
  - [Using Image Analysis Tools](#using-image-analysis-tools)
  - [Managing Projects](#managing-projects)
- [Dependencies](#dependencies)
- [Developers](#developers)
- [License](#license)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)

## Features

DOMStudioImage is packed with a multitude of features tailored for comprehensive geological image analysis:

![Alt Text](/images/image.png)
### Load and Display Geological Images

- **Supported Formats:** PNG, JPEG, BMP, GeoTIFF (`.tif`, `.tiff`).
- **Geospatial Support:** For GeoTIFFs, preserves geospatial metadata including transform and CRS.
- **Display Panels:** Original and processed images are displayed side by side for easy comparison.
- **Resizing:** Automatically resizes images to 256x256 pixels for optimal display and processing efficiency.

### Advanced Edge Detection Filters

Enhance and detect geological lineaments using a variety of edge detection algorithms, each with customizable parameters:

#### Canny Edge Detection

- **Adjustable Parameters:**
  - **Threshold 1:** Lower threshold for hysteresis.
  - **Threshold 2:** Upper threshold for hysteresis.
- **Skeletonization:** Option to refine edges through skeletonization.
- **Customization:** Configure aperture size and enable L2 gradient calculation for more accurate edge magnitude.

#### Sobel Edge Detection

- **Adjustable Parameters:**
  - **Kernel Size:** Configurable odd kernel size for gradient calculation (e.g., 3, 5, 7).
- **Gradient Magnitude:** Calculates and normalizes the gradient magnitude from X and Y directions.
- **Skeletonization:** Option to refine edges through skeletonization.

#### Shearlet Edge Detection

- **Advanced Technique:** Utilizes shearlet transforms for multi-scale and multi-directional edge detection.
- **Adjustable Parameters:**
  - **Minimum Contrast:** Controls the sensitivity of the shearlet detector.
- **Skeletonization:** Option to refine edges through skeletonization.

#### Laplacian Edge Detection

- **Adjustable Parameters:**
  - **Kernel Size:** Configurable odd kernel size for Laplacian operator (e.g., 3, 5, 7).
- **Edge Enhancement:** Highlights regions of rapid intensity change.
- **Skeletonization:** Option to refine edges through skeletonization.

#### Roberts Edge Detection

- **Basic Gradient Operator:** Detects edges using the Roberts cross operator.
- **Skeletonization:** Option to refine edges through skeletonization.

#### HED Edge Detection

- **Neural Network-Based:** Employs a pre-trained Holistically-Nested Edge Detection (HED) model for precise edge detection.
- **Adjustable Parameters:**
  - **Model Selection:** Choose between available models (e.g., `bsds500`, `pascal`).
  - **Threshold:** Configure the threshold to binarize the edge probability map.
- **Side Outputs:** Option to view individual side outputs from different layers of the network.
- **Skeletonization:** Option to refine edges through skeletonization.

### Manual Interpretation Tools

Refine and customize detected edges with comprehensive manual and semi-automatic tools:

#### Manual Drawing

- **Freehand Drawing:** Manually add or adjust lineaments directly on the image.
- **Control Points:** Add, move, or delete nodes (control points) to shape lines precisely.

#### Semi-Automatic Tracking

- **A\* Pathfinding:** Automatically connect selected points using the A\* algorithm, enhancing accuracy and efficiency.
- **Parameter Adjustments:** Configure minimum edge length, maximum gap, and minimum angle to control the tracking behavior.

#### Edit Mode

- **Modify Existing Lines:** Enter edit mode to select and drag existing nodes for precise adjustments.
- **Delete Functionality:** Remove unwanted nodes or entire lines through context menus.

#### Undo/Redo Functionality

- **Action History:** Maintain a history of actions to easily undo or redo changes during manual interpretation.

### Edge Linking and Visualization

Enhance and visualize connected edges for better analysis and interpretation:

- **Edge Linking Parameters:**
  - **Minimum Edge Length:** Filter out short, insignificant edges.
  - **Maximum Gap:** Control the allowable gap between connected edges.
  - **Minimum Angle:** Ensure continuity in edge direction.
- **Visualization:** Display linked edges with options to highlight connections and remove noise.
- **Clean Short Edges:** Remove edges that do not meet the minimum length criteria.

### Batch Processing

Efficiently process multiple images with consistent settings:

- **Batch Process Dialog:**
  - **Input Selection:** Add multiple image files for simultaneous processing.
  - **Filter Selection:** Choose between Canny, Sobel, Shearlet, Laplacian, Roberts, or HED filters.
  - **Output Configuration:** Specify output directories and configure filter parameters.
  - **Progress Notifications:** Receive notifications upon completion of batch processing.

### Exporting Results

Export processed data in various formats suitable for GIS applications and further analysis:

- **Shapefile Export:**
  - **GeoDataFrame Creation:** Convert detected lineaments into Shapefiles compatible with GIS software.
  - **CRS Support:** Preserve Coordinate Reference System (CRS) information when available.
- **Vector Formats:**
  - **Supported Formats:** SVG, GeoJSON, KML, GeoPackage.
  - **SVG Conversion:** Transform edge detections into scalable vector graphics for versatile use.
- **Raster Formats:**
  - **Supported Formats:** PNG, JPEG, TIFF.
  - **Quality Adjustments:** Configure compression settings for JPEG exports and save GeoTIFFs with geospatial metadata if CRS information is available.

### Image Analysis Tools

Gain deeper insights into your geological images with comprehensive analysis tools:

#### Histogram

- **Pixel Intensity Distribution:** Visualize the distribution of pixel intensities across the image.

#### Fourier Transform

- **Frequency Analysis:** Analyze the frequency components of the image to identify periodic structures.

#### Intensity Profile

- **Line-Based Analysis:** Examine intensity variations along a specific line in the image for detailed study.

#### Color Histogram

- **Channel-Wise Distribution:** For colored images, analyze the distribution across different color channels (Red, Green, Blue).

### Project Management

Maintain and revisit your analysis workflow with robust project management features:

- **Save Project:**
  - **State Preservation:** Save the current state of your analysis, including loaded images, applied filters, and annotations.
  - **File Format:** Projects are saved in a pickle (`.pkl`) format for easy loading and continuation.
- **Load Project:**
  - **Restore Analysis:** Load previously saved projects to continue your work seamlessly.
- **New Project:**
  - **Reset Workspace:** Start a fresh analysis by resetting all loaded images and settings.

### TabView Flexibility

Organize your workflow efficiently with a flexible tabbed interface:

- **Dynamic Tabs:** Easily switch between different filters and analysis tools using tabs.
- **Customizable Layout:** Add, remove, or rearrange tabs to suit your workflow preferences.
- **Movable Tabs:** Drag and drop tabs to reorder them as needed.
- **Add New Filter Tabs:** Click the "+" tab to add additional filter tabs dynamically, allowing for extended analysis capabilities.

## Installation

### Using Conda (Recommended)

1. **Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)** if you haven't already.

2. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/domstudioimage.git
   cd domstudioimage
   ```

3. **Create the Conda Environment from the `environment.yml` File:**
   ```bash
   conda env create -f environment.yml
   ```

4. **Activate the Conda Environment:**
   ```bash
   conda activate domstudioimage
   ```

5. **Run the Application:**
   ```bash
   python main.py
   ```

### Manual Installation (Alternative)

If you prefer not to use Conda, you can install the dependencies manually:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/domstudioimage.git
   cd domstudioimage
   ```

2. **Install the Required Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application:**
   ```bash
   python main.py
   ```

**Note:** The `pycoshrem` package is included in the main folder for easy access. If you encounter issues, install it using:
```bash
pip install pycoshrem
```

## Usage

### Loading and Displaying Images

1. **Load an Image:**
   - Click the **"Load Image"** button in the main interface.
   - Supported formats include PNG, JPEG, BMP, and GeoTIFF.

2. **View Images:**
   - The original image is displayed alongside processed images in separate tabs.
   - Utilize the tabbed interface to switch between different filters and views.

### Applying Edge Detection Filters

1. **Select a Filter Tab:**
   - Choose from available filter tabs such as Canny, Sobel, Shearlet, Laplacian, Roberts, or HED.

2. **Adjust Filter Parameters:**
   - Use the provided sliders and controls within each filter tab to adjust parameters like thresholds, kernel sizes, and minimum contrast.

3. **Apply Filters:**
   - Filters are applied in real-time based on parameter adjustments.
   - **Skeletonization:** Optionally refine edges by applying skeletonization to enhance connectivity.

### Manual Interpretation Workflow

Refine detected edges with manual and semi-automatic tools:

1. **Open Manual Interpretation:**
   - Click the **"Manual Interpretation"** button to launch the manual interpretation dialog.

2. **Manual Drawing:**
   - **Enable Drawing:** Activate manual drawing mode to add or adjust lineaments by clicking and dragging on the image.
   - **Add Nodes:** Click to add control points (nodes) that define the shape of the lines.

3. **Semi-Automatic Tracking:**
   - **Enable Semi-Auto Mode:** Activate semi-automatic drawing to utilize the A\* algorithm for connecting selected points.
   - **Set Points:** Click to set the start and end points, and the algorithm will automatically generate the connecting path.

4. **Edit Mode:**
   - **Modify Lines:** Enter edit mode to select and drag existing nodes for precise adjustments.
   - **Delete Functionality:** Right-click on nodes or lines to delete them as needed.

5. **Undo/Redo Actions:**
   - Utilize the undo and redo buttons to revert or reapply actions during the interpretation process.

### Edge Linking and Visualization

Enhance and visualize connected edges for better analysis:

1. **Access Edge Linking:**
   - Navigate to **`Tools > Edge Linking`** in the menu bar.

2. **Configure Parameters:**
   - **Minimum Edge Length:** Set the minimum length for edges to be considered significant.
   - **Maximum Gap:** Define the maximum allowable gap between connected edges.
   - **Minimum Angle:** Specify the minimum angle to ensure continuity in edge direction.

3. **Apply Edge Linking:**
   - Click **"Apply Edge Link"** to process and visualize the connected edges.
   - **Clean Short Edges:** Optionally remove edges that do not meet the minimum length criteria.

### Batch Processing Images

Efficiently process multiple images with consistent settings:

1. **Open Batch Processing Dialog:**
   - Navigate to **`Tools > Batch Processing`** in the menu bar.

2. **Select Input Files:**
   - Click **"Add Files"** to select multiple image files for batch processing.

3. **Choose Filters:**
   - Select the desired filter (Canny, Sobel, Shearlet, Laplacian, Roberts, HED) to apply to all selected images.

4. **Configure Output Settings:**
   - Specify the output directory where processed images will be saved.
   - Adjust filter-specific parameters as needed.

5. **Start Processing:**
   - Click **"Process"** to begin batch processing. A progress notification will inform you upon completion.

### Exporting Results

Export processed data in various formats for further analysis or integration with GIS applications:

1. **Export to Shapefile:**
   - Go to **`File > Export > Export to Shapefile`**.
   - Select the desired filter and specify the save location.
   - **GeoDataFrame:** Converts detected lineaments into Shapefiles, preserving CRS information if available.

2. **Export to Vector Formats:**
   - Supported formats include **SVG**, **GeoJSON**, **KML**, and **GeoPackage**.
   - Access via **`File > Export > Export to Vector File`**.
   - **SVG Conversion:** Transforms edge detections into scalable vector graphics.

3. **Export to Raster Formats:**
   - Save images in **PNG**, **JPEG**, or **TIFF** formats.
   - Access via **`File > Export > Export to PNG/JPEG/TIFF`**.
   - **Quality Adjustments:** Configure compression settings for JPEG exports and save GeoTIFFs with geospatial metadata if CRS information is available.

### Using Image Analysis Tools

Gain deeper insights into your geological images with comprehensive analysis tools:

1. **Histogram:**
   - Navigate to **`Tools > Image Properties > Plot Histogram`** to view the distribution of pixel intensities.

2. **Fourier Transform:**
   - Access via **`Tools > Image Properties > Fourier Transform`** to analyze the frequency components of the image.

3. **Intensity Profile:**
   - Go to **`Tools > Image Properties > Intensity Profile`** to examine intensity variations along a specific line in the image.

4. **Color Histogram:**
   - For colored images, view the distribution across different color channels via **`Tools > Image Properties > Color Histogram`**.

### Project Management

Maintain and revisit your analysis workflow with robust project management features:

1. **Save Project:**
   - Click **`File > Save`** to save the current state of your analysis, including loaded images, applied filters, and annotations.
   - **File Format:** Projects are saved in a pickle (`.pkl`) format for easy loading and continuation.

2. **Load Project:**
   - Use **`File > Open`** to load a previously saved project.
   - **Restore Analysis:** Loads the state of your analysis, allowing you to continue where you left off.

3. **New Project:**
   - Click **`File > New`** to start a fresh analysis by resetting all loaded images and settings.

## Dependencies

Ensure all dependencies are installed for optimal performance. Refer to the `environment.yml` file for a complete list. Key dependencies include:

- **PyQt5:** For the graphical user interface.
- **OpenCV:** For image processing tasks.
- **NumPy:** For numerical operations.
- **Matplotlib:** For plotting and visualization.
- **Scikit-image:** For additional image processing functionalities.
- **Shapely & GeoPandas:** For geometric operations and handling spatial data.
- **rasterio & GDAL:** For working with raster data and GeoTIFF files.
- **Pycoshrem:** For shearlet-based edge detection.
- **SciPy:** For advanced scientific computations.
- **scikit-learn:** For precision and recall metrics.
- **PyTorch:** For HED neural network-based edge detection.
- **XML Libraries:** For SVG export functionality.

## Developers

- **Waqas Hussain**

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPLv3) - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

**Steps to Contribute:**

1. **Fork the Repository**
2. **Create a New Branch:**
   ```bash
   git checkout -b feature/YourFeatureName
   ```
3. **Commit Your Changes:**
   ```bash
   git commit -m "Add some feature"
   ```
4. **Push to the Branch:**
   ```bash
   git push origin feature/YourFeatureName
   ```
5. **Open a Pull Request**

Please ensure your code follows the project's coding standards and includes appropriate tests and documentation.

## Acknowledgments

- **Open-Source Libraries:**
  - Thanks to all the open-source libraries that made this project possible, including PyQt5, OpenCV, NumPy, Matplotlib, Scikit-image, Shapely, GeoPandas, rasterio, GDAL, SciPy, scikit-learn, PyTorch, and Pycoshrem.

- **Geological Community:**
  - Special thanks to the geological community for their invaluable input and feedback, which has been instrumental in shaping the functionalities of this tool.

- **Contributors:**
  - Gratitude to all contributors who have helped improve DOMStudioImage through their code, documentation, and bug reports.

---

*Feel free to reach out with any questions, suggestions, or feedback!*
```