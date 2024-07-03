# PZEROSTUDIOIMAGE: Lineament Detection application tool in Geological Images

This project provides a tool for detecting lineaments in geological images using various edge detection techniques. It offers a user-friendly interface for loading, processing, and analyzing geological images to identify and extract lineament features.

## Features

- Load and display geological images
- Apply multiple edge detection filters:
  - Canny
  - Sobel
  - Shearlet
- Manual interpretation tools
- Export results in various formats:
  - Shapefile
  - Vector (GeoJSON, KML, GeoPackage)
  - Raster (PNG, JPEG, TIFF)
- Batch processing capabilities
- Image analysis tools:
  - Histogram
  - Fourier Transform
  - Intensity Profile
  - Color Histogram

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the main script:
   ```
   python main.py
   ```

## Usage

1. Load an image using the "Load Image" button
2. Adjust filter parameters using the sliders
3. Use the manual interpretation tools for fine-tuning
4. Export the results in your desired format

For detailed usage instructions, please refer to the user manual.

## Dependencies

- Python 3.x
- PyQt5
- OpenCV
- NumPy
- Matplotlib
- scikit-image
- scipy
- shapely
- geopandas
- rasterio
- GDAL
- pycoshrem

## Developers

- Waqas Hussain

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Thanks to all the open-source libraries that made this project possible.
- Special thanks to the geological community for their input and feedback.
