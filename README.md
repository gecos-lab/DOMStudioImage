# DOMSTUDIOIMAGE: Lineament Detection application tool in Geological Images

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

### Using Conda (Recommended)

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) if you haven't already.

2. Clone this repository:
   ```
   git clone https://github.com/yourusername/domstudioimage.git
   cd domstudioimage
   ```

3. Create the Conda environment from the `environment.yml` file:
   ```
   conda env create -f environment.yml
   ```

4. Activate the Conda environment:
   ```
   conda activate domstudioimage
   ```

5. Run the main script:
   ```
   python main.py
   ```

### Manual Installation (Alternative)

If you prefer not to use Conda, you can install the dependencies manually:

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the main script:
   ```
   python main.py
   ```

Note: The pycoshrem package is already included in the main folder for easy access. If it doesn't work, install it using `pip install pycoshrem`.

## Usage

1. Load an image using the "Load Image" button
2. Adjust filter parameters using the sliders
3. Use the manual interpretation tools for fine-tuning
4. Export the results in your desired format


## Dependencies

See the `environment.yml` file for a complete list of dependencies.

## Developers

- Waqas Hussain

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPLv3) - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Thanks to all the open-source libraries that made this project possible.
- Special thanks to the geological community for their input and feedback.
