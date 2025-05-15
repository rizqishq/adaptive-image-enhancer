# adaptive-image-enhancer

An advanced image enhancement project for improving visibility in images, especially those taken in low-light or challenging conditions. Includes tools for batch processing and detailed visual comparison of original versus enhanced images.

## Features

*   **Advanced Image Enhancement**: Utilizes a multi-stage pipeline including:
    *   Illumination estimation and refinement.
    *   Adaptive gamma correction.
    *   Multi-scale fusion for balancing global and local contrast.
    *   Detail enhancement.
    *   Contrast Limited Adaptive Histogram Equalization (CLAHE).
    *   Denoising using a bilateral filter.
*   **Batch Processing**: Efficiently process entire directories of images.
*   **Configurable Parameters**: Adjust gamma range, detail strength, and denoising strength.
*   **Detailed Metrics**: Calculates PSNR, SSIM, and improvements in Entropy and Contrast. These are saved in a JSON summary for batch processing.
*   **Comparison Tool**: Generate detailed side-by-side visual comparisons of original and enhanced images, including their luminance channels and histograms.

## Prerequisites

*   Python 3.x
*   OpenCV (`cv2`)
*   NumPy
*   SciPy
*   scikit-image (`skimage`)
*   Matplotlib
*   tqdm
*   A C++ compiler might be required for some dependencies if installing from source (e.g., for OpenCV on certain systems).

## Setup

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <your-repository-url>
    cd adaptive-image-enhancer
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    Create a `requirements.txt` file with the following content:
    ```txt
    opencv-python
    numpy
    scipy
    scikit-image
    matplotlib
    tqdm
    # Add opencv-contrib-python if cv2.ximgproc.guidedFilter is used and not in base opencv-python
    # Consider opencv-contrib-python if you encounter issues with ximgproc
    opencv-contrib-python 
    ```
    Then run:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The project consists of two main scripts: `image_enhancement.py` for enhancing images and `comparison.py` for comparing original and enhanced images.

### 1. Image Enhancement (`image_enhancement.py`)

This script enhances input images using various image processing techniques.

**Arguments:**

*   `--input` (required): Path to the input image file or a directory containing images for batch processing.
*   `--output` (required): Path to save the enhanced image file (for single image) or a directory to save enhanced images and summary (for batch processing).
*   `--gamma-min` (optional): Minimum gamma value for adaptive gamma correction. Default: `0.4`.
*   `--gamma-max` (optional): Maximum gamma value for adaptive gamma correction. Default: `1.1`.
*   `--detail` (optional): Detail enhancement strength. Default: `0.3`.
*   `--denoise-strength` (optional): Denoising strength (h parameter for bilateral filter). Default: `5`.
*   `--batch` (optional): Flag to enable batch processing for a directory of images.

**Examples:**

*   **Enhance a single image:**
    ```bash
    python image_enhancement.py --input path/to/your/original_image.jpg --output path/to/your/enhanced_image.jpg
    ```

*   **Enhance all images in a directory (batch processing):**
    ```bash
    python image_enhancement.py --input path/to/original_images_dir/ --output path/to/enhanced_images_dir/ --batch
    ```
    In batch mode, enhanced images will be saved as `enhanced_<original_filename>.jpg` in the output directory. A `processing_summary.json` file will also be created in the output directory, containing processing details and metrics for each image.

### 2. Image Comparison (`comparison.py`)

This script generates a detailed visual comparison plot for an original image and its enhanced version. The plot includes the images themselves, their luminance (L) channels, and their color histograms.

**Arguments:**

*   `--original` (optional): Path to the original image file or a directory. If not provided, defaults to `./sample/`.
*   `--enhanced` (optional): Path to the enhanced image file or a directory. If not provided, defaults to `./results/`.
*   `--output` (optional): Directory where comparison plots will be saved. Default: `comparisons_detailed`.

**Examples:**

*   **Compare a single pair of images:**
    ```bash
    python comparison.py --original path/to/original.jpg --enhanced path/to/enhanced.jpg --output path/to/comparison_output_dir/
    ```
    This will save a `detailed_comparison_<original_filename_stem>.png` in the specified output directory.

*   **Compare images from default directories (`sample/` and `results/`):**
    If you have original images in a `./sample/` directory and their corresponding enhanced versions (e.g., `enhanced_image1.jpg` for `image1.jpg`) in a `./results/` directory:
    ```bash
    python comparison.py 
    ```
    Or specify the output directory:
    ```bash
    python comparison.py --output my_comparisons/
    ```

*   **Compare images from specified directories:**
    ```bash
    python comparison.py --original path/to/originals/ --enhanced path/to/enhanced_versions/ --output path/to/comparison_output_dir/
    ```
    The script will look for images in the original directory and try to find corresponding enhanced images (named `enhanced_<original_stem>.jpg`) in the enhanced directory.

## Output Files

*   **`image_enhancement.py`**:
    *   Single mode: The enhanced image file.
    *   Batch mode: Enhanced image files (e.g., `enhanced_name.jpg`) and `processing_summary.json` in the specified output directory.
*   **`comparison.py`**:
    *   `detailed_comparison_<original_stem>.png` files saved in the specified output directory.

## Notes

*   The `opencv-contrib-python` package might be needed if `cv2.ximgproc.guidedFilter` is not available in the standard `opencv-python` package for your environment. The `requirements.txt` above includes it.
*   Ensure the paths provided to the scripts are correct.
*   For batch comparison, the enhanced images are expected to be named following the pattern `enhanced_<original_image_stem>.jpg` relative to the original image names.