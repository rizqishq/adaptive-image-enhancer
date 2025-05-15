import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path
import argparse

def plot_channel_histograms(axes_subplot, bgr_image_data, plot_title):
    if len(bgr_image_data.shape) == 3:
        channel_colors = ('b', 'g', 'r')
        for channel_index, plot_color in enumerate(channel_colors):
            histogram = cv2.calcHist([bgr_image_data], [channel_index], None, [256], [0, 256])
            axes_subplot.plot(histogram, color=plot_color, alpha=0.7)
    else:
        histogram = cv2.calcHist([bgr_image_data], [0], None, [256], [0, 256])
        axes_subplot.plot(histogram, color='black', alpha=0.7)
    
    axes_subplot.set_xlim([0, 256])
    axes_subplot.set_title(plot_title)
    axes_subplot.grid(True, alpha=0.3)

def create_detailed_comparison_plot(original_image_path, enhanced_image_path, output_target_directory=None):
    original_bgr_image = cv2.imread(original_image_path)
    enhanced_bgr_image = cv2.imread(enhanced_image_path)
    
    if original_bgr_image is None:
        print(f"Error: Could not read original image at {original_image_path}")
        return False
    if enhanced_bgr_image is None:
        print(f"Error: Could not read enhanced image at {enhanced_image_path}")
        return False
    
    original_rgb_image = cv2.cvtColor(original_bgr_image, cv2.COLOR_BGR2RGB)
    enhanced_rgb_image = cv2.cvtColor(enhanced_bgr_image, cv2.COLOR_BGR2RGB)
    
    original_lab_image = cv2.cvtColor(original_bgr_image, cv2.COLOR_BGR2LAB)
    enhanced_lab_image = cv2.cvtColor(enhanced_bgr_image, cv2.COLOR_BGR2LAB)
    
    original_l_channel = original_lab_image[:,:,0]
    enhanced_l_channel = enhanced_lab_image[:,:,0]
    
    comparison_figure = plt.figure(figsize=(18, 14))
    
    ax_original_img = comparison_figure.add_subplot(3, 2, 1)
    ax_original_img.imshow(original_rgb_image)
    ax_original_img.set_title('Original Image')
    ax_original_img.axis('off')
    
    ax_enhanced_img = comparison_figure.add_subplot(3, 2, 2)
    ax_enhanced_img.imshow(enhanced_rgb_image)
    ax_enhanced_img.set_title('Enhanced Image')
    ax_enhanced_img.axis('off')
    
    ax_original_l = comparison_figure.add_subplot(3, 2, 3)
    ax_original_l.imshow(original_l_channel, cmap='gray')
    ax_original_l.set_title('Original Luminance (L-channel)')
    ax_original_l.axis('off')
    
    ax_enhanced_l = comparison_figure.add_subplot(3, 2, 4)
    ax_enhanced_l.imshow(enhanced_l_channel, cmap='gray')
    ax_enhanced_l.set_title('Enhanced Luminance (L-channel)')
    ax_enhanced_l.axis('off')
    
    ax_original_hist = comparison_figure.add_subplot(3, 2, 5)
    plot_channel_histograms(ax_original_hist, original_bgr_image, 'Original Image Histogram')
    
    ax_enhanced_hist = comparison_figure.add_subplot(3, 2, 6)
    plot_channel_histograms(ax_enhanced_hist, enhanced_bgr_image, 'Enhanced Image Histogram')
    
    base_image_name = Path(original_image_path).stem
    comparison_figure.suptitle(f"Detailed Image Comparison: {base_image_name}", fontsize=18, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    if output_target_directory:
        os.makedirs(output_target_directory, exist_ok=True)
        comparison_image_save_path = os.path.join(output_target_directory, f"detailed_comparison_{base_image_name}.png")
        try:
            plt.savefig(comparison_image_save_path, dpi=300, bbox_inches='tight')
            print(f"Detailed comparison plot saved to: {comparison_image_save_path}")
        except Exception as e:
            print(f"Error saving comparison plot to {comparison_image_save_path}: {e}")
            return False
        finally:
            plt.close(comparison_figure)
        return comparison_image_save_path
    else:
        plt.show()
        return True

def main():
    argument_parser = argparse.ArgumentParser(description='Create detailed comparison plots between original and enhanced images.')
    argument_parser.add_argument('--original', type=str, help='Path to the original image or a directory containing original images.')
    argument_parser.add_argument('--enhanced', type=str, help='Path to the enhanced image or a directory containing corresponding enhanced images.')
    argument_parser.add_argument('--output', type=str, default='comparisons_detailed', help='Directory where comparison plots will be saved. Default is "comparisons_detailed".')
    
    cli_arguments = argument_parser.parse_args()
    
    original_input_path = cli_arguments.original
    enhanced_input_path = cli_arguments.enhanced
    output_plots_directory = cli_arguments.output

    if not original_input_path and not enhanced_input_path:
        default_original_dir = 'sample'
        default_enhanced_dir = 'results'
        print(f"Info: --original and --enhanced paths not provided. Attempting to use default directories.")
        if os.path.exists(default_original_dir):
            original_input_path = default_original_dir
            print(f"Info: Using default original directory: '{default_original_dir}'")
        else:
            print(f"Error: Default original directory '{default_original_dir}' not found. Please specify --original path.")
            return 1
            
        if os.path.exists(default_enhanced_dir):
            enhanced_input_path = default_enhanced_dir
            print(f"Info: Using default enhanced directory: '{default_enhanced_dir}'")
        else:
            print(f"Error: Default enhanced directory '{default_enhanced_dir}' not found. Please specify --enhanced path.")
            return 1
    elif not original_input_path or not enhanced_input_path:
        argument_parser.error("Both --original and --enhanced paths must be specified if not using default directories.")

    if os.path.isdir(original_input_path) and os.path.isdir(enhanced_input_path):
        print(f"Processing directories: Comparing images from '{original_input_path}' with enhanced versions in '{enhanced_input_path}'.")
        
        original_image_file_paths = []
        supported_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff')
        for extension_pattern in supported_extensions:
            original_image_file_paths.extend(glob.glob(os.path.join(original_input_path, extension_pattern)))
        
        if not original_image_file_paths:
            print(f"No images found in the original directory: {original_input_path}")
            return 0

        successful_comparisons_count = 0
        for single_original_path in original_image_file_paths:
            original_image_stem = Path(single_original_path).stem
            corresponding_enhanced_path = os.path.join(enhanced_input_path, f"enhanced_{original_image_stem}.jpg")
            
            if os.path.exists(corresponding_enhanced_path):
                saved_plot_path = create_detailed_comparison_plot(single_original_path, corresponding_enhanced_path, output_plots_directory)
                if saved_plot_path:
                    successful_comparisons_count += 1
            else:
                print(f"Warning: Enhanced version for '{original_image_stem}' not found at '{corresponding_enhanced_path}'. Skipping.")
        
        print(f"Finished. Created {successful_comparisons_count} detailed comparison plot(s) in '{output_plots_directory}'.")
    
    elif os.path.isfile(original_input_path) and os.path.isfile(enhanced_input_path):
        print(f"Processing single files: Comparing '{original_input_path}' with '{enhanced_input_path}'.")
        create_detailed_comparison_plot(original_input_path, enhanced_input_path, output_plots_directory)
    
    else:
        print("Error: Input paths for --original and --enhanced must both be either valid files or valid directories.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code) 