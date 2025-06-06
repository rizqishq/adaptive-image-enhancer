import matplotlib
matplotlib.use('Agg') 
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import metrics, exposure
import os
from tqdm import tqdm
import argparse
from datetime import datetime
import json
import glob
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

def estimate_illumination(grayscale_image, radius=15, epsilon=1e-3):
    min_intensity_channel = cv2.erode(grayscale_image, np.ones((3, 3), np.uint8))
    guided_filter_output = cv2.ximgproc.guidedFilter(grayscale_image, min_intensity_channel, radius, epsilon)
    return guided_filter_output

def refine_illumination_map(illumination_map, kernel_size=15):
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed_illumination = cv2.morphologyEx(illumination_map, cv2.MORPH_CLOSE, morph_kernel)
    smoothed_illumination = gaussian_filter(closed_illumination, sigma=3)
    return smoothed_illumination

def adaptive_gamma_correction(illumination_map_normalized, gamma_min=0.4, gamma_max=1.1):
    mean_illumination = np.mean(illumination_map_normalized)
    std_illumination = np.std(illumination_map_normalized)
    
    if mean_illumination < 0.3 and std_illumination < 0.2:
        gamma_value = gamma_max
    elif mean_illumination > 0.7 and std_illumination > 0.3:
        gamma_value = gamma_min
    else:
        gamma_value = np.interp(mean_illumination, [0.1, 0.9], [gamma_max, gamma_min])
        
    gamma_corrected_illumination = np.power(illumination_map_normalized, gamma_value)
    brightness_factor = 1.0 + (0.3 * (1.0 - mean_illumination))
    brightened_illumination = gamma_corrected_illumination * brightness_factor
    
    return np.clip(brightened_illumination, 0, 1)

def multi_scale_fusion(luminance_normalized, corrected_illumination_map):
    global_enhanced_luminance = np.clip(luminance_normalized / (corrected_illumination_map + 1e-6), 0, 1)
    
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
    luminance_8bit = (luminance_normalized * 255).astype(np.uint8)
    local_enhanced_luminance = clahe.apply(luminance_8bit).astype(np.float32) / 255.0
    
    global_histogram_equalized_luminance = cv2.equalizeHist(luminance_8bit).astype(np.float32) / 255.0
    
    mean_luminance_intensity = np.mean(luminance_normalized)
    std_luminance_intensity = np.std(luminance_normalized)
    
    if mean_luminance_intensity < 0.3:
        fusion_weights = [0.4, 0.4, 0.2] if std_luminance_intensity < 0.2 else [0.5, 0.3, 0.2]
    else:
        fusion_weights = [0.3, 0.5, 0.2] if std_luminance_intensity > 0.3 else [0.4, 0.4, 0.2]
        
    fused_luminance = (fusion_weights[0] * global_enhanced_luminance +
                       fusion_weights[1] * local_enhanced_luminance +
                       fusion_weights[2] * global_histogram_equalized_luminance)
                       
    return np.clip(fused_luminance, 0, 1)

def enhance_image_details(base_luminance_normalized, enhanced_luminance_normalized, detail_strength_param=0.3):
    gaussian_sigma = max(1.0, min(2.0, base_luminance_normalized.shape[0] / 1000.0))
    blurred_luminance = gaussian_filter(base_luminance_normalized, sigma=gaussian_sigma)
    detail_layer = base_luminance_normalized - blurred_luminance
    
    mean_luminance_val = np.mean(base_luminance_normalized)
    std_luminance_val = np.std(base_luminance_normalized)
    
    current_detail_strength = detail_strength_param
    if mean_luminance_val < 0.3:
        current_detail_strength = 0.25 if std_luminance_val < 0.2 else 0.2
    elif mean_luminance_val > 0.7:
        current_detail_strength = 0.35 if std_luminance_val > 0.3 else 0.3
        
    final_detailed_luminance = enhanced_luminance_normalized + current_detail_strength * detail_layer
    return np.clip(final_detailed_luminance, 0, 1)

def denoise_image_bgr(image_float_0_1, h_parameter=5):
    image_8bit = (image_float_0_1 * 255).astype(np.uint8)
    denoised_8bit = cv2.bilateralFilter(image_8bit, 9, h_parameter, h_parameter)
    return denoised_8bit.astype(np.float32) / 255.0

def enhance_image(input_image_path, output_image_path=None, enhancement_params=None):
    current_enhancement_params = {
        'gamma_min': 0.4, 'gamma_max': 1.1,
        'brightness_boost': 1.3,
        'clahe_clip_limit': 1.2,
        'detail_strength': 0.3,
        'denoise_strength': 5
    }
    if enhancement_params:
        current_enhancement_params.update(enhancement_params)

    original_bgr_image = cv2.imread(input_image_path)
    if original_bgr_image is None:
        raise ValueError(f"Image not found or could not be read: {input_image_path}")

    ycrcb_image = cv2.cvtColor(original_bgr_image, cv2.COLOR_BGR2YCrCb)
    y_channel_normalized = ycrcb_image[:,:,0].astype(np.float32) / 255.0
    
    estimated_illumination = estimate_illumination(y_channel_normalized, radius=15, epsilon=1e-3)
    clipped_illumination = np.clip(estimated_illumination, 0, 1)
    
    refined_illumination = refine_illumination_map(clipped_illumination, kernel_size=15)
    clipped_refined_illumination = np.clip(refined_illumination, 0, 1)
    
    corrected_illumination = adaptive_gamma_correction(clipped_refined_illumination)

    fused_luminance = multi_scale_fusion(y_channel_normalized, corrected_illumination)

    detail_enhanced_luminance = enhance_image_details(
        y_channel_normalized, 
        fused_luminance, 
        current_enhancement_params['detail_strength']
    )

    contrast_enhanced_luminance = exposure.equalize_adapthist(detail_enhanced_luminance, clip_limit=0.02)
    clipped_contrast_luminance = np.clip(contrast_enhanced_luminance, 0, 1)
    
    denoised_luminance_normalized = denoise_image_bgr(
        clipped_contrast_luminance, 
        h_parameter=current_enhancement_params['denoise_strength']
    )
    
    final_ycrcb_image = ycrcb_image.copy()
    final_ycrcb_image[:,:,0] = (denoised_luminance_normalized * 255).astype(np.uint8)
    
    final_enhanced_bgr_image = cv2.cvtColor(final_ycrcb_image, cv2.COLOR_YCrCb2BGR)
    
    if output_image_path:
        output_directory = os.path.dirname(output_image_path)
        if output_directory and not os.path.exists(output_directory):
             os.makedirs(output_directory, exist_ok=True)
        cv2.imwrite(output_image_path, final_enhanced_bgr_image)
        
    return final_enhanced_bgr_image, current_enhancement_params

def process_image_in_parallel(image_file_path, batch_main_output_dir, enhancement_parameters):
    image_stem_name = Path(image_file_path).stem
    individual_enhanced_image_path = os.path.join(batch_main_output_dir, f"enhanced_{image_stem_name}.jpg")

    try:
        enhanced_image_data = enhance_image(
            image_file_path,
            individual_enhanced_image_path,
            enhancement_parameters
        )
        print(f"Successfully enhanced: {image_stem_name}")
        return True
    except Exception as e:
        print(f"Failed to enhance {image_stem_name}: {str(e)}")
        return False

def process_directory_batch(input_images_dir, main_batch_output_dir, enhancement_parameters=None):
    print("\n=== BATCH PROCESSING ===")
    print(f"Input Directory: {input_images_dir}")
    print(f"Output Directory: {main_batch_output_dir}")
    os.makedirs(main_batch_output_dir, exist_ok=True)
    
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff')
    image_file_paths = []
    for ext_pattern in image_extensions:
        image_file_paths.extend(glob.glob(os.path.join(input_images_dir, ext_pattern)))
    
    if not image_file_paths:
        print(f"No image files found in {input_images_dir} with extensions {image_extensions}")
        return

    num_images_found = len(image_file_paths)
    print(f"\nFound {num_images_found} images to process.")
    
    cpu_cores = os.cpu_count()
    max_workers = min(cpu_cores if cpu_cores else 4, num_images_found) 
    
    print(f"\nProcessing with up to {max_workers} worker(s)...")
    
    successful_count = 0
    with tqdm(total=num_images_found, desc="Processing Images") as progress_bar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_image_in_parallel, img_path, main_batch_output_dir, enhancement_parameters)
                for img_path in image_file_paths
            ]
            for future in futures:
                if future.result():
                    successful_count += 1
                progress_bar.update(1)

    print("\n=== PROCESSING SUMMARY ===")
    print(f"Total Images: {num_images_found}")
    print(f"Successfully Enhanced: {successful_count}")
    print(f"Failed: {num_images_found - successful_count}")

def main():
    parser = argparse.ArgumentParser(description='Image Enhancement')
    parser.add_argument('--input', type=str, required=True, help='Input image path or directory for batch processing.')
    parser.add_argument('--output', type=str, required=True, help='Output image path (for single image) or directory (for batch processing).')
    parser.add_argument('--gamma-min', type=float, default=0.4, help='Minimum gamma value for adaptive gamma correction.')
    parser.add_argument('--gamma-max', type=float, default=1.1, help='Maximum gamma value for adaptive gamma correction.')
    parser.add_argument('--detail', type=float, default=0.3, help='Detail enhancement strength.')
    parser.add_argument('--denoise-strength', type=int, default=5, help='Denoising strength (h parameter for bilateral filter).')
    parser.add_argument('--batch', action='store_true', help='Enable batch processing for a directory of images.')
    
    cli_args = parser.parse_args()

    enhancement_params_from_cli = {
        'gamma_min': cli_args.gamma_min,
        'gamma_max': cli_args.gamma_max,
        'detail_strength': cli_args.detail,
        'denoise_strength': cli_args.denoise_strength
    }

    try:
        if cli_args.batch:
            input_path_arg = cli_args.input
            output_path_arg = cli_args.output
            if not os.path.isdir(input_path_arg):
                print(f"Error: For batch processing, input '{input_path_arg}' must be a directory.")
                return 1
            
            batch_output_target_dir = output_path_arg
            if os.path.isfile(batch_output_target_dir):
                 print(f"Warning: Output path '{batch_output_target_dir}' for batch mode appears to be a file. Using its parent directory: '{os.path.dirname(batch_output_target_dir)}'.")
                 batch_output_target_dir = os.path.dirname(batch_output_target_dir)
            
            os.makedirs(batch_output_target_dir, exist_ok=True)
            process_directory_batch(input_path_arg, batch_output_target_dir, enhancement_params_from_cli)
        else:
            input_file_arg = cli_args.input
            single_output_file_path = cli_args.output

            if not os.path.isfile(input_file_arg):
                print(f"Error: Input file '{input_file_arg}' not found or is not a file.")
                return 1
            
            single_output_file_directory = os.path.dirname(single_output_file_path)
            if single_output_file_directory and not os.path.exists(single_output_file_directory):
                os.makedirs(single_output_file_directory, exist_ok=True)

            print("\n=== SINGLE IMAGE ENHANCEMENT ===")
            print(f"Processing: {input_file_arg}")
            print(f"Output to: {single_output_file_path}")

            enhanced_image_data = enhance_image(
                input_file_arg, 
                single_output_file_path, 
                enhancement_params_from_cli
            )
            print("\nEnhancement completed successfully!")

    except Exception as e:
        print("\n=== ERROR ===")
        print(f"An error occurred: {str(e)}")
        return 1
    
    print("\n=== PROCESSING COMPLETE ===")
    return 0

if __name__ == "__main__":
    exit_status = main()
    exit(exit_status)
