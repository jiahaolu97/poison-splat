import os
import shutil
import cv2
import numpy as np

ns_scenes = ['chair', 'drums', 'mic', 'ship']
mip_scenes = ['bicycle', 'flowers', 'garden', 'room']

# Function to apply Gaussian filter
def gaussian_filter(image, kernel_size=5, sigma=1.5):
    kernel = cv2.getGaussianKernel(kernel_size, sigma)
    gaussian_kernel = kernel @ kernel.T
    return cv2.filter2D(image, -1, gaussian_kernel)

# Function to apply Low-Pass filter using FFT
def low_pass_filter(image, cutoff=50):
    rows, cols = image.shape[:2]
    crow, ccol = rows // 2, cols // 2

    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    mask = np.zeros((rows, cols, 2), np.uint8)
    cv2.circle(mask, (ccol, crow), cutoff, (1, 1), -1)

    filtered_dft = dft_shift * mask
    dft_ishift = np.fft.ifftshift(filtered_dft)
    filtered_image = cv2.idft(dft_ishift)
    return cv2.magnitude(filtered_image[:, :, 0], filtered_image[:, :, 1])

# Function to apply Bilateral filter
def bilateral_filter(image, diameter=9, sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)


def smooth_Nerf_Synthetic(setting, scene):
    assert setting in ['', '_eps16']
    # Paths
    source_dir = f"dataset/Nerf_Synthetic{setting}/{scene}/train/"
    target_base_dir = f"dataset/Nerf_Synthetic{setting}_smooth/"
    filters = ["GaussFilter", "BilateralFilter"]
    common_files = ["points3d.ply", "transforms_test.json", "transforms_train.json", "transforms_val.json"]

    # Create target directories for filters
    for filter_type in filters:
        target_dir = os.path.join(target_base_dir, filter_type, f"{scene}/train")
        os.makedirs(target_dir, exist_ok=True)

    # Apply filters to images
    for i in range(100):
        filename = f"r_{i}.png"
        filepath = os.path.join(source_dir, filename)
        if not os.path.exists(filepath):
            print(f"File {filename} not found in source directory.")
            continue

        # Read image
        image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        # Apply Gaussian filter
        gaussian_smoothed = gaussian_filter(image)
        gauss_target = os.path.join(target_base_dir, "GaussFilter", f"{scene}/train", filename)
        cv2.imwrite(gauss_target, gaussian_smoothed)

        # # Apply Low-Pass filter
        # lowpass_smoothed = low_pass_filter(image)
        # lowpass_target = os.path.join(target_base_dir, "LowPassFilter", "chair/train", filename)
        # cv2.imwrite(lowpass_target, lowpass_smoothed)

        # Apply Bilateral filter
        bilateral_smoothed = bilateral_filter(image)
        bilateral_target = os.path.join(target_base_dir, "BilateralFilter", f"{scene}/train", filename)
        cv2.imwrite(bilateral_target, bilateral_smoothed)

    # Copy common files
    for file in common_files:
        source_file = os.path.join(f"dataset/Nerf_Synthetic{setting}/{scene}/", file)
        if not os.path.exists(source_file):
            print(f"File {file} not found in source directory.")
            continue

        for filter_type in filters:
            target_file = os.path.join(target_base_dir, filter_type, scene, file)
            target_dir = os.path.dirname(target_file)
            os.makedirs(target_dir, exist_ok=True)
            shutil.copy(source_file, target_file)

    print("Processing complete.")

def smooth_MIP_360(setting, scene):
    assert setting in ['', '_eps16']
    # Paths
    source_base_dir = f"dataset/MIP_Nerf_360{setting}/{scene}/"
    source_image_dir = f"{source_base_dir}images/"
    target_base_dir = f"dataset/MIP_Nerf_360{setting}_smooth/"
    filters = ["GaussFilter", "BilateralFilter"]

    # Create target directories for filters
    for filter_type in filters:
        target_dir = os.path.join(target_base_dir, filter_type, f"{scene}/images")
        os.makedirs(target_dir, exist_ok=True)

    # For all images under source_dir
    for filename in os.listdir(source_image_dir):
        print(f"Processing image {filename}")
        filepath = os.path.join(source_image_dir, filename)
        # Read image
        image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        
        # Apply Gaussian filter
        gaussian_smoothed = gaussian_filter(image)
        gauss_target = os.path.join(target_base_dir, "GaussFilter", f"{scene}/images", filename)
        cv2.imwrite(gauss_target, gaussian_smoothed)

        # Apply Bilateral filter
        bilateral_smoothed = bilateral_filter(image)
        bilateral_target = os.path.join(target_base_dir, "BilateralFilter", f"{scene}/images", filename)
        cv2.imwrite(bilateral_target, bilateral_smoothed)

    # Copy common files
    source_file = os.path.join(source_base_dir, "poses_bounds.npy")
    target_file = os.path.join(target_base_dir, "BilateralFilter", f"{scene}/poses_bounds.npy")
    shutil.copy(source_file, target_file)
    target_file = os.path.join(target_base_dir, "GaussFilter", f"{scene}/poses_bounds.npy")
    shutil.copy(source_file, target_file)

    # copy 'sparse' folder
    source_folder = os.path.join(source_base_dir, "sparse")
    target_folder = os.path.join(target_base_dir, "BilateralFilter", f"{scene}/sparse")
    shutil.copytree(source_folder, target_folder)
    target_folder = os.path.join(target_base_dir, "GaussFilter", f"{scene}/sparse")
    shutil.copytree(source_folder, target_folder)

    print("Processing complete.")


# for setting in ['', '_eps16']:
#     for scene in ['chair', 'drums', 'mic', 'ship']:
#         smooth_Nerf_Synthetic(setting, scene)

for setting in ['', '_eps16']:
    for scene in ['flowers']:
        smooth_MIP_360(setting, scene)