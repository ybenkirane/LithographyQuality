import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.feature import register_translation
from scipy.ndimage import fourier_shift


def preprocess_images(pattern_image_path, afm_scan_image_path):
    # Read images
    pattern_image = cv2.imread(pattern_image_path, cv2.IMREAD_GRAYSCALE)
    afm_scan_image = cv2.imread(afm_scan_image_path, cv2.IMREAD_GRAYSCALE)

    # Resize images, if necessary
    if pattern_image.shape != afm_scan_image.shape:
        afm_scan_image = cv2.resize(afm_scan_image, pattern_image.shape[::-1])

    # Normalize images
    pattern_image = pattern_image / 255.0
    afm_scan_image = afm_scan_image / 255.0

    return pattern_image, afm_scan_image


def align_images(pattern_image, afm_scan_image):
    # Register images using phase correlation
    shift, error, diffphase = register_translation(pattern_image, afm_scan_image)

    # Apply the shift to the AFM scan image
    aligned_afm_scan_image = np.fft.ifftn(
        fourier_shift(np.fft.fftn(afm_scan_image), shift)
    ).real

    return aligned_afm_scan_image


def main():
    pattern_image_path = "path/to/pattern/image.png"
    afm_scan_image_path = "path/to/afm/scan/image.png"

    # Preprocess images
    pattern_image, afm_scan_image = preprocess_images(
        pattern_image_path, afm_scan_image_path
    )

    # Align images
    aligned_afm_scan_image = align_images(pattern_image, afm_scan_image)

    # Calculate SSIM
    ssim_value = ssim(pattern_image, aligned_afm_scan_image)

    print("SSIM value:", ssim_value)


if __name__ == "__main__":
    main()
