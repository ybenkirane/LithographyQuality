import cv2
import numpy as np
from skimage import io
from skimage.metrics import structural_similarity as ssim
from skimage.transform import rescale, resize

def register_images(img1, img2):
    # Convert images to grayscale
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Find keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1_gray, None)
    kp2, des2 = orb.detectAndCompute(img2_gray, None)

    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(des1, des2)

    # Sort them in ascending order of their distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract the matched points
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Estimate the homography
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

    # Warp the original image to align with the generated pattern
    height, width, _ = img2.shape
    aligned_img1 = cv2.warpPerspective(img1, H, (width, height))

    return aligned_img1

def compute_quality(original_img, generated_img):
    # Read the images
    img1 = io.imread(original_img)
    img2 = io.imread(generated_img)

    # Resize img2 to match img1's dimensions
    img2_resized = resize(img2, (img1.shape[0], img1.shape[1]))

    # Register the images
    aligned_img1 = register_images(img1, img2_resized)

    # Convert the images to grayscale
    img1_gray = cv2.cvtColor(aligned_img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)

    # Compute SSIM
    quality = ssim(img1_gray, img2_gray)

    return quality

original_img = 'path/to/original_image.png'
generated_img = 'path/to/generated_image.png'

quality = compute_quality(original_img, generated_img)
print(f"Quality of the lithography pattern: {quality:.4f}")
