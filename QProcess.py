import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.spatial.distance import directed_hausdorff


def preprocess_image(image_path):
    """
    Preprocess the image by converting it to grayscale and applying adaptive thresholding.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    thresholded = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return thresholded


def align_images(reference, target):
    """
    Align the target image to the reference image using ORB feature matching and homography.
    """
    orb = cv2.ORB_create()
    kp_ref, des_ref = orb.detectAndCompute(reference, None)
    kp_tar, des_tar = orb.detectAndCompute(target, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_ref, des_tar)

    src_pts = np.float32([kp_ref[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_tar[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    M, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    aligned_target = cv2.warpPerspective(
        target, M, (reference.shape[1], reference.shape[0])
    )
    return aligned_target


def extract_contours(image):
    """
    Extract contours from the binary image.
    """
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def compare_contours(contours_ref, contours_tar):
    """
    Compare the reference and target contours using the Hausdorff distance.
    """
    similarity_scores = []
    for contour_ref, contour_tar in zip(contours_ref, contours_tar):
        contour_ref_2d = contour_ref.reshape(-1, 2)
        contour_tar_2d = contour_tar.reshape(-1, 2)
        score = max(
            directed_hausdorff(contour_ref_2d, contour_tar_2d)[0],
            directed_hausdorff(contour_tar_2d, contour_ref_2d)[0],
        )
        similarity_scores.append(score)
    return similarity_scores


def compute_quality_score(similarity_scores, defect_count, quality_weight=0.8):
    """
    Compute the quality score by combining the similarity scores and defect count.
    """
    max_hausdorff_distance = max(similarity_scores)

    if max_hausdorff_distance == 0:
        normalized_scores = [1 for score in similarity_scores]
    else:
        normalized_scores = [
            1 - (score / max_hausdorff_distance) for score in similarity_scores
        ]

    similarity_score = np.mean(normalized_scores)
    defect_score = 1 - (defect_count / len(similarity_scores))
    quality_score = (
        quality_weight * similarity_score + (1 - quality_weight) * defect_score
    )
    return quality_score


def main():
    # Load and preprocess images
    ref_image_path = "NF_Input.png"
    tar_image_path = "NF_Output.png"
    ref_image = preprocess_image(ref_image_path)
    tar_image = preprocess_image(tar_image_path)

    # Align the target image to the reference image
    aligned_tar_image = align_images(ref_image, tar_image)

    # Extract contours from the reference and target images
    contours_ref = extract_contours(ref_image)
    contours_tar = extract_contours(aligned_tar_image)

    # Compare contours and calculate similarity scores
    similarity_scores = compare_contours(contours_ref, contours_tar)

    # Calculate defect count
    defect_count = abs(len(contours_ref) - len(contours_tar))

    # Compute the quality score
    quality_score = compute_quality_score(similarity_scores, defect_count)

    # Print the results
    print(f"Quality Score: {quality_score}")
    print(f"Defect Count: {defect_count}")


if __name__ == "__main__":
    main()
