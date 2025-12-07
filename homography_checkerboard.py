import numpy as np
import cv2
import matplotlib.pyplot as plt


# ------------ 1. Helper: DLT homography estimation ------------- #

def compute_homography_dlt(pts1, pts2):
    """
    Compute 3x3 homography H such that pts2 ~ H @ pts1
    using the (unnormalized) DLT algorithm.

    pts1, pts2: arrays of shape (N, 2) with corresponding points.
    """
    assert pts1.shape == pts2.shape
    N = pts1.shape[0]
    assert N >= 4, "Need at least 4 points for homography"

    A = []
    for i in range(N):
        x, y = pts1[i, :]
        xp, yp = pts2[i, :]
        # Each correspondence gives 2 rows
        # [-x, -y, -1,  0,  0,  0, x*xp, y*xp, xp]
        # [ 0,  0,  0, -x, -y, -1, x*yp, y*yp, yp]
        A.append([-x, -y, -1, 0, 0, 0, x * xp, y * xp, xp])
        A.append([0, 0, 0, -x, -y, -1, x * yp, y * yp, yp])

    A = np.asarray(A)
    # Solve Ah = 0 via SVD
    U, S, Vt = np.linalg.svd(A)
    h = Vt[-1, :]           # last row of V^T (smallest singular value)
    H = h.reshape((3, 3))
    # Normalize so that H[2,2] = 1
    if H[2, 2] != 0:
        H = H / H[2, 2]
    return H


# ------------ 2. Main pipeline ------------- #

def main():
    
    # ---- USER: set your filenames here ----
    img1_path = "img_homo/left01.jpg"   # first view
    img2_path = "img_homo/left09.jpg"   # second view

    # Chessboard pattern size: number of INNER corners per row and column.
    # This is a guess: try (9, 6) first; if it fails, adjust.
    pattern_size = (9, 6)  # (num_corners_across, num_corners_down)
    
    # ---- Load images ----
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None:
        raise FileNotFoundError(f"Could not load image '{img1_path}'")
    if img2 is None:
        raise FileNotFoundError(f"Could not load image '{img2_path}'")

    print(f"Loaded images: {img1_path} and {img2_path}")
    h2, w2 = img2.shape[:2]

    # ---- Find chessboard corners in both images ----
    # Flags improve robustness a bit
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

    ret1, corners1 = cv2.findChessboardCorners(img1, pattern_size, flags)
    ret2, corners2 = cv2.findChessboardCorners(img2, pattern_size, flags)

    if not ret1 or not ret2:
        raise RuntimeError(
            f"Chessboard not found in one or both images with pattern_size={pattern_size}. "
            f"Try changing pattern_size."
        )

    print("Chessboard corners detected in both images.")

    # Refine corner locations to subpixel accuracy
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners1 = cv2.cornerSubPix(img1, corners1, (11, 11), (-1, -1), criteria)
    corners2 = cv2.cornerSubPix(img2, corners2, (11, 11), (-1, -1), criteria)

    # corners1, corners2 are (N, 1, 2) -> reshape to (N, 2)
    pts1 = corners1.reshape(-1, 2)
    pts2 = corners2.reshape(-1, 2)
    N = pts1.shape[0]
    print(f"Number of correspondences: {N}")

    # ---- Estimate H via OpenCV's findHomography (like tutorial) ----
    H_cv, mask = cv2.findHomography(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    print("\nHomography from cv2.findHomography (RANSAC):")
    print(H_cv)

    # ---- Estimate H via our DLT implementation ----
    H_dlt = compute_homography_dlt(pts1, pts2)
    print("\nHomography from custom DLT:")
    print(H_dlt)

    # ---- Compute transfer error for both H's ----
    def transfer_error(H, pts_src, pts_dst):
        pts_src_h = np.hstack([pts_src, np.ones((pts_src.shape[0], 1))])  # (N, 3)
        pts_proj_h = (H @ pts_src_h.T).T                                  # (N, 3)
        pts_proj = pts_proj_h[:, :2] / pts_proj_h[:, 2:3]
        errors = np.linalg.norm(pts_dst - pts_proj, axis=1)
        return errors

    errors_cv = transfer_error(H_cv, pts1, pts2)
    errors_dlt = transfer_error(H_dlt, pts1, pts2)

    print("\nTransfer errors (OpenCV H):")
    print(f"  mean: {errors_cv.mean():.4f} px,  median: {np.median(errors_cv):.4f} px,  max: {errors_cv.max():.4f} px")

    print("\nTransfer errors (DLT H):")
    print(f"  mean: {errors_dlt.mean():.4f} px,  median: {np.median(errors_dlt):.4f} px,  max: {errors_dlt.max():.4f} px")

    # ---- Warp img1 to img2's coordinate frame using both homographies ----
    warped_cv = cv2.warpPerspective(img1, H_cv, (w2, h2))
    warped_dlt = cv2.warpPerspective(img1, H_dlt, (w2, h2))

        # ---- Visualization ----
    # Convert to RGB
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    warped_cv_rgb = cv2.cvtColor(warped_cv, cv2.COLOR_GRAY2RGB)
    warped_dlt_rgb = cv2.cvtColor(warped_dlt, cv2.COLOR_GRAY2RGB)

    # ----- (1) Simple side-by-side comparison -----
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.title("Image 1")
    plt.imshow(img1_rgb)
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.title("Image 2 (target)")
    plt.imshow(img2_rgb)
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.title("Warped img1 with H (OpenCV)")
    plt.imshow(warped_cv_rgb)
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.title("Warped img1 with H (DLT)")
    plt.imshow(warped_dlt_rgb)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # ----- (2) Color-coded overlay (OpenCV H) -----
    # Make img2 green, warped img1 red
    base = img2_rgb.copy().astype(np.float32) / 255.0
    warp = warped_cv_rgb.copy().astype(np.float32) / 255.0

    overlay_color = np.zeros_like(base)
    overlay_color[..., 1] = base[..., 0]      # green channel from img2
    overlay_color[..., 0] = warp[..., 0]      # red channel from warped

    plt.figure(figsize=(6, 5))
    plt.title("Color overlay (red = warped img1, green = img2)")
    plt.imshow(overlay_color)
    plt.axis("off")
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()
