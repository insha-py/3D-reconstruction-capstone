import numpy as np
import cv2
import matplotlib.pyplot as plt

def main():
    img1_path = "img_homo/leuvenA.jpg"   # first view
    img2_path = "img_homo/leuvenB.jpg"   # second view

    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        raise FileNotFoundError("Could not load leuven images")

    print("Loaded images:", img1.shape, img2.shape)
    h2, w2 = img2.shape[:2]

    # ---- 1. Detect and describe features (ORB for simplicity) ----
    orb = cv2.ORB_create(2000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    print(f"Found {len(kp1)} and {len(kp2)} keypoints")

    # ---- 2. Match features with brute-force Hamming matcher ----
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    print("Raw matches:", len(matches))

    # Sort matches by descriptor distance (best first)
    matches = sorted(matches, key=lambda m: m.distance)
    good_matches = matches[:200]  # keep best N

    print("Using good matches:", len(good_matches))

    # Extract point coordinates
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    # ---- 3. Estimate homography with RANSAC ----
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
    inliers = mask.ravel().sum()
    print("Estimated H (RANSAC), inliers:", inliers, "/", len(good_matches))
    print(H)

    # ---- 4. Warp image 1 into image 2's frame ----
    warped = cv2.warpPerspective(img1, H, (w2, h2))

    # ---- 5. Visualization ----
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    warped_rgb = cv2.cvtColor(warped, cv2.COLOR_GRAY2RGB)

    # Color overlay: red = warped img1, green = img2
    base = img2_rgb.astype(np.float32) / 255.0
    warp = warped_rgb.astype(np.float32) / 255.0

    overlay = np.zeros_like(base)
    overlay[..., 1] = base[..., 0]   # green = target
    overlay[..., 0] = warp[..., 0]   # red = warped

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.title("Image 1")
    plt.imshow(img1_rgb)
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.title("Image 2")
    plt.imshow(img2_rgb)
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.title("Warped Image 1 (using H)")
    plt.imshow(warped_rgb)
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.title("Overlay (red = warped, green = img2)")
    plt.imshow(overlay)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # ---- 6. Optional: compute symmetric transfer error on inliers ----
    inlier_pts1 = pts1[mask.ravel() == 1]
    inlier_pts2 = pts2[mask.ravel() == 1]

    pts1_h = np.hstack([inlier_pts1, np.ones((inlier_pts1.shape[0], 1))])
    pts2_h = np.hstack([inlier_pts2, np.ones((inlier_pts2.shape[0], 1))])

    proj1_to_2 = (H @ pts1_h.T).T
    proj1_to_2 = proj1_to_2[:, :2] / proj1_to_2[:, 2:3]

    H_inv = np.linalg.inv(H)
    proj2_to_1 = (H_inv @ pts2_h.T).T
    proj2_to_1 = proj2_to_1[:, :2] / proj2_to_1[:, 2:3]

    err_1to2 = np.linalg.norm(inlier_pts2 - proj1_to_2, axis=1)
    err_2to1 = np.linalg.norm(inlier_pts1 - proj2_to_1, axis=1)
    sym_err = (err_1to2 + err_2to1) / 2.0

    print(f"Symmetric transfer error: mean={sym_err.mean():.3f} px, "
          f"median={np.median(sym_err):.3f} px, max={sym_err.max():.3f} px")


if __name__ == "__main__":
    main()
