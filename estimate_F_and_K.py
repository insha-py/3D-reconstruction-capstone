import numpy as np
import cv2

# =========================
# 1. Point normalization
# =========================

def normalize_points(pts):
    """
    Normalize 2D points so that:
      - centroid is at the origin
      - average distance to origin is sqrt(2)
    pts: (N, 2)

    Returns:
      pts_norm: (N, 2) normalized points
      T: (3, 3) normalization transform such that x_norm ~ T * x
    """
    mean = np.mean(pts, axis=0)
    pts_centered = pts - mean

    dists = np.sqrt(np.sum(pts_centered**2, axis=1))
    mean_dist = np.mean(dists)
    scale = np.sqrt(2) / mean_dist

    T = np.array([
        [scale,     0, -scale * mean[0]],
        [    0, scale, -scale * mean[1]],
        [    0,     0,               1]
    ])

    # homogenous
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])  # (N, 3)
    pts_norm_h = (T @ pts_h.T).T
    pts_norm = pts_norm_h[:, :2] / pts_norm_h[:, 2:3]

    return pts_norm, T


# =========================
# 2. Normalized 8-point F
# =========================

def compute_fundamental_normalized(pts1, pts2):
    """
    Compute fundamental matrix F from corresponding points
    using the normalized 8-point algorithm.

    pts1, pts2: (N, 2) arrays, N >= 8
    """
    assert pts1.shape == pts2.shape
    N = pts1.shape[0]
    assert N >= 8, "Need at least 8 point correspondences"

    # Normalize each set of points
    pts1_norm, T1 = normalize_points(pts1)
    pts2_norm, T2 = normalize_points(pts2)

    x1 = pts1_norm[:, 0]
    y1 = pts1_norm[:, 1]
    x2 = pts2_norm[:, 0]
    y2 = pts2_norm[:, 1]

    # Build design matrix A (N x 9)
    A = np.column_stack([
        x2 * x1,  # x' x
        x2 * y1,  # x' y
        x2,       # x'
        y2 * x1,  # y' x
        y2 * y1,  # y' y
        y2,       # y'
        x1,       # x
        y1,       # y
        np.ones(N)
    ])

    # Solve Af = 0 using SVD
    U, S, Vt = np.linalg.svd(A)
    f = Vt[-1]     # last row of V^T (smallest singular value)
    F_norm = f.reshape(3, 3)

    # Enforce rank-2 on F
    U_F, S_F, Vt_F = np.linalg.svd(F_norm)
    S_F[2] = 0     # force smallest singular value to 0
    F_norm_rank2 = U_F @ np.diag(S_F) @ Vt_F

    # Denormalize: F = T2^T * F_norm * T1
    F = T2.T @ F_norm_rank2 @ T1

    # Optional: normalize so that ||F|| = 1
    F = F / np.linalg.norm(F)

    return F


# =========================
# 3. Estimate K from F
# =========================

def estimate_focal_from_F(F, img_shape,
                          f_min=200, f_max=4000, n_steps=500):
    """
    Estimate focal length f assuming:
      K = [[f, 0, cx],
           [0, f, cy],
           [0, 0,  1]]

      and K' = K.

    Strategy:
      - For each candidate f in [f_min, f_max], build K(f)
      - Compute E(f) = K(f)^T F K(f)
      - Do SVD on E(f) -> singular values s1 >= s2 >= s3
      - Cost(f) = (s1 - s2)^2 + s3^2
      - Pick f that minimizes cost(f)

    Returns:
      K_est: (3, 3) estimated intrinsic matrix
      best_f: scalar focal length
      best_cost: value of the cost function
    """
    h, w = img_shape[:2]
    cx, cy = w / 2.0, h / 2.0

    f_candidates = np.linspace(f_min, f_max, n_steps)
    best_f = None
    best_cost = np.inf

    for f in f_candidates:
        K = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1]
        ])

        # E = K'^T F K, assume K' = K
        E = K.T @ F @ K

        U, S, Vt = np.linalg.svd(E)
        # S is [s1, s2, s3]
        cost = (S[0] - S[1])**2 + (S[2])**2

        if cost < best_cost:
            best_cost = cost
            best_f = f

    # Build K with best_f
    K_est = np.array([
        [best_f, 0, cx],
        [0, best_f, cy],
        [0,      0,  1]
    ])

    return K_est, best_f, best_cost


# =========================
# 4. Main script
# =========================

def main():
    # ---- USER SETTINGS ----
    left_path = "left.png"
    points_file = "stereo_points.npz"

    # ---- Load image (for size) ----
    left = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
    if left is None:
        raise FileNotFoundError(f"Could not load left image from '{left_path}'")

    h, w = left.shape[:2]
    print(f"Left image size: width={w}, height={h}")

    # ---- Load correspondences ----
    data = np.load(points_file)
    pts_left = data["pts_left"]   # (N, 2)
    pts_right = data["pts_right"] # (N, 2)

    if pts_left.shape != pts_right.shape:
        raise ValueError("Left and right point arrays have different shapes.")

    N = pts_left.shape[0]
    print(f"Loaded {N} correspondences from '{points_file}'")

    # ---- Estimate Fundamental Matrix F ----
    print("\nEstimating Fundamental Matrix F using normalized 8-point algorithm...")
    F_est = compute_fundamental_normalized(pts_left, pts_right)
    print("Estimated F:")
    print(F_est)

    # ---- Estimate Calibration Matrix K ----
    print("\nEstimating calibration matrix K from F (searching over focal length)...")
    K_est, f_best, cost = estimate_focal_from_F(F_est, left.shape,
                                                f_min=200, f_max=4000, n_steps=500)

    print("\nEstimated focal length (pixels):", f_best)
    print("Estimated K:")
    print(K_est)
    print(f"Best cost value: {cost:.6e}")

    # ---- Optionally save results ----
    np.save("F_est.npy", F_est)
    np.save("K_est.npy", K_est)
    print("\nSaved F_est.npy and K_est.npy.")

if __name__ == "__main__":
    main()
