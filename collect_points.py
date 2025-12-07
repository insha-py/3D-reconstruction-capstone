import numpy as np
import cv2
import matplotlib.pyplot as plt

def collect_points_on_image(img, n_points, title="Click points"):
    """
    Show an image and let the user click n_points points.
    Returns an array of shape (n_points, 2) with [x, y] pixel coordinates.
    """
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    ax.set_title(f"{title} (click {n_points} points)")
    ax.axis('off')

    # ginput returns a list of (x, y) tuples
    pts = plt.ginput(n_points, timeout=0)
    plt.close(fig)

    pts = np.array(pts, dtype=float)
    return pts

def main():
    # ---- USER SETTINGS ----
    left_path = "left.png"   # change to your left image filename
    right_path = "right.png" # change to your right image filename
    n_points = 12            # how many correspondences you want to pick

    # ---- LOAD IMAGES ----
    left = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)

    if left is None:
        raise FileNotFoundError(f"Could not load left image from '{left_path}'")
    if right is None:
        raise FileNotFoundError(f"Could not load right image from '{right_path}'")

    # Convert from BGR (OpenCV) to RGB for matplotlib display
    # (for grayscale, it's already one channel, so just pass as is)
    # If you later use color images, convert with cv2.cvtColor.

    print(f"Loaded left image: {left.shape[1]} x {left.shape[0]}")
    print(f"Loaded right image: {right.shape[1]} x {right.shape[0]}")

    # ---- COLLECT POINTS ON LEFT IMAGE ----
    print(f"\n[LEFT IMAGE] Click {n_points} points (in order).")
    pts_left = collect_points_on_image(left, n_points, title="Left image")

    # ---- COLLECT POINTS ON RIGHT IMAGE ----
    print(f"\n[RIGHT IMAGE] Now click the corresponding {n_points} points in the SAME ORDER.")
    pts_right = collect_points_on_image(right, n_points, title="Right image")

    # ---- CHECK AND SAVE ----
    if pts_left.shape != pts_right.shape:
        raise ValueError("Number of points in left and right images do not match.")

    out_file = "stereo_points.npz"
    np.savez(out_file, pts_left=pts_left, pts_right=pts_right)

    print(f"\nSaved {n_points} correspondences to '{out_file}'.")
    print("Arrays inside file: 'pts_left' (N x 2), 'pts_right' (N x 2).")

if __name__ == "__main__":
    main()
