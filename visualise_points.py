import numpy as np
import cv2
import matplotlib.pyplot as plt

def main():
    # ---- USER SETTINGS ----
    left_path = "left.png"           # same as in collect_points.py
    right_path = "right.png"
    points_file = "stereo_points.npz"

    # ---- LOAD IMAGES ----
    left = cv2.imread(left_path, cv2.IMREAD_COLOR)
    right = cv2.imread(right_path, cv2.IMREAD_COLOR)

    if left is None:
        raise FileNotFoundError(f"Could not load left image from '{left_path}'")
    if right is None:
        raise FileNotFoundError(f"Could not load right image from '{right_path}'")

    # Convert BGR (OpenCV) -> RGB (matplotlib)
    left_rgb = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
    right_rgb = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)

    # ---- LOAD CORRESPONDENCES ----
    data = np.load(points_file)
    pts_left = data["pts_left"]   # shape (N, 2)
    pts_right = data["pts_right"] # shape (N, 2)

    if pts_left.shape != pts_right.shape:
        raise ValueError("Left and right point arrays have different shapes.")

    n_points = pts_left.shape[0]
    print(f"Loaded {n_points} correspondences.")

    # ---- FIGURE 1: side-by-side images with numbered points ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(left_rgb)
    axes[0].set_title("Left image with points")
    axes[0].axis("off")

    axes[1].imshow(right_rgb)
    axes[1].set_title("Right image with points")
    axes[1].axis("off")

    for i in range(n_points):
        xL, yL = pts_left[i]
        xR, yR = pts_right[i]

        # plot points
        axes[0].scatter(xL, yL, s=30)
        axes[1].scatter(xR, yR, s=30)

        # label them with index
        axes[0].text(xL + 3, yL - 3, str(i), color="yellow", fontsize=9)
        axes[1].text(xR + 3, yR - 3, str(i), color="yellow", fontsize=9)

    plt.tight_layout()
    plt.show()

    # ---- FIGURE 2: concatenated image with lines between matches ----
    h1, w1, _ = left_rgb.shape
    h2, w2, _ = right_rgb.shape
    h = max(h1, h2)
    w = w1 + w2

    # make a canvas big enough for both
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:h1, :w1, :] = left_rgb
    canvas[:h2, w1:w1 + w2, :] = right_rgb

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.imshow(canvas)
    ax2.set_title("Correspondences (lines across stereo pair)")
    ax2.axis("off")

    for i in range(n_points):
        xL, yL = pts_left[i]
        xR, yR = pts_right[i]

        # right image x is shifted by w1 in the concatenated canvas
        xR_shifted = xR + w1

        # draw points
        ax2.scatter([xL, xR_shifted], [yL, yR], s=20)
        # draw line between them
        ax2.plot([xL, xR_shifted], [yL, yR], linewidth=1)

        # optional: label index near the middle of the segment
        xm = 0.5 * (xL + xR_shifted)
        ym = 0.5 * (yL + yR)
        ax2.text(xm, ym, str(i), color="yellow", fontsize=8)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
