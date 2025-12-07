import numpy as np
import os

def main():
    # -------- 1. Load F --------
    F_file = "F_est.npy"
    if not os.path.exists(F_file):
        raise FileNotFoundError(f"Could not find '{F_file}'. Run the F estimation script first.")
    F = np.load(F_file)
    print("Loaded F_est.npy")
    print("F =")
    print(F)

    # -------- 2. Ground-truth intrinsics for Rig A --------
    # cam0 and cam1 are identical for Rig A
    K_left = np.array([
        [1758.23,    0.0,   872.36],
        [   0.0,  1758.23,  552.32],
        [   0.0,     0.0,     1.0 ]
    ])
    K_right = K_left.copy()

    print("\nGround-truth K (Rig A):")
    print(K_left)

    # -------- 3. Compute raw essential matrix E_raw = K'^T F K --------
    E_raw = K_right.T @ F @ K_left
    print("\nRaw essential matrix E_raw = K_right^T * F * K_left:")
    print(E_raw)

    # Singular values of E_raw
    U, S, Vt = np.linalg.svd(E_raw)
    print("\nSingular values of E_raw:")
    print(S)

    # -------- 4. Enforce essential matrix constraints --------
    # Make first two singular values equal, last one zero
    s = (S[0] + S[1]) / 2.0
    S_corrected = np.diag([s, s, 0.0])
    E_corrected = U @ S_corrected @ Vt

    print("\nCorrected essential matrix E_corrected (idealized):")
    print(E_corrected)

    # Singular values after correction
    U_c, S_c, Vt_c = np.linalg.svd(E_corrected)
    print("\nSingular values of E_corrected:")
    print(S_c)

    # -------- 5. (Optional) Compare with essential matrix from estimated K --------
    K_est_file = "K_est.npy"
    if os.path.exists(K_est_file):
        K_est = np.load(K_est_file)
        print("\nFound K_est.npy, comparing essential matrix using estimated K...")

        E_from_Kest = K_est.T @ F @ K_est
        print("\nE_est = K_est^T * F * K_est:")
        print(E_from_Kest)

        U_e, S_e, Vt_e = np.linalg.svd(E_from_Kest)
        print("\nSingular values of E_est:")
        print(S_e)
    else:
        print("\nK_est.npy not found, skipping comparison with estimated K.")

    # -------- 6. Save E matrices --------
    np.save("E_gt_raw.npy", E_raw)
    np.save("E_gt_corrected.npy", E_corrected)

    print("\nSaved E_gt_raw.npy and E_gt_corrected.npy.")

if __name__ == "__main__":
    main()
