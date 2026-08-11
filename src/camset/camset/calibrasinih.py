import cv2
import numpy as np
import glob
import os

# --- Konfigurasi ---
CHESSBOARD_SIZE = (7, 5)   # inner corners
SQUARE_SIZE = 0.03         # meter
FRAME_W, FRAME_H = 640, 240
HALF_W = FRAME_W // 2      # 320

# --- Siapkan object points (0,0,0), (1,0,0), ... dikali ukuran kotak ---
objp = np.zeros((CHESSBOARD_SIZE[0]*CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []
imgpoints_left = []
imgpoints_right = []

cap = cv2.VideoCapture('/dev/video2')
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

print("Tekan SPASI untuk capture frame kalau papan catur terdeteksi di kedua sisi.")
print("Tekan 'c' untuk mulai proses kalibrasi. Tekan 'q' untuk keluar tanpa kalibrasi.")

captured = 0
img_shape = None

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    left = frame[:, :HALF_W]
    right = frame[:, HALF_W:]

    gray_l = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    img_shape = gray_l.shape[::-1]

    ret_l, corners_l = cv2.findChessboardCorners(gray_l, CHESSBOARD_SIZE, None)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, CHESSBOARD_SIZE, None)

    disp_l = left.copy()
    disp_r = right.copy()
    if ret_l:
        cv2.drawChessboardCorners(disp_l, CHESSBOARD_SIZE, corners_l, ret_l)
    if ret_r:
        cv2.drawChessboardCorners(disp_r, CHESSBOARD_SIZE, corners_r, ret_r)

    combined = np.hstack((disp_l, disp_r))
    cv2.putText(combined, f'Captured: {captured}', (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imshow('Kalibrasi Stereo - kiri | kanan', combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' ') and ret_l and ret_r:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
        corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints_left.append(corners_l)
        imgpoints_right.append(corners_r)
        captured += 1
        print(f"Captured {captured} pasang gambar")
    elif key == ord('c'):
        break
    elif key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

cap.release()
cv2.destroyAllWindows()

if captured < 10:
    print(f"Cuma {captured} capture, disarankan minimal 15-20 buat hasil bagus.")

print("Menghitung kalibrasi individual kamera kiri...")
ret_l, mtx_l, dist_l, _, _ = cv2.calibrateCamera(objpoints, imgpoints_left, img_shape, None, None)
print("Menghitung kalibrasi individual kamera kanan...")
ret_r, mtx_r, dist_r, _, _ = cv2.calibrateCamera(objpoints, imgpoints_right, img_shape, None, None)

print("Menghitung kalibrasi stereo (ekstrinsik)...")
flags = cv2.CALIB_FIX_INTRINSIC
criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

ret, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right,
    mtx_l, dist_l, mtx_r, dist_r, img_shape,
    criteria=criteria_stereo, flags=flags
)

print("RMS error stereo calibration:", ret)
print("Baseline (T):", T.ravel(), "meter")

# Hitung rectification map
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    mtx_l, dist_l, mtx_r, dist_r, img_shape, R, T, alpha=0
)

np.savez('stereo_calib.npz',
         mtx_l=mtx_l, dist_l=dist_l,
         mtx_r=mtx_r, dist_r=dist_r,
         R=R, T=T, E=E, F=F,
         R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
         img_shape=img_shape)

print("Kalibrasi selesai, disimpan ke stereo_calib.npz")