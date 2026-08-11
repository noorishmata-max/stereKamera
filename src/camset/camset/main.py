import cv2
import numpy as np
from ultralytics import YOLO

FRAME_W, FRAME_H = 640, 240
HALF_W = FRAME_W // 2

# --- Load hasil kalibrasi ---
calib = np.load('stereo_calib.npz')
mtx_l, dist_l = calib['mtx_l'], calib['dist_l']
mtx_r, dist_r = calib['mtx_r'], calib['dist_r']
R1, R2, P1, P2, Q = calib['R1'], calib['R2'], calib['P1'], calib['P2'], calib['Q']
img_shape = tuple(calib['img_shape'])

map1_l, map2_l = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, img_shape, cv2.CV_16SC2)
map1_r, map2_r = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, img_shape, cv2.CV_16SC2)

# --- Stereo matcher ---
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=64,       # kelipatan 16
    blockSize=7,
    P1=8 * 3 * 7 ** 2,
    P2=32 * 3 * 7 ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)

# --- YOLO ---
model = YOLO('src/camset/camset/best_openvino_model')

cap = cv2.VideoCapture('/dev/video2')
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    left = frame[:, :HALF_W]
    right = frame[:, HALF_W:]

    # Rectify
    rect_l = cv2.remap(left, map1_l, map2_l, cv2.INTER_LINEAR)
    rect_r = cv2.remap(right, map1_r, map2_r, cv2.INTER_LINEAR)

    gray_l = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)

    # Disparity (dibagi 16 karena StereoSGBM output fixed-point)
    disparity = stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0

    # Deteksi YOLO di frame kiri yang sudah rectified
    results = model(rect_l, verbose=False)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # ambil median disparity di area bbox biar lebih stabil
        roi = disparity[max(0,y1):y2, max(0,x1):x2]
        valid = roi[roi > 0]

        cls_name = model.names[int(box.cls[0])]
        conf = float(box.conf[0])

        label = cls_name
        if len(valid) > 0:
            d = np.median(valid)
            # depth = f * baseline / disparity, pakai Q matrix biar akurat (termasuk skala)
            point = np.array([[[cx, cy, d]]], dtype=np.float32)
            point_3d = cv2.perspectiveTransform(point, Q)
            depth = point_3d[0][0][2]  # Z dalam satuan sama dengan baseline (meter)
            label = f'{cls_name} {depth:.2f}m'

        cv2.rectangle(rect_l, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(rect_l, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

    cv2.imshow('YOLO + Depth', rect_l)
    cv2.imshow('Disparity', disp_vis)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()