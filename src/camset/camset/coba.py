import cv2

cap = cv2.VideoCapture("/dev/video0")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    print(frame.shape)

    # cv2.imshow("Stereo", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()