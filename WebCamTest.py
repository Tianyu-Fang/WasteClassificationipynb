import cv2

cap = cv2.VideoCapture(0)  # 0 is the default camera
if not cap.isOpened():
    print("Webcam not found or could not be opened.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    cv2.imshow("Webcam Test", frame)

    # Press 'q' to exit the webcam view
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
