import math
import cv2

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize YuNet detector
detector = cv2.FaceDetectorYN.create(
    model="face_detection_yunet_2023mar.onnx",
    config="",
    input_size=(640, 480),
    score_threshold=0.85,
    nms_threshold=0.3,
    top_k=5000
)

while True:
    # Read webcam frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to access webcam")
        break

    # Get current frame dimensions
    h, w, _ = frame.shape
    detector.setInputSize((w, h))

    # Detect faces
    _, faces = detector.detect(frame)

    # Draw bounding boxes
    if faces is not None:
        for face in faces:
            x, y, box_w, box_h = map(int, face[:4])
            center = (round(box_w/2) + x, round(box_h/2) + y)
            print(center)
            cv2.rectangle(frame,(x, y), (x + box_w, y + box_h), (0, 255, 0), 2)
            cv2.circle(frame, center, 3, (0,0,255), -1)

    # Display webcam feed
    cv2.namedWindow("Live Face Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Live Face Detection", 1920, 1080)
    cv2.imshow("Live Face Detection", frame)


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
