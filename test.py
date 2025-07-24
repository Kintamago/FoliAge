from ultralytics import YOLO
import cv2

# 🔁 Load your trained model
model = YOLO("runs/detect/train6/weights/best.pt")  # or wherever your model is saved

# 🎥 Open MacBook's webcam
cap = cv2.VideoCapture(0)  # use 1 or 2 if you have external cameras

# 🖼️ Loop over frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 🔍 Run inference
    results = model(frame)

    # 🖌️ Visualize results
    annotated_frame = results[0].plot()

    # 💬 Display
    cv2.imshow("YOLOv8 - Custom Model Detection", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
