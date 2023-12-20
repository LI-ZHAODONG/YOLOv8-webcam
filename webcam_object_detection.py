import cv2

from yolov8 import YOLOv8

# Initialize the webcam
cap = cv2.VideoCapture(0)

model_path = "D:/My Programming shit/tinygrad/yolov8n.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    boxes, scores, class_ids = yolov8_detector(frame)

    combined_img = yolov8_detector.draw_detections(frame)
    cv2.imshow("Detected Objects", combined_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
