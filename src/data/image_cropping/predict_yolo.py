from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("../../../runs/detect/yolov8_detection_model8/weights/best.pt")

# Define path to the image file
source = "D:/Martin/thesis/data/processed/dataset_0328_final/test/imgs/"

model.predict(source, save=True, imgsz=512, conf=0.5)
