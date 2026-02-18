from ultralytics import YOLO
model = YOLO("H:/mobile_used_project/runs/detect/train2/weights/best.pt")

model.export(format="onnx")

