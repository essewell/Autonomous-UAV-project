from ultralytics import YOLO

print("Loading YOLOv8-nano model...")
model = YOLO('yolov8n.pt')

print("Exporting to ONNX format...")
model.export(
    format='onnx',
    imgsz=320,
    simplify=True,
    opset=11,
    dynamic=False
)

print("✓ Export complete! Model saved as yolov8n.onnx")
