from ultralytics import YOLO
m = YOLO('yolov8n.pt')
m.train(data='data/yolo/data.yaml', imgsz=640, epochs=40, batch=16, name='yolo_algae')
# ultralytics prints its save_dir. Copy weights/best.pt from there to data/models/yolo_best.pt
