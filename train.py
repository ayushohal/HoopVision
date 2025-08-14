from ultralytics import YOLO

model = YOLO("yolov8s.pt")


model.train(data=r"D:\Projects\Basket-ball Analytics\Dataset\data.yaml", epochs=50, imgsz=640)
