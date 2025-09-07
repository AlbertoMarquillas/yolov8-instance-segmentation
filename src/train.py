from ultralytics import YOLO

config = r'D:\Code\Repositorios\Computer_Vision_Projects\20_Image_Segmentation_Yolov8_Custom\config.yaml'

model = YOLO(r'D:\Code\Repositorios\Computer_Vision_Projects\yolov8n-seg.pt')  # load a pretrained model (recommended for training)

results = model.train(data=config, epochs=1, imgsz=640)