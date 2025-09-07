from ultralytics import YOLO

import cv2


model_path = r'D:\Code\Repositorios\Computer_Vision_Projects\runs\segment\train4\weights\last.pt'

image_path = r'D:\Code\Repositorios\Computer_Vision_Projects\20_Image_Segmentation_Yolov8_Custom\data\images\val\bf804e584764cd64.jpg'

img = cv2.imread(image_path)
H, W, _ = img.shape

model = YOLO(model_path)

results = model(img)

for result in results:
    for j, mask in enumerate(result.masks.data):

        mask = mask.numpy() * 255

        mask = cv2.resize(mask, (W, H))

        cv2.imwrite('./output.png', mask)

