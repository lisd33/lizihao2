from ultralytics import YOLO
import os
import cv2
import random

CLS_ID_NAME_MAP = {
    0: 'student_id',
    1: 'subjective_problem',
    2: 'fillin_problem',
    3: 'objective_problem'
}

model = YOLO(model='./runs/detect/train10/weights/best.pt')
folder = './dataset4'
file_names = os.listdir(folder)

random.shuffle(file_names)

imgs = []
for file_name in file_names[:100]:
    img_path = os.path.join(folder, file_name)
    img = cv2.imread(img_path)
    imgs += [img]

# Run predictions with label and confidence display turned off
results = model.predict(source=imgs, save=True, imgsz=640, show_labels=True, show_conf=True)

# If `show_labels` and `show_conf` are not supported, use the following method:

for idx, result in enumerate(results):
    img = imgs[idx]
    for box in result.boxes:
        # Get bounding box coordinates
        x, y, w, h = box.xywh.cpu().numpy()[0]
        
        # Draw bounding box only
        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Save image without labels and confidence scores
    save_path = f"./results_no_labels/img_{idx}.png"
    cv2.imwrite(save_path, img)
