import torch
import cv2
import numpy as np
import os

from ultralytics import YOLO
#image = cv2.imread("c:\1\x.jpg")

model1 = YOLO('yolo11x.pt')
model = YOLO(r'c:\1\best_bar8000_70.pt')

colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
    (255, 0, 255), (192, 192, 192), (128, 128, 128), (128, 0, 0), (128, 128, 0),
    (0, 128, 0), (128, 0, 128), (0, 128, 128), (0, 0, 128), (72, 61, 139),
    (47, 79, 79), (47, 79, 47), (0, 206, 209), (148, 0, 211), (255, 20, 147)
]

# Функция для обработки изображения
def process_image(image_path):
    # Загрузка изображения
    image = cv2.imread(image_path)
    results = model(image,conf=0.2)[0]
    print(results)
    # Получение оригинального изображения и результатов
    image = results.orig_img
    classes_names = results.names
    sss = results.boxes.conf
    classes = results.boxes.cls.cpu().numpy()
    boxes = results.boxes.xyxy.cpu().numpy().astype(np.int32)
    casc=0
    # Подготовка словаря для группировки результатов по классам
    grouped_objects = {}

    # Рисование рамок и группировка результатов
    for class_id, box, sss1 in zip(classes, boxes, sss):
        class_name = classes_names[int(class_id)]
        color = colors[int(class_id) % len(colors)]  # Выбор цвета для класса
        if class_name not in grouped_objects:
            grouped_objects[class_name] = []
        grouped_objects[class_name].append(box)
        if class_name == "Helmet":
            casc=casc+1
            print(casc)
            # Рисование рамок на изображении
            x1, y1, x2, y2 = box
        #  if sss1<0.4:
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, str(class_name)+"_"+str(sss1), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Сохранение измененного изображения
    new_image_path = os.path.splitext(image_path)[0] + '_yolo' + os.path.splitext(image_path)[1]
    cv2.imwrite(new_image_path, image)

    # Сохранение данных в текстовый файл
    text_file_path = os.path.splitext(image_path)[0] + '_data.txt'
    with open(text_file_path, 'w') as f:
        for class_name, details in grouped_objects.items():
            f.write(f"{class_name}:\n")
            for detail in details:
                f.write(f"Coordinates: ({detail[0]}, {detail[1]}, {detail[2]}, {detail[3]})\n")

  #  print(f"Processed {image_path}:")
 #   print(f"Saved bounding-box image to {new_image_path}")
 #   print(f"Saved data to {text_file_path}")
    return casc
def process_image2(image_path):
    # Загрузка изображения
    image = cv2.imread(image_path)
    results = model1(image,conf=0.3)[0]
    print(results)
    # Получение оригинального изображения и результатов
    image = results.orig_img
    classes_names = results.names
    sss = results.boxes.conf
    classes = results.boxes.cls.cpu().numpy()
    boxes = results.boxes.xyxy.cpu().numpy().astype(np.int32)
    casc=0
    # Подготовка словаря для группировки результатов по классам
    grouped_objects = {}

    # Рисование рамок и группировка результатов
    for class_id, box, sss1 in zip(classes, boxes, sss):
        class_name = classes_names[int(class_id)]
        color = colors[int(class_id) % len(colors)]  # Выбор цвета для класса
        if class_name not in grouped_objects:
            grouped_objects[class_name] = []
        grouped_objects[class_name].append(box)
        if class_name == "person":
            casc=casc+1
        # Рисование рамок на изображении
        x1, y1, x2, y2 = box
      #  if sss1<0.4:
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, str(class_name)+"_"+str(sss1), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Сохранение измененного изображения
    new_image_path = os.path.splitext(image_path)[0] + '_yoloX' + os.path.splitext(image_path)[1]
    cv2.imwrite(new_image_path, image)

    # Сохранение данных в текстовый файл
    text_file_path = os.path.splitext(image_path)[0] + '_dataX.txt'
    with open(text_file_path, 'w') as f:
        for class_name, details in grouped_objects.items():
            f.write(f"{class_name}:\n")
            for detail in details:
                f.write(f"Coordinates: ({detail[0]}, {detail[1]}, {detail[2]}, {detail[3]})\n")

  #  print(f"Processed {image_path}:")
 #   print(f"Saved bounding-box image to {new_image_path}")
 #   print(f"Saved data to {text_file_path}")
    return casc

print(process_image(r'c:\1\x.jpg'))
print(process_image2(r'c:\1\x.jpg'))

torch.cuda.memory_summary()