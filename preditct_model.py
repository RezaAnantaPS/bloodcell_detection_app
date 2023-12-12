
import os
import torch
import torchvision
from transformers import DetrForObjectDetection, DetrImageProcessor
import cv2
import supervision as sv

# Your code here
from ultralyticsplus import YOLO, render_result
import cv2
import numpy as np
import random
import math
from collections import Counter

# inference
image_path = "static/img/img_normal.jpg"
input_video_path = "static/videos/video_now.mp4"
output_video_path = "static/videos/video_result.mp4"

classNames = ['Platelets', 'RBC', "WBC", "sickle cell"]
def detection_yolo():
    title = "YOLOv8"
    # load model
    model = YOLO("best.pt")
    # set model parameters
    model.overrides['conf'] = 0.25  # NMS confidence threshold
    model.overrides['iou'] = 0.45  # NMS IoU threshold
    model.overrides['agnostic_nms'] = False  # NMS class-agnostic
    model.overrides['max_det'] = 1000  # maximum number of detections per image
    
    array = []
    counts = Counter()
    width = int(640)
    height = int(640)
    dim = (width, height)
    img = cv2.imread(image_path)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    results = model(resized, show = False)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            #print(x1, y1, x2, y2)

            color1 = random.randrange(128, 255)
            color2 = random.randrange(128, 255)
            color3 = random.randrange(128, 255)

            # Draw filled rectangle as background for text
            cv2.rectangle(resized, (max(0, x1), max(35, y1) - 25), (x2, y1), (color1, color2, color3), -1)
            
            # Draw bounding box
            cv2.rectangle(resized,(x1,y1),(x2,y2),(color1,color2, color3),3)

            conf = math.ceil((box.conf[0]*100))/100

            cls = int(box.cls[0])

            cv2.putText(img=resized, text=f'{classNames[cls]} {conf}', org=(max(0, x1), max(35, y1) - 5),
                fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.5, color=(0, 0, 0), thickness=2)
            array.append(classNames[cls])
            counts[classNames[cls]] += 1
    print(array)
    
    for label, count in counts.items():
        print(f'{label} = {count}')
    
    cv2.imwrite("static/img/img_now.jpg", resized)
    return counts, title
    

# def detection_yolo():
#     # load model
#     model = YOLO("best.pt")

#     # set model parameters
#     model.overrides['conf'] = 0.25  # NMS confidence threshold
#     model.overrides['iou'] = 0.45  # NMS IoU threshold
#     model.overrides['agnostic_nms'] = False  # NMS class-agnostic
#     model.overrides['max_det'] = 1000  # maximum number of detections per image
#     results = model.predict(image_path)
#     # observe results
#     print("results[0].boxes: ", results[0].boxes)
#     render = render_result(model=model, image=image_path, result=results[0])
#     render.save("static/img/img_now.jpg")



def detection_yolo_video():
    # load model
    model = YOLO("best.pt")

    # set model parameters
    model.overrides['conf'] = 0.25  # NMS confidence threshold
    model.overrides['iou'] = 0.45  # NMS IoU threshold
    model.overrides['agnostic_nms'] = False  # NMS class-agnostic
    model.overrides['max_det'] = 1000  # maximum number of detections per image
    
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to a NumPy array
        frame_np = np.array(frame)

        # Make a copy of the frame to avoid modifying the original
        frame_copy = frame_np.copy()

        results = model.predict(frame_copy)
        render = render_result(model=model, image=frame_copy, result=results[0])
        
        # Convert render back to BGR format before writing
        render_bgr = cv2.cvtColor(np.array(render), cv2.COLOR_RGB2BGR)
        out.write(render_bgr)

    cap.release()
    out.release()

def detection_detr():
    # Check if a GPU is available, and use it if available; otherwise, use CPU.
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_path = "detr-bloodcell-model"
    image_processor = DetrImageProcessor.from_pretrained(model_path)
    detr = DetrForObjectDetection.from_pretrained(model_path)

    # Set model ke mode evaluasi
    detr.to(DEVICE)

    # settings
    ANNOTATION_FILE_NAME = "_annotations.coco.json"
    TRAIN_DIRECTORY = "./dataset/anemia-detection-3/train"
    VAL_DIRECTORY = "./dataset/anemia-detection-3/valid"
    TEST_DIRECTORY = "./dataset/anemia-detection-3/test"


    class CocoDetection(torchvision.datasets.CocoDetection):
        def __init__(
            self,
            image_directory_path: str,
            image_processor,
            train: bool = True
        ):
            annotation_file_path = os.path.join(image_directory_path, ANNOTATION_FILE_NAME)
            super(CocoDetection, self).__init__(image_directory_path, annotation_file_path)
            self.image_processor = image_processor

        def __getitem__(self, idx):
            images, annotations = super(CocoDetection, self).__getitem__(idx)
            image_id = self.ids[idx]
            annotations = {'image_id': image_id, 'annotations': annotations}
            encoding = self.image_processor(images=images, annotations=annotations, return_tensors="pt")
            pixel_values = encoding["pixel_values"].squeeze()
            target = encoding["labels"][0]

            return pixel_values, target


    TRAIN_DATASET = CocoDetection(
        image_directory_path=TRAIN_DIRECTORY,
        image_processor=image_processor,
        train=True)
    VAL_DATASET = CocoDetection(
        image_directory_path=VAL_DIRECTORY,
        image_processor=image_processor,
        train=False)
    TEST_DATASET = CocoDetection(
        image_directory_path=TEST_DIRECTORY,
        image_processor=image_processor,
        train=False)

    print("Number of training examples:", len(TRAIN_DATASET))
    print("Number of validation examples:", len(VAL_DATASET))
    print("Number of test examples:", len(TEST_DATASET))

    # utils
    categories = TEST_DATASET.coco.cats
    id2label = {k: v['name'] for k,v in categories.items()}
    box_annotator = sv.BoxAnnotator()

    image = cv2.imread("static/img/img_normal.jpg")
    
    # inference
    with torch.no_grad():

        # load image and predict
        inputs = image_processor(images=image, return_tensors='pt').to(DEVICE)
        outputs = detr(**inputs)

        # post-process
        target_sizes = torch.tensor([image.shape[:2]]).to(DEVICE)
        results = image_processor.post_process_object_detection(
            outputs=outputs,
            threshold=0.5,
            target_sizes=target_sizes
        )[0]

    # annotate
    detections = sv.Detections.from_transformers(transformers_results=results).with_nms(threshold=0.5)
    labels = [f"{id2label[class_id]} {confidence:.2f}" for _, confidence, class_id, _ in detections]
    frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

    cv2.imwrite("static/img/img_now.jpg",frame)
    
    title = "DETR"
    # Create a dictionary to store the counts
    counts = {}

    # Iterate through labels and count occurrences
    for label in labels:
        class_name = label.split()[0]  # Extract the class name (e.g., 'WBC' or 'RBC')
        if class_name in counts:
            counts[class_name] += 1
        else:
            counts[class_name] = 1

    print(counts)
    for key, value in counts.items():
        if key == "sickle":
            key = "sickle cell"
        print(f"{key} : {value}")
    return counts, title
