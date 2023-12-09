
import os
import torch
import torchvision
from transformers import DetrForObjectDetection, DetrImageProcessor
import cv2
import supervision as sv

# Your code here
from ultralyticsplus import YOLO, render_result


def detection_yolo():
    # load model
    model = YOLO("best.pt")

    # set model parameters
    model.overrides['conf'] = 0.25  # NMS confidence threshold
    model.overrides['iou'] = 0.45  # NMS IoU threshold
    model.overrides['agnostic_nms'] = False  # NMS class-agnostic
    model.overrides['max_det'] = 1000  # maximum number of detections per image

    # inference
    image_path = "static/img/img_normal.jpg"
    results = model.predict(image_path)
    # observe results
    print(results[0].boxes)
    render = render_result(model=model, image=image_path, result=results[0])
    render.save("static/img/img_now.jpg")


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
    
    # Create a dictionary to store the counts
    terdeteksi = {}

    # Iterate through labels and count occurrences
    for label in labels:
        class_name = label.split()[0]  # Extract the class name (e.g., 'WBC' or 'RBC')
        if class_name in terdeteksi:
            terdeteksi[class_name] += 1
        else:
            terdeteksi[class_name] = 1

    print(terdeteksi)
