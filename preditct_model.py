from ultralyticsplus import YOLO, render_result
# load model
model = YOLO("best.pt")

# set model parameters
model.overrides['conf'] = 0.25  # NMS confidence threshold
model.overrides['iou'] = 0.45  # NMS IoU threshold
model.overrides['agnostic_nms'] = False  # NMS class-agnostic
model.overrides['max_det'] = 1000  # maximum number of detections per image

# inference
image_path = "static/img/img_normal.jpg"

def detection_yolo():
    results = model.predict(image_path)
    # observe results
    print(results[0].boxes)
    render = render_result(model=model, image=image_path, result=results[0])
    render.save("static/img/img_now.jpg")
