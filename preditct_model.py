from ultralyticsplus import YOLO, render_result
import cv2
import numpy as np
# load model
model = YOLO("best.pt")

# set model parameters
model.overrides['conf'] = 0.25  # NMS confidence threshold
model.overrides['iou'] = 0.45  # NMS IoU threshold
model.overrides['agnostic_nms'] = False  # NMS class-agnostic
model.overrides['max_det'] = 1000  # maximum number of detections per image

# inference
image_path = "static/img/img_normal.jpg"
input_video_path = "static/videos/video_now.mp4"
output_video_path = "static/videos/video_result.mp4"

def detection_yolo():
    results = model.predict(image_path)
    # observe results
    print(results[0].boxes)
    render = render_result(model=model, image=image_path, result=results[0])
    render.save("static/img/img_now.jpg")

def detection_yolo_video():
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
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



