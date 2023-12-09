import numpy as np
from PIL import Image
import preditct_model
import os
from flask import Flask, render_template, request, make_response, send_file
from datetime import datetime
from functools import wraps, update_wrapper
from shutil import copyfile
import random
from werkzeug.utils import secure_filename
import time
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))


def nocache(view):
    @wraps(view)
    def no_cache(*args, **kwargs):
        response = make_response(view(*args, **kwargs))
        response.headers['Last-Modified'] = datetime.now()
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
        return response
    return update_wrapper(no_cache, view)


@app.route("/index")
@app.route("/")
@nocache
def index():
    return render_template("home.html", file_path="img/image_here.jpg")


@app.route("/about")
@nocache
def about():
    return render_template('about.html')


@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


@app.route("/detection-yolo", methods=["POST"])
@nocache
def detection_yolo():
    preditct_model.detection_yolo()
    return render_template("uploaded.html", file_path="img/img_now.jpg")

ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mkv', 'mov'}

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


def run_detection():
    try:
        with app.app_context():
            preditct_model.detection_yolo_video()
    except Exception as e:
        print(f"Error in run_detection: {e}")
    
@app.route("/upload", methods=["POST"])
@nocache
def upload():
    target = os.path.join(APP_ROOT, "static")

    if not os.path.exists(os.path.join(target, "img")):
        os.makedirs(os.path.join(target, "img"))

    if not os.path.exists(os.path.join(target, "videos")):
        os.makedirs(os.path.join(target, "videos"))

    for file in request.files.getlist("file"):
        if file:
            filename = secure_filename(file.filename)
            if allowed_file(filename, ALLOWED_VIDEO_EXTENSIONS):
                file_path = os.path.join(target, "videos", "video_now.mp4")
                file.save(file_path)

                run_detection()
                
                time.sleep(2)

                return render_template("uploaded_video.html", file_path=file_path)
            elif allowed_file(filename, ALLOWED_IMAGE_EXTENSIONS):
                file_path = os.path.join(target, "img", "img_now.jpg")
                file.save(file_path)

                # Copy image for rendering in uploaded.html
                copyfile(file_path, os.path.join(target, "img", "img_normal.jpg"))
                return render_template("uploaded.html", file_path=file_path)

    return "Invalid file format. Allowed formats: mp4, png, jpg, jpeg, gif"

@app.route('/static/videos/video_result.mp4')
def get_video_result():
    video_result_path = os.path.join(APP_ROOT, 'static', 'videos', 'video_result.mp4')
    response = send_file(video_result_path, mimetype='video/mp4')
    response.headers.add('Accept-Ranges', 'bytes')
    response.headers.add('Content-Length', os.path.getsize(video_result_path))
    return response


@app.route("/detection-detr", methods=["POST"])
@nocache
def detection_detr():
    preditct_model.detection_detr()
    return render_template("uploaded.html", file_path="img/img_now.jpg")

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")

