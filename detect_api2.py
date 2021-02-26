import argparse

import torch.backends.cudnn as cudnn

from flask import Flask, request, Response, jsonify, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
from models.experimental import *
from utils.datasets import *
from utils import *
import sys
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized
import argparse
import os, logging
import platform
import shutil
import time, io
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized
import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
app = Flask(__name__)

weights = 'yolov5s.pt' if len(sys.argv) == 1 else sys.argv[1]
device_number = '' if len(sys.argv) <= 2 else sys.argv[2]
device = torch_utils.select_device(device_number)

model = attempt_load(weights, map_location=device)  # load FP32 model
out = "inference/output"
crop = "inference/crop"
global text


def detect(saved_path):
    save_img = False
    global imageafterpred, bound
    # form_data = request.json
    # print(form_data)
    bound = []
    source = saved_path
    out = "inference/output"
    imgsz = 640
    conf_thres = 0.5
    iou_thres = 0.5
    view_img = False
    save_txt = False
    classes = None
    agnostic_nms = False
    augment = False
    update = False

    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http')

    # Initialize
    # device = torch_utils.select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    # model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        t2 = torch_utils.time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s
                imageafterpred = im0

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].detach().unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    xyx = torch.tensor(xyxy).view(1, 4)
                    xyx = xyx.numpy()
                    # print(xyx)
                    x1, y1, x2, y2 = xyx[0][0], xyx[0][1], xyx[0][2], xyx[0][3]
                    # print(x1,y1,x2,y2)
                    bound.append(x1)
                    bound.append(y1)
                    bound.append(x2)
                    bound.append(y2)
                    # print(bound)
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        print(xywh)
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 6 + '\n') % (cls, *xywh, conf))  # label format

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin' and not update:  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))
    return imageafterpred, bound


app = Flask(__name__, static_url_path="/static")
file_handler = logging.FileHandler('server.log')
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)

#PROJECT_HOME = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = "static/uploads/"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def create_new_folder(local_dir):
    newpath = local_dir
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    return newpath

def process_file(path, filename):
    image = cv2.imread(path)
    res, bound = detect(saved_path)
    x1, y1, x2, y2 = bound[0], bound[1], bound[2], bound[3]
    Cropped = image[int(y2):int(y1), int(x2):int(x1)]
    print(Cropped.shape)
    text = pytesseract.image_to_string(Cropped, config='--psm 8')
    print(text)


@app.route("/home")
def Home_page():
    return render_template('template.html')


# route http posts to this method
@app.route('/api/test', methods=['POST', 'GET'])
def main():
    if request.method == 'GET':
        return render_template('result.html')
    if request.method == 'POST':
        img = request.files["image"]
        # app.logger.info(app.config['UPLOAD_FOLDER'])
        img_name = secure_filename(img.filename)
        create_new_folder(app.config['UPLOAD_FOLDER'])
        saved_path = os.path.join(app.config['UPLOAD_FOLDER'], img_name)
        app.logger.info("saving {}".format(saved_path))
        img.save(saved_path)
        '''process_file(saved_path, img_name)
        data = {“uploaded_img”:"static / uploads /"+img_name}
        print(saved_path)
        in_memory_file = io.BytesIO()
        img.save(in_memory_file)
        data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
        color_image_flag = 1
        img = cv2.imdecode(saved_path, color_image_flag)
        img.save(saved_path)'''
        image = cv2.imread(saved_path)
        '''image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # cv2.imshow("image",image)'''
        res, bound = detect(saved_path)
        x1, y1, x2, y2 = bound[0], bound[1], bound[2], bound[3]
        # print(res) #res = get_predection(image, nets, Lables, Colors)
        # image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        # show the output image
        # cv2.imshow("Image", res)
        # cv2.waitKey()
        # save_path = os.path.join(out, img_name)
        # print(save_path)
        # image = cv2.imread(saved_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imshow("2", image)
        # print(x1,y1,x2,y2)
        Cropped = image[int(y2):int(y1), int(x2):int(x1)]
        print(Cropped.shape)
        # sav_path = os.path.join(crop, img_name)
        # cv2.imwrite(sav_path,Cropped)
        cv2.imshow("1", Cropped)
        text = pytesseract.image_to_string(Cropped, config='--psm 8')
        print(text)
        # print(image) #np_img = Image.fromarray(image)
        # img_encoded = image_to_byte_array(np_img)
        #return render_template(“index.html”, data = data)
        return send_from_directory(out, img_name, as_attachment=False)


if __name__ == '__main__':
    app.run(debug=True)
