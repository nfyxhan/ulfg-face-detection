# -*- coding: utf-8 -*-
import argparse
import os
import sys

import cv2
import time
from PIL import Image, ImageDraw, ImageFont
import numpy as np


from vision.ssd.config.fd_config import define_img_size


import tornado.web
import tornado.httpserver
from tornado.options import define, options
import tornado.ioloop
import json
import base64
from io import BytesIO

_BASEDIR = os.path.dirname(os.path.abspath(__file__))

import logging
from logging.handlers import RotatingFileHandler
import datetime
logger = logging.getLogger("tr" + '.' + __name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - [%(levelname)s] | %(message)s', datefmt='%Y/%m/%d %H:%M:%S')

logfile_name = datetime.date.today().__format__('%Y-%m-%d.log')
logfile_path = os.path.join(_BASEDIR, f'logs/')
if not os.path.exists(logfile_path):
        os.mkdir(logfile_path)
        handler_logfile = RotatingFileHandler(logfile_path + logfile_name, maxBytes=1 * 1024 * 1024, backupCount=3, encoding="utf-8")
        handler_logfile.setLevel(logging.INFO)
        handler_logfile.setFormatter(formatter)
        logger.addHandler(handler_logfile)

        #console_output = logging.StreamHandler()
        #console_output.setLevel(logging.INFO)
        #console_output.setFormatter(formatter)
        #logger.addHandler(console_output)
settings = dict()
args = dict()
predictor = None

def init(args):
    define_img_size(args.input_size)  # must put define_img_size() before 'import create_mb_tiny_fd, create_mb_tiny_fd_predictor'
    from vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
    from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor
    label_path = "./models/voc-model-labels.txt"
    test_device = args.test_device
    class_names = [name.strip() for name in open(label_path).readlines()]
    global predictor
    if args.net_type == 'slim':
        model_path = "models/pretrained/version-slim-320.pth"
        # model_path = "models/pretrained/version-slim-640.pth"
        net = create_mb_tiny_fd(len(class_names), is_test=True, device=test_device)
        predictor = create_mb_tiny_fd_predictor(net, candidate_size=args.candidate_size, device=test_device)
    elif args.net_type == 'RFB':
        model_path = "models/pretrained/version-RFB-320.pth"
        # model_path = "models/pretrained/version-RFB-640.pth"
        net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device=test_device)
        predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=args.candidate_size, device=test_device)
    else:
        print("The net type is wrong!")
        sys.exit(1)
    net.load(model_path)
    print(predictor)
    return predictor

class FaceDetector(tornado.web.RequestHandler):
    def get(self):
        self.set_status(405)
        self.write("405 : Please use POST")

    @tornado.gen.coroutine
    def post(self):
        t = time.time()
        img_up = self.request.files.get('file', None)
        r = json.loads(self.request.body)
        img_b64 = r.get('img', None)
        draw = r.get('draw', False)
        img_pil = None
        if img_up is not None and len(img_up) > 0:
            img_up = img_up[0]
            up_image_type = img_up.content_type
            up_image_name = img_up.filename
            img_pil = Image.open(BytesIO(img_up.body))
        elif img_b64 is not None:
            raw_image = base64.b64decode(img_b64.encode('utf8'))
            img_pil = Image.open(BytesIO(raw_image))
        if img_pil is None:
            logger.error(json.dumps({'code': 400, 'msg': u'没有传入参数'}, ensure_ascii=False))
            self.finish(json.dumps("img is none", cls=NpEncoder))
            return 
        logger.info(img_pil.size)
        origin_size = img_pil.size 
        face_data = run(img_pil)
        print(face_data)
        result =  {
               'code': 200, 
               'data': face_data,
               "cost": time.time()-t,
               "origin_size": origin_size,
               "draw_img": draw,
        }
        logger.info(json.dumps(result,cls=NpEncoder, ensure_ascii=False))
        if draw:
            img_draw = compare(img_pil, face_data)
            img_byte = BytesIO()
            img_draw.save(img_byte, format="png")
            img_b64 = base64.b64encode(img_byte.getvalue()).decode()
            result['draw_img'] = img_b64
        self.finish(json.dumps(result,cls=NpEncoder, ensure_ascii=False))
        return

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)
    
def run(img_pil):
    orig_image = cv2.cvtColor(np.asarray(img_pil),cv2.COLOR_RGB2BGR)
    #orig_image = cv2.imread(img_path)
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    boxes, labels, probs = predictor.predict(image, args.candidate_size / 2, args.threshold)
    print(f"Found {len(probs)} faces.")
    result = []
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        w = float(box[2] - box[0] )
        h = float(box[3] - box[1])
        cx = float(box[0] + w/2 )
        cy = float(box[1] + h/2)
        result.append({
            "box":[cx,cy,w,h],
            "labels": float(labels[i]),
            "probs": float(probs[i]),
            })
    return result

def compare(img_pil, results):
    color_pil = img_pil.copy()
    img_draw = ImageDraw.Draw(color_pil)
    colors = ['red', 'green', 'blue', "purple"]
    for i, rect in enumerate(results):
        cx, cy, w, h = tuple(rect['box'])
        a = 0
        box = cv2.boxPoints(((cx, cy), (w, h), a))
        box = np.int0(np.round(box))
        for p1, p2 in [(0, 1), (1, 2), (2, 3), (3, 0)]:
            img_draw.line(xy=(box[p1][0], box[p1][1], box[p2][0], box[p2][1]), fill=colors[i % len(colors)], width=2)
    return color_pil
 
def make_app(port):
    app = tornado.web.Application([
        (r"/api/face-detection", FaceDetector),
    ], **settings)
    server = tornado.httpserver.HTTPServer(app)
    server.bind(port)
    server.start(1)
    print(f'Server is running: http://0.0.0.0:{port}')
    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='detect_imgs')
    
    parser.add_argument('--net_type', default="RFB", type=str,
                        help='The network architecture ,optional: RFB (higher precision) or slim (faster)')
    parser.add_argument('--input_size', default=640, type=int,
                        help='define network input size,default optional value 128/160/320/480/640/1280')
    parser.add_argument('--threshold', default=0.6, type=float,
                        help='score threshold')
    parser.add_argument('--candidate_size', default=1500, type=int,
                        help='nms candidate size')
    parser.add_argument('--path', default="imgs", type=str,
                        help='imgs dir')
    parser.add_argument('--test_device', default="cpu", type=str,
                        help='cuda:0 or cpu')
    parser.add_argument('--port', default=8090, type=int,
                        help='int')
    args = parser.parse_args()

    init(args)

#    tornado.options.parse_command_line()
    port = args.port
    make_app(port)
