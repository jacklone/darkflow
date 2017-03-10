# -*- coding: utf-8 -*-

"""
Created on 12/19/16 3:33 PM
File name   :   person_find.py
Author      :   jacklone
Email       :   jichenglong1988@gmail.com
Description :

"""

from flask import Flask, request, jsonify, url_for, render_template
from skimage import io, color
import cv2
from model_client import tfnet
from utils.box import BoundBox, prob_compare2, box_intersection
import logging
import os
import random
import math
import numpy as np
import base64
from StringIO import StringIO

TMP_IMG_DIR = "tmp"
StupidCount = {0: 0, 1: 0, 2: 0}

logging.basicConfig(level=logging.WARN,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%Y %b %H:%M:%S')

app = Flask(__name__)


def expit(x):
    return 1. / (1. + np.exp(-x))


class RectCalculator:
    def __init__(self):
        meta = tfnet.framework.meta
        self.threshold = meta['thresh']
        self.C, self.B, self.anchors = meta['classes'], meta['num'], meta['anchors']
        self.H, self.W, _ = meta['out_size']

    def calc_rect(self, net_out):
        net_out = net_out.reshape([self.H, self.W, self.B, -1])

        boxes = list()
        for row in range(self.H):
            for col in range(self.W):
                for b in range(self.B):
                    bx = BoundBox(self.C)
                    bx.x, bx.y, bx.w, bx.h, bx.c = net_out[row, col, b, :5]
                    bx.c = expit(bx.c)
                    bx.x = (col + expit(bx.x)) / self.W
                    bx.y = (row + expit(bx.y)) / self.H
                    bx.w = math.exp(bx.w) * self.anchors[2 * b + 0] / self.W
                    bx.h = math.exp(bx.h) * self.anchors[2 * b + 1] / self.H
                    p = net_out[row, col, b, 5:] * bx.c
                    mi = np.argmax(p)
                    if p[mi] < 1.5: continue
                    bx.ind = mi;
                    bx.pi = p[mi]
                    boxes.append(bx)

        # non max suppress boxes
        boxes = sorted(boxes, cmp=prob_compare2)
        for i in range(len(boxes)):
            boxi = boxes[i]
            if boxi.pi == 0: continue
            for j in range(i + 1, len(boxes)):
                boxj = boxes[j]
                areaj = boxj.w * boxj.h
                if box_intersection(boxi, boxj) / areaj >= .4:
                    boxes[j].pi = 0.

        # h, w, _ = imgcv.shape
        valid_boxes = []
        for b in boxes:
            if b.pi > 0.:
                valid_boxes.append([b.x, b.y, b.w, b.h, b.ind])
        return valid_boxes

BoxCalculator = RectCalculator()


@app.route('/segment', methods=["GET", "POST"])
def segment():
    logging.info("receive a predict request")

    if request.method == "GET":
        param_dic = request.args
    else:
        param_dic = request.form

    try:
        if "img" in param_dic:
            img_buff = param_dic['img']
            img = io.imread(StringIO(base64.b64decode(img_buff)))
            logging.info("image shape: {}".format(img.shape))
        else:
            img_url = param_dic["url"]
            img = io.imread(img_url, timeout=5)
    except Exception as e:
        logging.error("load image from param: {}, error: {}".format(param_dic, e))
        return jsonify({"success": False})

    if len(img.shape) != 3:
        img = color.gray2rgb(img)
    elif img.shape[2] > 3:
        img = img[:, :, :3]
    tmp_fn = str(random.randint(0, 10000)+100000)
    tmp_file = os.path.join(TMP_IMG_DIR, tmp_fn+".jpg")
    tmp_with_rect_file = os.path.join(TMP_IMG_DIR, "rect_"+tmp_fn+".jpg")
    io.imsave("static/"+tmp_file, img)
    # cv2.imwrite("static/"+tmp_file, img)
    img_inp = tfnet.framework.preprocess(img)
    out = tfnet.sess.run(tfnet.out, feed_dict={tfnet.inp: img_inp.reshape([1]+list(img_inp.shape))})
    valid_box = BoxCalculator.calc_rect(out[0])
    return jsonify({"success": True, "result": valid_box})


@app.route('/submit', methods=["GET", "POST"])
def predict():
    logging.debug("receive a predict request")

    if request.method == "GET":
        img_url = request.args.get("url")
    else:
        img_url = request.form.get("url")

    try:
        img = io.imread(img_url, timeout=5)
    except Exception as e:
        logging.error("load image from url: %s, error: %s", img_url, e)
        return jsonify({"success": False})

    if len(img.shape) != 3:
        img = color.gray2rgb(img)
    elif img.shape[2] > 3:
        img = img[:, :, :3]
    tmp_fn = str(random.randint(0, 10000)+100000)
    tmp_file = os.path.join(TMP_IMG_DIR, tmp_fn+".jpg")
    tmp_with_rect_file = os.path.join(TMP_IMG_DIR, "rect_"+tmp_fn+".jpg")
    io.imsave("static/"+tmp_file, img)
    # cv2.imwrite("static/"+tmp_file, img)
    img_inp = tfnet.framework.preprocess(img)
    out = tfnet.sess.run(tfnet.out, feed_dict={tfnet.inp: img_inp.reshape([1]+list(img_inp.shape))})

    print out.shape
    prediction = out[0]
    # print tfnet.framework.meta
    img_with_rect = tfnet.framework.postprocess(prediction, "static/"+tmp_file, save=False)
    # io.imsave("static/"+tmp_with_rect_file, img_with_rect)
    cv2.imwrite("static/"+tmp_with_rect_file, img_with_rect)

    search_for = {"src": url_for("static", filename=tmp_file), "iid": 1, "cid": 1}
    result = [{"iid": 0, "href": "xxx", "src": url_for("static", filename=tmp_with_rect_file), "sim":1}]
    # return jsonify({"success": True})
    return render_template("index.html", stupid=-1, status=StupidCount, has_result=True, search_for=(True, search_for),
                           similar=result)

@app.route('/')
def index():
    return render_template('index.html', has_result=False)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=False, port=9912)

