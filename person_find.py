# -*- coding: utf-8 -*-

"""
Created on 12/19/16 3:33 PM
File name   :   person_find.py
Author      :   jacklone
Email       :   jichenglong1988@gmail.com
Description :

"""

from flask import Flask, request, jsonify, url_for, render_template
from skimage import io
import cv2
from model_client import tfnet
import logging
import os
import random

TMP_IMG_DIR = "tmp"
StupidCount = {0: 0, 1: 0, 2: 0}

logging.basicConfig(level=logging.WARN,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%Y %b %H:%M:%S')

app = Flask(__name__)


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

    tmp_fn = str(random.randint(0, 10000)+100000)
    tmp_file = os.path.join(TMP_IMG_DIR, tmp_fn+".jpg")
    tmp_with_rect_file = os.path.join(TMP_IMG_DIR, "rect_"+tmp_fn+".jpg")
    io.imsave("static/"+tmp_file, img)
    # cv2.imwrite("static/"+tmp_file, img)
    img_inp = tfnet.framework.preprocess(img)
    out = tfnet.sess.run(tfnet.out, feed_dict={tfnet.inp: img_inp.reshape([1]+list(img_inp.shape))})

    prediction = out[0]
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
    app.run(host='0.0.0.0', debug=True, port=9911)

