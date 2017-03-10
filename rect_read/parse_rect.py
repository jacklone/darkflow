# -*- coding: utf-8 -*-

"""
Created on 12/22/16 10:48 AM
File name   :   parse_rect.py
Author      :   jacklone
Email       :   jichenglong1988@gmail.com
Description :

"""

import struct
import matplotlib.pylab as plt
import pymongo
import cv2
from skimage import io
import sys

MongoOS = "mongodb://spider:spiderpasw@10.0.0.95/tts_spider_deploy?authMechanism=SCRAM-SHA-1"
DbName = "tts_spider_deploy"
CollName = "t_spider_goods_info"

DbClient = pymongo.MongoClient(MongoOS)
DbColl = DbClient[DbName][CollName]

DelColor = (255, 0, 0)
ValidColor = (0, 0, 255)


def parse_one(fi):
    iid, = struct.unpack("Q", fi.read(8))
    n_box, = struct.unpack("i", fi.read(4))
    boxes = []
    for i in range(n_box):
        x, y, w, h, l = struct.unpack("ffffi", fi.read(20))
        boxes.append([x, y, w, h, l])
    return [iid, n_box, boxes]


def plot_rect(img, box):
    rows, cols = img.shape[:2]
    x, y, w, h, _ = box
    rect_top = int((y-h/2) * rows)
    rect_btm = int((y+h/2) * rows)
    rect_left = int((x-w/2) * cols)
    rect_right = int((x+w/2) * cols)
    color = ValidColor if w*h > 0.2 else DelColor
    cv2.rectangle(img, (rect_left, rect_top), (rect_right, rect_btm), color=color)


def download_image(iid):
    proj = {"imageUrl": True}
    db_obj = DbColl.find_one({"_id": iid}, projection=proj)
    img = None
    if not db_obj is None:
        try:
            img_url = db_obj["imageUrl"]
            img = io.imread(img_url)
        except Exception as e:
            print "error: {}".format(e)
    else:
        print "iid: {} is not in db".format(iid)

    return img


if __name__ == "__main__":
    fn = "rect.bin"
    fi = open(fn)

    import random
    tid = random.randint(0, 1000)
    try:
        for i in range(tid):
            iid, n_box, boxes = parse_one(fi)
    except Exception as e:
        print "error: {}".format(e)
        sys.exit()

    print iid, n_box, boxes
    img = download_image(iid)
    if not img is None:
        for i in range(n_box):
            plot_rect(img, boxes[i])

        plt.imshow(img)
        plt.show(block=False)
        plt.waitforbuttonpress()


