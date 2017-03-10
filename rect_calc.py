# -*- coding: utf-8 -*-

"""
Created on 12/17/16 9:57 AM
File name   :   rect_calc.py
Author      :   jacklone
Email       :   jichenglong1988@gmail.com
Description :

"""

import pymongo
import threading
from threading import Lock
import Queue
import requests
from StringIO import StringIO
from skimage import io, transform, color
import urllib2
import numpy as np
from model_client import tfnet
from utils.box import BoundBox, prob_compare2, box_intersection
import json
import math
import time
import random
import struct

from zk_client import *
from proxy_client import PROXY_LIST, zk_listener, watch_proxy_node
ZK_CLIENT.add_listener(zk_listener)
ZK_CLIENT.ChildrenWatch(ZK_LOCK_PATH, watch_proxy_node)

MongoOS = "mongodb://spider:spiderpasw@10.0.0.95/tts_spider_deploy?authMechanism=SCRAM-SHA-1"
DbName = "tts_spider_deploy"
CollName = "t_spider_goods_info"

DbClient = pymongo.MongoClient(MongoOS)
DbColl = DbClient[DbName][CollName]

image_shape = [int(i) for i in tfnet.inp.get_shape().as_list()[1:]]
batch_size = 16

n_download_workers = 32
n_save_workers = 2
UrlQueue = Queue.Queue(1000)
ImgQueue = Queue.Queue(100)
RectQueue = Queue.Queue(1000)
PrintLock = Lock()
SaveLock = Lock()

RequestHeads = {"User-Agent":
                    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 "
                    "Safari/537.36"}


def say(*msgs):
    msgs = list(msgs)
    for msg in msgs:
        if msg is None: continue
        PrintLock.acquire()
        print msg
        PrintLock.release()


def enqueue_url(coll, n_worker):
    total = 0
    for obj in coll.find(projection={"_id": True, "imageUrl": True}, no_cursor_timeout=True):
        total += 1

        if "_id" in obj and "imageUrl" in obj:
            UrlQueue.put(obj)

    # put some end task into the queue
    for i in range(n_worker):
        UrlQueue.put(None)

    PrintLock.acquire()
    print "enqueue url thread stop, create task number:", total
    PrintLock.release()


def request_url(url):
    req = urllib2.urlopen(url)
    cont = req.read()


def img_preprocess(img):
    """
    according to function preprocess in yolo/test.py, resize & rgb transform
    :param img:
    :return:
    """
    if len(img.shape) != 3:
        img = color.gray2rgb(img)
    elif img.shape[2] > 3:
        img = img[:, :, :3]
    reshape_img = np.expand_dims(transform.resize(img, image_shape)[:,:,::-1], 0)
    return reshape_img


def image_download(tid):
    success = 0
    total = 0
    buff_objs = []
    buff_iids = []
    # while not UrlQueue.empty():
    while True:
        urlObj = UrlQueue.get()

        if urlObj is None:
            PrintLock.acquire()
            print "image download thread", tid, "receive a None task..."
            PrintLock.release()
            break

        total += 1
        # if total % 10 == 0:
        #     PrintLock.acquire()
        #     print "download thread total success:", success, " url queue: ", UrlQueue.qsize(), \
        #         " download queue:", ImgQueue.qsize(), " rect queue:", RectQueue.qsize()
        #     PrintLock.release()

        image_url = urlObj['imageUrl']
        d_succ = False
        for i in range(3):
            try:
                ip = PROXY_LIST[random.randint(0, 2)][0]
                # ip = PROXY_LIST[0][0]
                proxy = {"http": "http://%s:3128" % ip, "https": "http://%s:3128" % ip}
                res = requests.get(image_url, proxies=proxy, timeout=10)
                cont = res.content
                img = io.imread(StringIO(cont))
                reshape_img = img_preprocess(img)
                d_succ = True
                break
            except Exception as e:
                continue

        if not d_succ:
            say("url read error, iid: {}, url: {}".format(urlObj['_id'], image_url))
            continue

        buff_iids.append(urlObj["_id"])
        buff_objs.append(reshape_img)
        success += 1

        if len(buff_iids) == batch_size:
            ImgQueue.put([buff_iids, np.concatenate(buff_objs, 0)])
            buff_iids = []
            buff_objs = []

    if len(buff_iids) > 0:
        ImgQueue.put([buff_iids, np.concatenate(buff_objs, 0)])

    ImgQueue.put(None)
    PrintLock.acquire()
    print "image download thread", tid, " stop, get total:", total, ", success:", success
    PrintLock.release()


def image_calc(n_worker, n_saver):
    success = 0
    while True:
        imgObj = ImgQueue.get()
        if imgObj is None:
            n_worker -= 1
            if n_worker == 0:
                PrintLock.acquire()
                print "all stop tasks received"
                PrintLock.release()
                break
            else:
                PrintLock.acquire()
                print "receive one stop task, need", n_worker, "more"
                PrintLock.release()
                continue

        else:
            success += 1
            batch_iid, batch_data = imgObj

            result = tfnet.sess.run(tfnet.out, feed_dict={tfnet.inp: batch_data})
            RectQueue.put([batch_iid, result])

            # if success % 10 == 0:
            #     PrintLock.acquire()
            #     print "calc thread total success:", success, " url queue: ", UrlQueue.qsize(), \
            #         " download queue:", ImgQueue.qsize(), " rect queue:", RectQueue.qsize()
            #     PrintLock.release()

    for i in range(n_saver):
        RectQueue.put(None)

    PrintLock.acquire()
    print "insert image data thread stop, total success:", success
    PrintLock.release()


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


def save_boxes(iid, boxes, fo):
    iid_st = struct.pack("q", iid)
    fo.write(iid_st)
    fo.write(struct.pack("i", len(boxes)))
    for b in boxes:
        for i in range(4):
            fo.write(struct.pack("f", b[i]))
        fo.write(struct.pack("i", b[4]))
        # fo.write(struct.pack("f", b[0]))
        # fo.write(struct.pack("f", b.y))
        # fo.write(struct.pack("f", b.w))
        # fo.write(struct.pack("f", b.h))
        # fo.write(struct.pack("i", b.ind))


def save_netout(iid, netout, fo):
    iid_st = struct.pack("q", iid)
    fo.write(iid_st)
    for o in netout:
        fo.write(struct.pack("f", o))


def rect_save(coll, tid, box_calculator, fo):
    success = 0
    total = 0
    start = time.time()
    # while not UrlQueue.empty():
    while True:
        rectObj = RectQueue.get()

        if rectObj is None:
            say("rect save thread {} receive a None task...".format(tid))
            break

        total += 1
        for out_i, iid in enumerate(rectObj[0]):
            net_out = rectObj[1][out_i]
            valid_boxes = box_calculator.calc_rect(net_out)

            SaveLock.acquire()
            save_boxes(iid, valid_boxes, fo)
            SaveLock.release()

            success += 1
            if success % 100 == 0:
                say("rect save thread {} total success: {}, url queue: {}," \
                    " download queue: {}, rect queue: {}, v: {}".format(tid, success, UrlQueue.qsize(),
                                                                      ImgQueue.qsize(), RectQueue.qsize(),
                                                                      success / (time.time()-start)))

    say("rect save thread {} stop, get total: {} , success: {}".format(tid, total, success))


def rect_save2(coll, tid, box_calculator, fo):
    success = 0
    total = 0
    start = time.time()
    # while not UrlQueue.empty():
    while True:
        rectObj = RectQueue.get()

        if rectObj is None:
            say("rect save thread {} receive a None task...".format(tid))
            break

        total += 1
        for out_i, iid in enumerate(rectObj[0]):
            net_out = rectObj[1][out_i]
            valid_boxes = box_calculator.calc_rect(net_out)
            json_str = json.dumps(valid_boxes)

            SaveLock.acquire()
            fo.write(str(iid) + "\t" + json_str+"\n")
            SaveLock.release()

            success += 1
            if success % 100 == 0:
                PrintLock.acquire()
                print "rect save thread total success:", success, " url queue: ", UrlQueue.qsize(), \
                    " download queue:", ImgQueue.qsize(), " rect queue:", RectQueue.qsize(), \
                    " v: ", success / (time.time()-start)
                PrintLock.release()

    say("rect save thread {} stop, get total: {} , success: {}".format(tid, total, success))

box_calculator = RectCalculator()
fo = open("rect.bin", "wb")
url_thread = threading.Thread(target=enqueue_url, args=[DbColl, n_download_workers, ])
download_threads = [threading.Thread(target=image_download, args=[i, ]) for i in range(n_download_workers)]
calc_thread = threading.Thread(target=image_calc, args=[n_download_workers, n_save_workers, ])
save_threads = [threading.Thread(target=rect_save, args=[None, i, box_calculator, fo]) for i in range(n_save_workers)]

url_thread.setDaemon(True)
url_thread.start()
for t in download_threads:
    t.setDaemon(True)
    t.start()

calc_thread.start()
for t in save_threads:
    t.start()

for t in save_threads:
    t.join()

fo.close()

print "all thread stop!"
