# -*- coding: utf-8 -*-

"""
Created on 12/20/16 10:41 AM
File name   :   calc_engine2.py
Author      :   jacklone
Email       :   jichenglong1988@gmail.com
Description :

"""

from model_client import tfnet
import tensorflow as tf
import threading
import numpy as np
import pymongo
from skimage import io, transform, color
from StringIO import StringIO
import matplotlib.pylab as plt

# MongoOS = "199.155.122.196"
MongoOS = "mongodb://spider:spiderpasw@10.0.0.95/tts_spider_deploy?authMechanism=SCRAM-SHA-1"
DbName = "tts_spider_deploy"
CollName = "t_spider_goods_info"

DbClient = pymongo.MongoClient(MongoOS)
DbColl = DbClient[DbName][CollName]

image_shape = tfnet.inp.get_shape()[1:]
batch_size = 64
nthread = 6

info_ph = tf.placeholder(dtype=tf.string, shape=[None, 1])
info_queue = tf.FIFOQueue(capacity=1000, dtypes=tf.string)
info_enqueue_op = info_queue.enqueue()
info_dequeue_op = info_queue.dequeue()

# queue = tf.RandomShuffleQueue(capacity=5000, min_after_dequeue=800, dtypes=[tf.float32], shapes=[flatten_shape])
queue = tf.FIFOQueue(capacity=5000, dtypes=[tf.float32], shapes=tfnet.inp.get_shape()[1:])
enqueue_op = queue.enqueue(tfnet.inp)
dequeue_op = queue.dequeue_many(batch_size)


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


def enqueue_image(sess, coord):
    cur = DbColl.find()
    while not coord.should_stop():
        if not cur.alive:
            cur = DbColl.find()
        img_obj = cur.next()
        image = io.imread(StringIO(img_obj['data']))
        reshape_img = img_preprocess(image)

        sess.run(enqueue_op, feed_dict={tfnet.inp: reshape_img})

    print "enqueue operation stop"


init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = [threading.Thread(target=enqueue_image, args=[sess, coord, ]) for i in range(nthread)]
    for t in threads:
        t.start()

    for epoch_i in range(n_epoch):
        for batch_i in range(n_batch):
            batch_imgs = sess.run(dequeue_op)
            _, summary = sess.run([optimizer, summary_op], feed_dict={ae['x']: batch_imgs})


        batch_imgs = sess.run(dequeue_op)
        cost = sess.run(ae['cost'], feed_dict={ae['x']: batch_imgs})
        print "epoch", epoch_i, "cost:", cost

    # save the model
    saver.save(sess, "/tmp/tf_model")

    # plot test examples
    n_examples = 5
    test_xs = sess.run(dequeue_op)
    recon = sess.run(ae['y'], feed_dict={ae['x']: test_xs})
    fig, axs = plt.subplots(2, n_examples, figsize=(20, 2))
    for example_i in xrange(n_examples):
        axs[0][example_i].imshow(np.reshape(test_xs[example_i], image_shape))
        axs[1][example_i].imshow(np.reshape(recon[example_i], image_shape))
        print "example", example_i, "mean:", test_xs[example_i].mean(), "vs", recon[example_i].mean()

    print "example", example_i, ":", test_xs[example_i], "vs", recon[example_i]
    fig.show()
    plt.draw()

    coord.request_stop()
    plt.imshow(batch_imgs[0].reshape(image_shape))
    plt.show()
    coord.join(threads)
