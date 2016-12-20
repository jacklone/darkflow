# -*- coding: utf-8 -*-

"""
Created on 12/19/16 3:42 PM
File name   :   model_client.py
Author      :   jacklone
Email       :   jichenglong1988@gmail.com
Description :

"""
from net.build import TFNet
from tensorflow import flags
import os

flags.DEFINE_string("test", "./test/", "path to testing directory")
flags.DEFINE_string("binary", "./bin/", "path to .weights directory")
flags.DEFINE_string("config", "./cfg/", "path to .cfg directory")
flags.DEFINE_string("dataset", "../pascal/VOCdevkit/IMG/", "path to dataset directory")
flags.DEFINE_string("backup", "./ckpt/", "path to backup folder")
flags.DEFINE_string("annotation", "../pascal/VOCdevkit/ANN/", "path to annotation directory")
flags.DEFINE_float("threshold", 0.1, "detection threshold")
flags.DEFINE_string("model", "", "configuration of choice")
flags.DEFINE_string("trainer", "rmsprop", "training algorithm")
flags.DEFINE_float("momentum", 0.0, "applicable for rmsprop and momentum optimizers")
flags.DEFINE_boolean("verbalise", True, "say out loud while building graph")
flags.DEFINE_boolean("train", False, "train the whole net")
flags.DEFINE_string("load", "", "how to initialize the net? Either from .weights or a checkpoint, or even from scratch")
flags.DEFINE_boolean("savepb", False, "save net and weight to a .pb file")
flags.DEFINE_float("gpu", 0.0, "how much gpu (from 0.0 to 1.0)")
flags.DEFINE_float("lr", 1e-5, "learning rate")
flags.DEFINE_integer("keep", 20, "Number of most recent training results to save")
flags.DEFINE_integer("batch", 12, "batch size")
flags.DEFINE_integer("epoch", 1000, "number of epoch")
flags.DEFINE_integer("save", 2000, "save checkpoint every ? training examples")
flags.DEFINE_string("demo", '', "demo on webcam")
flags.DEFINE_boolean("profile", False, "profile")
FLAGS = flags.FLAGS


# make sure all necessary dirs exist
def get_dir(dirs):
    for d in dirs:
        this = os.path.abspath(os.path.join(os.path.curdir, d))
        if not os.path.exists(this): os.makedirs(this)


get_dir([FLAGS.test, FLAGS.binary, FLAGS.backup, os.path.join(FLAGS.test, 'out')])

# fix FLAGS.load to appropriate type
try:
    FLAGS.load = int(FLAGS.load)
except:
    pass

tfnet = TFNet(FLAGS)

if FLAGS.profile:
    tfnet.framework.profile(tfnet)
    exit()

if FLAGS.demo:
    tfnet.camera(FLAGS.demo)
    exit()

if FLAGS.train:
    print 'Enter training ...';
    tfnet.train()
    if not FLAGS.savepb: exit('Training finished')

if FLAGS.savepb:
    print 'Rebuild a constant version ...'
    tfnet.savepb();
    exit('Done')

