# -*- coding: utf-8 -*-

"""
Created on 2016/8/22 13:54
File name   :   zk_client.py
Author      :   jacklone
Email       :   jichenglong1988@gmail.com
Description :

"""

from kazoo.client import KazooClient, KazooState


ZK_HOST = "199.155.122.131:2181"
ZK_ROOT = "/adsl_proxy"
ZK_IP_PATH = "/adsl_proxy/ip"
ZK_LOCK_PATH = "/adsl_proxy/lock"
ZK_CONFIG_PATH = "/adsl_proxy/config"

ZK_CLIENT = KazooClient(hosts=ZK_HOST)
ZK_CLIENT.start()



