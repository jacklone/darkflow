# -*- coding: utf-8 -*-

"""
Created on 2016/8/24 11:46
File name   :   proxy_client.py
Author      :   jacklone
Email       :   jichenglong1988@gmail.com
Description :

"""

from multiprocessing import Manager
from zk_client import *
import logging

logger = logging.getLogger("log.proxy")

PORT = 3128
DEFAULT_INTERVAL = "20"

##########################################################################################

# PROXY_LIST = []  # list of ("199.155.122.89", 18889), 按锁大小从大到小排列
manager = Manager()
PROXY_LIST = manager.list()
CURRENT_LOCK_ID = 0

##########################################################################################


def zk_listener(state):
    if state == KazooState.LOST:
        # Register somewhere that the session was lost
        logger.warn("[zk listener] monitor LOST state, restart and reset listener")
    elif state == KazooState.SUSPENDED:
        # Handle being disconnected from Zookeeper
        logger.warn("[zk listener] monitor SUSPENDED state, restart and reset listener")
    else:  # CONNECTED
        logger.warn("[zk listener] monitor CONNECTED state, reset watchers")


def watch_proxy_node(children):
    global CURRENT_LOCK_ID

    # 从小到大排列
    children.sort(key=int)

    # 删除过期代理
    min_lock_idx = int(children[0])
    for old_proxy in PROXY_LIST[::-1]:
        if old_proxy[1] < min_lock_idx:
            proxy = PROXY_LIST.pop()
            logger.info("[proxy client] monitor deleted proxy: %s", proxy)
        else:
            break

    # 新增代理
    for p in children:
        p_idx = int(p)

        if p_idx <= CURRENT_LOCK_ID:  # 已经在用的代理
            continue

        # 新代理
        zk_path = ZK_LOCK_PATH + "/" + p
        try:
            data, stat = ZK_CLIENT.get(zk_path)
        except Exception as e:
            logger.warn("[proxy client] get client ip from zk_path[%s] failed, maybe redialing, error: %s", zk_path, e)
            continue

        PROXY_LIST.insert(0, (data, p_idx))
        CURRENT_LOCK_ID = p_idx
        logger.info("[proxy client] new proxy found, ip=%s, lock=%d", data, p_idx)


