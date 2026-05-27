#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import logging

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename='my.log', level=logging.DEBUG, format=LOG_FORMAT)
logging.debug("This is a debug log.")
logging.info("This is a info log.")
logging.warning("This is a warning log.")
logging.error("This is a error log.")
logging.critical("This is a critical log.")

def log_print(log_str,level='info'):
    logging.info("This is a info log.")