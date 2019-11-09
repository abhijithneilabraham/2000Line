#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 01:18:16 2019

@author: abhijithneilabraham
"""

import pyqrcode as pq
ID = "12345"
Name = "K&K Parking"
qr = pq.create(f'ID:{ID}\nName:{Name}')
qr.svg("Code.svg", scale = 8)