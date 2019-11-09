#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 17:17:05 2019

@author: abhijithneilabraham
"""

import cv2
cap=cv2.VideoCapture(0)
i=2
while(i>0):
    ret,frame=cap.read()
    if i==1:
        cv2.imwrite("test1.jpg",frame)
    i-=1
