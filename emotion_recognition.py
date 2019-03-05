#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 22:46:17 2019

@author: abhijithneilabraham
"""
#step 1 goes here
import cv2
import dlib
import glob
import random
import math
import numpy as np
import itertools
from sklearn.svm import SVC
emotions = ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
cap=cv2.VideoCapture(0)
det=dlib.get_frontal_face_detector() #face detetion purposes
pred=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #landmark plotter
while(1): 
    cap.grab()
    ret,frame=cap.retrieve()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)) #using histogram equalisation using cla    clahe_image=clahe.apply(gray)
    found=det(clahe_image,1)
    for d in found:
        shape=pred(clahe_image,d)
        for i in range(1,68):
            cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 1, (0,0,255), thickness=2) #For each point, draw a red circle with thickness2 on the original frame
    cv2.imshow("image", frame) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break