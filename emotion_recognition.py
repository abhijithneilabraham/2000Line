#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 22:46:17 2019

@author: abhijithneilabraham
"""
#step 1 goes here
import cv2
import dlib
cap=cv2.VideoCapture(0)
det=dlib.get_frontal_face_detector()
pred=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
while(1):
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)