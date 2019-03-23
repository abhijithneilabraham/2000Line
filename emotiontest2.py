#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 10:56:52 2019

@author: abhijithneilabraham
"""
import numpy as np
with open("/Users/abhijithneilabraham/Documents/challenges-in-representation-learning-facial-expression-recognition-challenge/fer2013/fer2013.csv") as f:
    content = f.readlines()
     
    lines = np.array(content)
     
    num_of_instances = lines.size
    print("number of instances: ",num_of_instances)
