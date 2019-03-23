#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 10:56:52 2019

@author: abhijithneilabraham
"""


from keras.preprocessing import image
import numpy as np
import cv2

from keras.models import load_model
import matplotlib.pyplot as plt
model=load_model('test2.h5')



def emotion_analysis(emotions):
    objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    y_pos = np.arange(len(objects))
    
    plt.bar(y_pos, emotions, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('percentage')
    plt.title('emotion')
    
    plt.show()
cap=cv2.VideoCapture(0)
'''   
img = image.load_img("hugh.jpeg", grayscale=True, target_size=(48, 48))

x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)

x /= 255
'''
while(1):
    cap.grab()
    ret,frame=cap.retrieve()
    cv2.imwrite("test1.jpg",frame)
    frame = cv2.resize(frame,(48, 48), interpolation = cv2.INTER_CUBIC)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    x = image.img_to_array(gray)
    x = np.expand_dims(x, axis = 0)

    x /= 255
    custom = model.predict(x)
    maxim=max(custom[0])
    em=[l for l, k in enumerate(custom[0]) if k == maxim]
    objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    print(objects[em[0]])

    
    cv2.imshow('fra',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
'''
custom = model.predict(x)

emotion_analysis(custom[0])

x = np.array(x, 'float32')
x = x.reshape([48, 48])
for i in custom[0]:
    print(i*100,"%")
plt.gray()
plt.imshow(x)
plt.show()
'''
 





