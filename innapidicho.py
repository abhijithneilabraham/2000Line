#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 20:17:57 2019

@author: abhijithneilabraham
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 22:15:55 2019

@author: abhijithneilabraham
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 11:27:24 2019

@author: abhijithneilabraham
"""

import cv2
import numpy as np



# handle command line arguments
#ap = argparse.ArgumentParser()
#ap.add_argument('-i', '--image', required=True,
#                help = 'path to input image')
#ap.add_argument('-c', '--config', required=True,
#                help = 'path to yolo config file')
#ap.add_argument('-w', '--weights', required=True,
#                help = 'path to yolo pre-trained weights')
#ap.add_argument('-cl', '--classes', required=True,
#                help = 'path to text file containing class names')
#args = ap.parse_args()
im='dog.jpg'
weights='yolov3.weights'
objclasses='yolov3.txt'
config='yolov3.cfg'
cap=cv2.VideoCapture(0)



def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

# function to draw bounding box on the detected object with class name
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


# initialization
class_ids = []
confidences = []
boxes = []
conf_threshold = 0.5
nms_threshold = 0.4
with open(objclasses, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# for each detetion from each output layer 
# get the confidence, class id, bounding box params
# and ignore weak detections (confidence < 0.5)
while True:
    ret,image =cap.read()
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

# read class names from text file



    # generate different colors for different classes 
    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
    
    # read pre-trained model and config file
    net = cv2.dnn.readNet(weights, config)
    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

# set input blob for the network
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))
    
    # create input blob 

    
    # set input blob for the network
    net.setInput(blob)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    
    # go through the detections remaining
    # after nms and draw bounding box
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        label = str(classes[class_ids[i]])
    #print(label)
    
    
        draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
    
    # display output image    
    cv2.imshow("object detection", image)
    
    # wait until q key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
     # save output image to disk
    cv2.imwrite("object-detection.jpg", image)
    
  
# release resources
cap.release()
cv2.destroyAllWindows()