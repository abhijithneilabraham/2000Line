#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 22:35:40 2019

@author: abhijithneilabraham
"""

import cv2
import face_recognition
input_movie = cv2.VideoCapture(0)
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))
image = face_recognition.load_image_file("test.jpg")
face_encoding = face_recognition.face_encodings(image)[0]

known_faces = [
face_encoding,
]
# Initialize variables
face_locations = []
face_encodings = []
face_names = []
while True:
    # Grab a single frame of video
    ret, frame = input_movie.read()
    

    # Quit when the input video file ends
    if not ret:
        break

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame, model="cnn")
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)

        name = None
        if match[0]:
            name = "Abhijith Neil Abraham"

        face_names.append(name)

    # Label the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        if not name:
            continue

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        
        
        
        

    # Write the resulting image to the output video file
    cv2.imshow('fra',frame)

# All done!
input_movie.release()
cv2.destroyAllWindows()
