import cv2
import os
import numpy as np
from PIL import Image # Agar error aaye toh 'pip install Pillow' karein

# 1. Recognizer aur Detector initialize karein
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# 2. Photos read karne ka function
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []

    for imagePath in imagePaths:
        # Gray scale mein convert karke open karein
        PIL_img = Image.open(imagePath).convert('L') 
        img_numpy = np.array(PIL_img, 'uint8')

        # File name se ID nikalna (User.1.30.jpg -> ID = 1)
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        
        faces = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return faceSamples, ids

print("\n [INFO] Training faces... Just a movment (please wait).")
faces, ids = getImagesAndLabels('data')

# 3. Model ko train karke save karna
recognizer.train(faces, np.array(ids))
recognizer.write('trainer.yml') # Ye file sabse important hai

print(f"\n [INFO] {len(np.unique(ids))} face(s) trained. 'trainer.yml' Created.")