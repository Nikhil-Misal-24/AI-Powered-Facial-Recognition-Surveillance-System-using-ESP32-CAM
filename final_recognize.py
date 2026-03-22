import cv2
import urllib.request
import numpy as np

# 1. Trained model aur Cascade load karein
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml') # Jo file abhi bani hai
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Naam ki list (ID 1 par Nikhil hai)
names = ['None', 'Nikhil Misal','Rahul Shrivastava sir'] 

url = 'http://10.200.1.131/capture'

print("\n [INFO] The Camera is turning ON...")

while True:
    img_resp = urllib.request.urlopen(url)
    imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    img = cv2.imdecode(imgnp, -1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.2, 5)

    for(x,y,w,h) in faces:
        # Match check karna
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        # Confidence < 100 ka matlab hai match mil gaya (jitna kam utna behtar)
        if (confidence < 80):
            name = names[id]
            conf_text = "  {0}%".format(round(100 - confidence))
        else:
            name = "Unknown"
            conf_text = "  {0}%".format(round(100 - confidence))
        
        # Green box agar Nikhil hai, Red agar Unknown
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        
        cv2.rectangle(img, (x,y), (x+w,y+h), color, 2)
        cv2.putText(img, str(name), (x+5,y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(img, str(conf_text), (x+5,y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1)  

    cv2.imshow('SISTec Face ID...', img)
    if cv2.waitKey(1) == ord('q'): break

print("\n [INFO] Program band ho raha hai.")
cv2.destroyAllWindows()