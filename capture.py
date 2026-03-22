import cv2
import urllib.request
import numpy as np
import os

# Check karo ki folder hai ya nahi
if not os.path.exists('data'):
    os.makedirs('data')

f_cas = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
url = 'http://10.200.1.131/capture' # Aapka ESP32-CAM URL

count = 0
print("Please come in front of the camera and Move your face slowly...")

while count < 30:
    try:
        img_resp = urllib.request.urlopen(url)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        img = cv2.imdecode(imgnp, -1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        face = f_cas.detectMultiScale(gray, 1.3, 5)
        
        for (x,y,w,h) in face:
            count += 1
            # Photo save karna (ID = 1 Nikhil ke liye)
            
            file_name = f"data/User.2.{count}.jpg"
            cv2.imwrite(file_name, gray[y:y+h, x:x+w])
            
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
            cv2.putText(img, f"Captured: {count}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            print(f"Photo {count} saved!")

        cv2.imshow("Capturing Photos...", img)
        
    except Exception as e:
        print(f"Error: {e}")
        break

    if cv2.waitKey(1) == ord('q'): break

print("\n[SUCCESS] 30 Photos have been captured!")
cv2.destroyAllWindows()