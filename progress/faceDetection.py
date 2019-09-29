import numpy as np
import cv2

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
test_cascade = cv2.CascadeClassifier('cascade20.xml')

cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #faces = face_cascade.detectMultiScale(gray, 1.9, 2)

    #test = test_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2)
    faces, rl, wl = face_cascade.detectMultiScale3(img, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30), flags=0, outputRejectLevels=True)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    '''for (x, y, w, h) in test:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)'''

    cv2.imshow('frame', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
