import numpy as np
import cv2
# import sys
import imutils

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
# face_cascade = cv2.CascadeClassifier('src/haarcascade_frontalface_default.xml')
# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
# eye_cascade = cv2.CascadeClassifier('src/haarcascade_eye.xml')
# mouse_cascade = cv2.CascadeClassifier('src/test_cascade.xml')

prototxt = "deploy.prototxt"
model = "face_detector.caffemodel"

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt, model)

print("[INFO] starting video stream...")

cap = cv2.VideoCapture('recordtired.avi')
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('out3.avi', fourcc, 20.0, (640, 480))
while cap.isOpened():
    ret, frame = cap.read()

    if ret is True:
        # cv2.imwrite("/home/nao/WORKSPACE/src/videoprocesspic.png", frame)
        # img_gray = cv2.imread("/home/nao/WORKSPACE/src/videoprocesspic.png", cv2.IMREAD_GRAYSCALE)
        frame = imutils.resize(frame, width=640, height=480)

        (height, weight) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (640, 480)), 1.0,
                                     (640, 480), (104.0, 177.0, 123.0))

        net.setInput(blob)
        detections = net.forward()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # marker = self.find_object(gray, face_cascade)

        '''faces = face_cascade.detectMultiScale3(
            gray,
            scaleFactor=1.7,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
            outputRejectLevels=True
        )
        rects = faces[0]
        weights = faces[2]'''

        # for (x, y, w, h) in rects:
        #     cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < 0.5:
                continue
            box = detections[0, 0, i, 3:7] * np.array([weight, height, weight, height])
            (startX, startY, endX, endY) = box.astype("int")
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else endY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (255, 0, 0), 2)
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (11, 255, 255), 2)

            '''if type(weights) == "<class 'tuple'>":
                continue
            else:
                for weight in weights:
                    print(weight)
                    for q in weight:
                        print(q)
                        result = "{:.2f}".format(q)
                        # weight = weights.tostring()
                        # print(np.fromstring(weight, dtype=int))
                        cv2.putText(frame, result, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                    (11, 255, 255), 2)'''

        cv2.imshow("Frame", frame)
        out.write(frame)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    else:
        break
out.release()
cap.release()
cv2.destroyAllWindows()
