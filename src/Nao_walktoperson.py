from __future__ import print_function
from imutils.object_detection import non_max_suppression
import os
import sys
import numpy as np
import imutils
import cv2
import time
import argparse
import Image
import random
import math
import errno
import almath as m  # python's wrapping of almath
from naoqi import ALProxy
from naoqi import ALBroker
from naoqi import ALModule
import speech_recognition as sr




KNOWN_DISTANCE = 30.0
KNOWN_PERSON_HEIGHT = 8

face_cascade = cv2.CascadeClassifier('/home/nao/WORKSPACE/src/haarcascade_frontalface_default.xml')
obj_cascade = cv2.CascadeClassifier('/home/nao/WORKSPACE/src/box_cascade.xml')
prototxt = "/home/nao/WORKSPACE/src/deploy.prototxt"
model = "/home/nao/WORKSPACE/src/face_detector.caffemodel"

speech_record_file_path = "/home/nao/WORKSPACE/src/speechRecord.wav"
sample_text_file_path = "/home/nao/WORKSPACE/src/sample_text.wav"

walkTo = None
memory = None


class walkTo(ALModule):

    def __init__(self, name):
        ALModule.__init__(self, name)

        self.tts = ALProxy("ALTextToSpeech")
        self.video_service = ALProxy("ALVideoDevice")
        self.motion = ALProxy("ALMotion")
        self.record = ALProxy("ALAudioRecorder")
        self.videoRecorderProxy = ALProxy("ALVideoRecorder")

        global memory
        memory = ALProxy("ALMemory")

        memory.subscribeToEvent("FrontTactilTouched",
                                "walkTo",
                                "onWake")

        memory.subscribeToEvent("MiddleTactilTouched",
                                "walkTo",
                                "onRest")

    def onWake(self, strVarName, value):

        memory.unsubscribeToEvent("FrontTactilTouched",
                                  "walkTo")

        if value > 0.0:
            self.communication()
            #self.walk()

        memory.subscribeToEvent("FrontTactilTouched",
                                "walkTo",
                                "onWake")

    def onRest(self, strVarName, value):

        memory.unsubscribeToEvent("MiddleTactilTouched",
                                  "walkTo")

        if value > 0.0:
            self.stop()
            os._exit(0)

        memory.subscribeToEvent("MiddleTactilTouched",
                                "walkTo",
                                "onRest")

    def to_cv_img(self, nao_img):
        image_width = nao_img[0]
        image_height = nao_img[1]
        array = nao_img[6]
        image_string = str(bytearray(array))

        result = Image.fromstring("RGB", (image_width, image_height), image_string)
        result.save("/home/nao/WORKSPACE/src/cameras/capture_0.png", "PNG")
        img_gray = cv2.imread("/home/nao/WORKSPACE/src/cameras/capture_0.png", cv2.IMREAD_GRAYSCALE)
        return img_gray

    def find_object(self, frame, cascade):
        object = cascade.detectMultiScale(frame, scaleFactor=1.9, minNeighbors=5, minSize=(35, 35))
        object = np.array([[x, y, x + w, y + h] for (x, y, w, h) in object])
        pick = non_max_suppression(object, probs=None, overlapThresh=0.65)
        return pick

    def find_face(self, frame, cascade):
        face = cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=3, minSize=(35, 35))
        face = np.array([[x, y, x + w, y + h] for (x, y, w, h) in face])
        pick = non_max_suppression(face, probs=None, overlapThresh=0.65)
        return pick

    def distance_to_camera(self, known_width, focal_length, per_width):
        return (known_width * focal_length) / per_width

    def get_focal_length(self):
        name_id = self.video_service.subscribe("getlength", 2, 11, 20)
        try:
            y1_max = 0
            y2_max = 0
            count = 0

            while True:
                nao_images = self.video_service.getImageRemote(name_id)
                if nao_images is None:
                    print("cannot capture images")
                    break
                cv_images = self.to_cv_img(nao_images)

                faces = self.find_face(cv_images, face_cascade)
                print(faces)
                print("-----")
                if len(faces) is not 0:
                    cv2.imwrite('/home/nao/WORKSPACE/src/cameras/image_0.png', cv_images)

                    face_frame = self.find_face(cv_images, face_cascade)
                    for (x1, y1, x2, y2) in face_frame:
                        # rectangle frame: img,pt1,pt2,color(green),thickness=None,lineType=None,shift=None
                        cv2.rectangle(cv_images, (x1, y1), (x2, y2), (255, 255, 255), 3)
                    cv2.imwrite('/home/nao/WORKSPACE/src/cameras/image_frame.png', cv_images)

                    break
                else:
                    count += 1
                    if count > 79:
                        return "no face detection"
                    continue

            img = cv2.imread('/home/nao/WORKSPACE/src/cameras/image_0.png')
            mask = self.find_face(img, face_cascade)
            for (x1, y1, x2, y2) in mask:
                # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
                y1_max = y1
                y2_max = y2


            pix_height = y2_max - y1_max
            real_focal_length = (pix_height * KNOWN_DISTANCE) / KNOWN_PERSON_HEIGHT
            # print("1",pix_height)
            # stop retrieving images
            self.video_service.unsubscribe(name_id)
            return real_focal_length

        except SystemExit as e:
            print("System error : {0}".format(e))
        except OSError as err:
            print("OS error: {0}".format(err))
        finally:
            print("Unsubscribe the video!")
            self.video_service.unsubscribe(name_id)

    def stop(self):
        print("See you next time ")
        self.tts.say(" See you next time")

        self.motion.stopMove()
        self.motion.rest()

        self.videoRecorderProxy.stopRecording()
        # stop retrieving images
        self.video_service.unsubscribe(self.video_service.subscribe("getlength", 1, 11, 20))
        self.video_service.unsubscribe(self.video_service.subscribe("walkToH", 1, 11, 20))

        self.processvideo()
    def headPitchYaw(self):
        names = ["HeadYaw", "HeadPitch"]
        angleLists = [[1.0, -1.0, 1.0, -1.0], [-1.0]]
        times = [[1.0, 2.0, 3.0, 4.0], [5.0]]
        isAbsolute = True
        self.motion.angleInterpolation(names, angleLists, times, isAbsolute)
        '''
        angleLists = [[1.0, 0.0], [-0.5, 0.5, 0.0]]
        timeLists = [[1.0, 2.0], [1.0, 2.0, 3.0]]
        isAbsolute = True
        self.motion.angleInterpolation(names, angleLists, timeLists, isAbsolute)
        '''

    def foot_step(self, marker, real_focal_length):
        # x = y = theta = 0
        # print(real_focal_length)
        pixel = 640
        if len(marker) == 0:
            marker = [[0, 0, 0, 0]]

            # self.tts.say("you are not in my vision")

        #print("1", marker)
        for (xA, yA, xB, yB) in marker:
            ya_max = yA
            yb_max = yB
            half_length = (xB - xA) / 2
            mid_point_value = xA + half_length

        pix_person_height = yb_max - ya_max
        # print("2", pix_person_height)
        if pix_person_height == 0:
            return 0, 0, 0
        inches = self.distance_to_camera(KNOWN_PERSON_HEIGHT, real_focal_length, pix_person_height)
        print("face detected", "%.2f inches" % inches)
        print("face detected", mid_point_value)
        # print("-------")

        if inches < 30.0:
            x = 0
            y = 0
            theta = 0

        else:
            if (pixel/2) - 80 <= mid_point_value < (pixel/2) + 80:
                x = 0.2
                y = 0
                theta = 0
                print("towards")

            elif (pixel/2) - 180 <= mid_point_value < (pixel/2) - 80:
                x = 0.3
                y = 0.2
                theta = math.pi / 9
                print("L", 1)

            elif 20 <= mid_point_value < (pixel/2) - 180:
                x = 0.3
                y = 0.2
                theta = math.pi / 6
                print("L", 2)

            elif 0 < mid_point_value < 20:
                x = 0.3
                y = 0.2
                theta = math.pi / 3
                print("L", 3)

            elif (pixel/2) + 80 <= mid_point_value < (pixel/2) + 180:
                x = 0.3
                y = -0.2
                theta = -math.pi / 9
                print("R", 1)

            elif (pixel/2) + 180 <= mid_point_value <= pixel - 20:
                x = 0.3
                y = -0.2
                theta = -math.pi / 6
                print("R", 2)

            elif pixel - 20 < mid_point_value <= pixel:
                x = 0.3
                y = -0.2
                theta = -math.pi / 3
                print("R", 3)

            else:
                x = 0
                y = 0
                theta = 0
                # print("R", 3)
        return x, y, theta

    def walk(self):
        # x, y, theta = 0
        global audio, r
        audio = r = None
        r = sr.Recognizer()

        count = 0
        self.motion.wakeUp()

        names = "HeadPitch"
        angles = -30.0 * m.TO_RAD
        fractionMaxSpeed = 0.1
        self.motion.setAngles(names, angles, fractionMaxSpeed)
        time.sleep(0.1)
        # resolution
        # kQQVGA (160x120), kQVGA (320x240), kVGA (640x480) or
        # k4VGA (1280x960, only with the HD camera).
        # color space desired
        # kYuvColorSpace, kYUVColorSpace, kYUV422InterlacedColorSpace, kRGBColorSpace, etc.
        # frames per second
        # Finally, select the minimal number of frames per second (fps) that your
        # vision module requires up to 30fps.
        name_id = self.video_service.subscribe("walkToH", 2, 11, 20)
        print("ALTracker successfully started, now show your face to robot!")
        print("Use Ctrl+c to stop this script.")
        pixel = 640
        try:
            while True:
                real_focal_length = self.get_focal_length()
                if real_focal_length == "no face detected":
                    self.tts.say("no face detection please try again")
                    continue
                else:
                    self.tts.say("ok I got your face ")
                    print("face detected")
                    break
            self.motion.moveInit()
            robotPositionBeforeCommand = m.Pose2D(self.motion.getRobotPosition(False))

            while True:
                count = count+1
                nao_images = self.video_service.getImageRemote(name_id)
                if nao_images is None:
                    print("Can't capture frames from Nao's camera")
                    break

                cv_images = self.to_cv_img(nao_images)
                # time.sleep(0.1)

                img = imutils.resize(cv_images, width=min(pixel, cv_images.shape[1]))

                marker = self.find_face(img, face_cascade)
                object_marker = self.find_object(img, obj_cascade)

                if len(object_marker) is not 1:
                    # object_marker = [[0, 0, 0, 0]]
                    print("-------")
                    print(marker)
                    x, y, theta = self.foot_step(marker, real_focal_length)
                else:
                    for (xA, yA, xB, yB) in object_marker:
                        ya_max = yA
                        yb_max = yB
                        half_length = (xB - xA) / 2
                        obj_pix_value = xA + half_length

                    pix_obj_height = yb_max - ya_max
                    if pix_obj_height == 0:
                        return 0, 0, 0
                    print("-------")
                    print(object_marker)
                    inches = self.distance_to_camera(KNOWN_PERSON_HEIGHT, real_focal_length, pix_obj_height)
                    print("object detected", "%.2f inches" % inches)
                    print("object detected", obj_pix_value)
                    # print("-------")

                    if inches > 40.0:
                        x, y, theta = self.foot_step(marker, real_focal_length)

                    elif 40 > inches > 0:
                        if (pixel/2) - 80 <= obj_pix_value:
                            x = 0
                            y = 0.3
                            theta = 0

                        elif obj_pix_value < (pixel/2) + 80:
                            x = 0
                            y = -0.3
                            theta = 0

                        else:
                            x, y, theta = self.foot_step(marker, real_focal_length)
                    else:
                        x = 0
                        y = 0
                        theta = 0

                frequency = 0

                self.motion.post.moveTo(x, y, theta, frequency)
                self.motion.waitUntilMoveIsFinished()
                #self.motion.setWalkTargetVelocity(0.0, 0.0, 0.0, 0.0)
                robotPositionAfterCommand = m.Pose2D(self.motion.getRobotPosition(False))

                # compute and print the robot motion
                robotMoveCommand = m.pose2DInverse(robotPositionBeforeCommand) * robotPositionAfterCommand
                print("The Robot Move Command: ", robotMoveCommand)
                print("-------")

                if count > 20:
                    # print(1)
                    self.headPitchYaw()
                    print("I'm tired, can I have a rest?")
                    self.tts.say("I'm tired \\pau=500\\ can I have a rest ")
                    self.motion.rest()
                    # ---------> Recording <---------
                    self.record.stopMicrophonesRecording()
                    print('Start recording...')
                    # tts.say("start recording...")
                    record_path = speech_record_file_path

                    self.record.startMicrophonesRecording(record_path, 'wav', 16000, (0, 0, 1, 0))
                    time.sleep(3)
                    self.record.stopMicrophonesRecording()
                    print('stop recording')
                    # tts.say("stop recording")

                    # ---------> Speech recognition <---------
                    with sr.WavFile("./src/speechRecord.wav") as source:
                        audio = r.record(source)
                    text = r.recognize_google(audio)
                    print(text)
                    if text == "no":
                        count = 0
                        self.motion.wakeUp()
                        continue
                    elif text == "yes":
                        self.tts.say("okay \\pau=500\\ I'll have a break\\pau=500\\ touch my front head if you want to play")
                        break
                    else:
                        self.tts.say("I did not get you\\pau=500\\ I'll have a break ")
                        break
                else:
                    continue

        except SystemExit as e:
            print("System error : {0}".format(e))
        except OSError as err:
            print("OS error: {0}".format(err))
        else:
            print("Unexpected error:", sys.exc_info()[0])
        finally:
            print("Unsubscribe the video!")
            self.video_service.unsubscribe(name_id)
            self.motion.rest()

        #self.stop()

    def videorecording(self):
        self.videoRecorderProxy.stopRecording()

        # This records a 640*480 MJPG video at 20 fps.
        # Note MJPG can't be recorded with a framerate lower than 3 fps.
        self.videoRecorderProxy.setResolution(2)
        self.videoRecorderProxy.setFrameRate(20)
        self.videoRecorderProxy.setVideoFormat("MJPG")
        self.videoRecorderProxy.startRecording("/home/nao/WORKSPACE/src/cameras", "record")

    def processvideo(self):
        self.videoRecorderProxy.stopRecording()

        print("[INFO] loading model...")
        net = cv2.dnn.readNetFromCaffe(prototxt, model)

        print("[INFO] starting video stream...")

        cap = cv2.VideoCapture('r/home/nao/WORKSPACE/src/record.avi')
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('/home/nao/WORKSPACE/src/out.avi', fourcc, 20.0, (640, 480))
        while cap.isOpened():
            ret, frame = cap.read()

            if ret is True:
                frame = imutils.resize(frame, width=640, height=480)

                (height, weight) = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(cv2.resize(frame, (640, 480)), 1.0,
                                             (640, 480), (104.0, 177.0, 123.0))

                net.setInput(blob)
                detections = net.forward()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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

                out.write(frame)
                key = cv2.waitKey(1) & 0xFF

            else:
                break
        out.release()
        cap.release()

    def communication(self):
        global audio, r
        audio = r = None
        r = sr.Recognizer()


        # start recoding video
        self.videorecording()

        # ---------> Robot introduction <---------
        intro = "Hi I am Nao robot, do you want to play with me?"
        print(intro)
        self.tts.setVolume(1.0)

        self.tts.say("Hi \\pau=1000\\I am Nao robot \\pau=1000\\ do you want to play with me ")
        # ---------> Recording <---------
        self.record.stopMicrophonesRecording()
        print('Start recording...')
        # tts.say("start recording...")
        record_path = speech_record_file_path

        self.record.startMicrophonesRecording(record_path, 'wav', 16000, (0, 0, 1, 0))
        time.sleep(3)
        self.record.stopMicrophonesRecording()
        print('stop recording')
        # tts.say("stop recording")

        # ---------> Speech recognition <---------
        with sr.WavFile("./src/speechRecord.wav") as source:
            audio = r.record(source)

        try:
            text = r.recognize_google(audio)
            print(text)

            if text == "yes":
                start = "let's start! by the way, touch my middle head if you want to stop!"
                print(start)
                self.tts.say("let's start \\pau=1000\\ by the way \\pau=500\\ touch my middle head if you want to stop ")
                self.walk()

            elif text == "no":
                print("Thank you bye, touch my front head if you want to play")
                self.tts.say("Thank you \\pau=500\\bye \\pau=1000\\ touch my front head if you want to play")

        except sr.UnknownValueError:
            print("Google speech Recognition could not understand")
            self.tts.say("Speech recognition failed \\pau=500\\ you can touch my front head and try again")

        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))
            self.tts.say("Speech recognition failed \\pau=500\\ you can touch my front head and try again")


def main(ip, port):
    """ Main entry point
    """
    # We need this broker to be able to construct
    # NAOqi modules and subscribe to other modules
    # The broker must stay alive until the program exists
    myBroker = ALBroker("myBroker",
                        "0.0.0.0",  # listen to anyone
                        0,  # find a free port and use it
                        ip,  # parent broker IP
                        port)  # parent broker port

    global walkTo
    walkTo = walkTo("walkTo")

    try:
        while True:
            time.sleep(1)
    except SystemExit as e:
        print("System error : {0}".format(e))
    except KeyboardInterrupt:
        print("")
        print("Interrupted by user, shutting down")
    finally:
        myBroker.shutdown()
        sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="127.0.0.1",
                        help="Robot ip address")
    parser.add_argument("--port", type=int, default=9559,
                        help="Robot port number")
    args = parser.parse_args()
    main(args.ip, args.port)
