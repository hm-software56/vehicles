import cv2
import os
import time
import argparse
import imutils
import time
import os
import glob
import math
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from sort import *
from time import gmtime, strftime, localtime

root = os.path.dirname(os.path.abspath(__file__))


class VideoCamera(object):
    def __init__(self):
        self.video = ''
        self.tracker = Sort()
        self.memory = {}

        self.line1 = [(0, 500), (1350, 500)]
        self.line2 = [(120, 400), (1120, 1080)]
        self.counter1 = 0
        self.counter2 = 0
        self.text_speed_to_hours = 0

        self.confidence = 0.35
        self.threshold = 0.25
        # initialize a list of colors to represent each possible class label
        np.random.seed(42)
        self.COLORS = np.random.randint(0, 255, size=(200, 3),
                                        dtype="uint8")

        (self.W, self.H) = (None, None)
        self.writer = None
        self.total = 0
        # derive the paths to the YOLO weights and model configuration
        self.weightsPath = os.path.sep.join(['yolococo', "yolov3.weights"])
        self.configPath = os.path.sep.join(['yolococo', "yolov3.cfg"])
        # load our YOLO object detector trained on COCO dataset (80 classes)
        # and determine only the *output* layer names that we need from YOLO
        # print("[INFO] loading YOLO from disk...")
        self.net = cv2.dnn.readNetFromDarknet(self.configPath, self.weightsPath)
        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def __del__(self):
        try:
            self.video.release()
        except:
            print('No Camera')

    # Return true if line segments AB and CD intersect
    def intersect(self, A, B, C, D):
        return self.ccw(A, C, D) != self.ccw(B, C, D) and self.ccw(A, B, C) != self.ccw(A, B, D)

    def ccw(self, A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def adjust_gamma(self, image, gamma=1.0):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")

        # apply gamma correction using the lookup table
        return cv2.LUT(image, table)

    # font laos
    def addlaotext(self, image, x, y, label, tex_color, font_size):
        cv2_im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im_rgb)
        draw = ImageDraw.Draw(pil_im)
        font = ImageFont.truetype('Phetsarath_OT.ttf', font_size)
        draw.text((x, y), label, font=font, fill=tex_color)
        return pil_im

    def get_frame(self):
        self.total = self.total + 1
        # read the next frame from the file
        (grabbed, frame) = self.video.read()

        # if the frame dimensions are empty, grab them
        if self.W is None or self.H is None:
            (self.H, self.W) = frame.shape[:2]

        h_line = self.H - 150
        w_line = self.W
        self.line1 = [(0, h_line), (w_line, h_line)]

        frame = self.adjust_gamma(frame, gamma=1.5)
        # construct a blob from the input frame and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes
        # and associated probabilities
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (256, 256),
                                     swapRB=True, crop=False)
        self.net.setInput(blob)
        start = time.time()
        layerOutputs = self.net.forward(self.ln)
        end = time.time()
        # initialize our lists of detected bounding boxes, confidences,
        # and class IDs, respectively
        boxes = []
        center = []
        confidences = []
        classIDs = []

        # loop over each of the layer outputs
        #print(layerOutputs)
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability)
                # of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > self.confidence:
                    if classID in [1,2,3,5,6,7]: # [1,2,3,5,6,7] number of labels want to detecttion

                        # scale the bounding box coordinates back relative to
                        # the size of the image, keeping in mind that YOLO
                        # actually returns the center (x, y)-coordinates of
                        # the bounding box followed by the boxes' width and
                        # height
                        box = detection[0:4] * np.array([self.W, self.H, self.W, self.H])
                        (centerX, centerY, width, height) = box.astype("int")

                        # use the center (x, y)-coordinates to derive the top
                        # and and left corner of the bounding box
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))

                        # update our list of bounding box coordinates,
                        # confidences, and class IDs
                        center.append(int(centerY))
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)
                        #print(classID)

        # apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, self.threshold)
        #print("idxs", idxs)
        # print("boxes", boxes[i][0])
        # print("boxes", boxes[i][1])

        dets = []
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                dets.append([x, y, x + w, y + h, confidences[i]])
                # print(confidences[i])
                # print(center[i])
        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        dets = np.asarray(dets)
        tracks = self.tracker.update(dets)

        boxes = []
        indexIDs = []
        c = []

        previous = self.memory.copy()
        # print("centerx",centerX)
        #  print("centery",centerY)
        memory = {}
        for track in tracks:
            boxes.append([track[0], track[1], track[2], track[3]])
            indexIDs.append(int(track[4]))
            memory[indexIDs[-1]] = boxes[-1]
        self.memory = memory
        if len(boxes) > 0:
            i = int(0)
            for box in boxes:
                # extract the bounding box coordinates
                (x, y) = (int(box[0]), int(box[1]))
                (w, h) = (int(box[2]), int(box[3]))

                # draw a bounding box rectangle and label on the image
                # color = [int(c) for c in COLORS[classIDs[i]]]
                # cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                color = [int(c) for c in self.COLORS[indexIDs[i] % len(self.COLORS)]]
                cv2.rectangle(frame, (x, y), (w, h), color, 2)

                if indexIDs[i] in previous:
                    previous_box = previous[indexIDs[i]]
                    (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                    (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                    p0 = (int(x + (w - x) / 2), int(y + (h - y) / 2))
                    p1 = (int(x2 + (w2 - x2) / 2), int(y2 + (h2 - y2) / 2))
                    cv2.line(frame, p0, p1, color, 3)

                    # Speed Calculation
                    y_pix_dist = int(y + (h - y) / 2) - int(y2 + (h2 - y2) / 2)
                    text_y = "{} y".format(y_pix_dist)
                    x_pix_dist = int(x + (w - x) / 2) - int(x2 + (w2 - x2) / 2)
                    text_x = "{} x".format(x_pix_dist)
                    # cv2.putText(frame, text_y, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 4)
                    # cv2.putText(frame, text_x, (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 4)
                    final_pix_dist = math.sqrt((y_pix_dist * y_pix_dist) + (x_pix_dist * x_pix_dist))
                    speed = np.round(1.5 * y_pix_dist, 2)
                    text_speed = "{} ກິໂລເມັດ/ຊົ່ວໂມງ".format(speed)
                    # cv2.putText(frame, text_speed, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                    label = text_speed
                    tex_color = 'red'
                    font_size = 16
                    img = self.addlaotext(frame, x, y - 5, label, tex_color, font_size)
                    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

                    if self.intersect(p0, p1, self.line1[0], self.line1[1]):
                        self.counter1 += 1
                        self.text_speed_to_hours = speed
                    # if self.intersect(p0, p1, self.line2[0], self.line2[1]):
                    #                counter2 += 1
                #print(classIDs[i])
                # text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                # text = "{}".format(indexIDs[i])
                # cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                i += 1

        # draw line
        cv2.line(frame, self.line1[0], self.line1[1], (0, 255, 255), 4)
        # cv2.line(frame, self.line2[0], self.line2[1], (255, 0, 255), 2)

        # note_text = "NOTE: Vehicle speeds are calibrated only at yellow line. speed of cars are more stable."
        # cv2.putText(frame, note_text, (50, 110), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)
        # draw counter
        counter_text = "ນັບຈໍານວນລົດ:{}".format(self.counter1) + " ຄັນ\n" + "ແລ່ນດ້ວຍຄວາມໄວ: " + str(
            self.text_speed_to_hours) + " ກິໂລເມັດ/ຊົ່ວໂມງ"
        # cv2.putText(frame, counter_text, (50, 650), cv2.FONT_HERSHEY_DUPLEX, 4.0, (0, 0, 255), 7)
        label = counter_text
        tex_color = 'red'
        font_size = 30
        img = self.addlaotext(frame, 50, h_line + 10, label, tex_color, font_size)
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        #    cv2.putText(frame, "ctr2",str(counter2), (100,400), cv2.FONT_HERSHEY_DUPLEX, 5.0, (255, 0, 255), 10)
        # counter += 1

        # saves image file
        # +cv2.imwrite("output/frame-{}.png".format(frameIndex), frame)
        if self.writer is None:
            # initialize our video writer
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            filename = str(strftime("%Y-%m-%d %H_%M_%S", localtime())) + '.avi'
            self.writer = cv2.VideoWriter('static/video/' + filename, fourcc, 15,
                                          (frame.shape[1], frame.shape[0]), True)

            # some information on processing single frame
            if self.total > 0:
                elap = (end - start)
                print("[INFO] single frame took {:.4f} seconds".format(elap))
                print("[INFO] estimated total time to finish: {:.4f}".format(
                    elap * self.total))

        # write the output frame to disk
        self.writer.write(frame)

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    def get_frame_redo(self):
        self.writer = None
        ret, frame = self.video.read()
        time.sleep(0.05)
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
