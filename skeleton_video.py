# @Author: ASHISH SASMAL <ashish>
# @Date:   20-10-2020
# @Last modified by:   ashish
# @Last modified time: 20-10-2020

import cv2
import numpy as np
import time

proto = "Models/pose_deploy_linevec_faster_4_stages.prototxt"
weights= "Models/pose_iter_160000.caffemodel"

net = cv2.dnn.readNetFromCaffe(proto, weights)

net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
print("Using CPU device")

wid = 368
height=368

gt = cv2.VideoCapture("sample2.mp4")
hasFrame, frame = gt.read()
vid_writer1 = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame.shape[1],frame.shape[0]))
vid_writer2 = cv2.VideoWriter('output2.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame.shape[1],frame.shape[0]))

while cv2.waitKey(1) < 0:
    image = gt.read()[1]
    image_copy = np.copy(image)
    image_wid = image.shape[1]
    image_height = image.shape[0]
    thresh = np.zeros((frame.shape[0],frame.shape[1],1), np.uint8)
    thresh = cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)
    blob = cv2.dnn.blobFromImage(image, 1.0/255, (wid,height), (0,0,0), swapRB = False, crop = False)

    net.setInput(blob)
    POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]

    preds = net.forward()

    H = preds.shape[2]
    W = preds.shape[3]
    # Empty list to store the detected keypoints
    points = []
    for i in range(15):
        probMap = preds[0, i, :, :]

        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        x = (image_wid * point[0]) / W
        y = (image_height * point[1]) / H

        if prob >0.1 :
            # cv2.circle(image_copy, (int(x), int(y)), 3, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            # cv2.putText(image_copy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, lineType=cv2.LINE_AA)

            points.append((int(x), int(y)))
        else :
            points.append(None)

    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv2.line(image, points[partA], points[partB], (199,99,0), 2)
            cv2.circle(image, points[partA], 4, (17,199,0), thickness=-1, lineType=cv2.FILLED)

            cv2.line(thresh, points[partA], points[partB], (199,99,0), 2)
            cv2.circle(thresh, points[partA], 4, (17,199,0), thickness=-1, lineType=cv2.FILLED)

    cv2.imshow('Output-Skeleton', image)
    cv2.imshow('Output-Skeleton2', thresh)
    vid_writer1.write(image)
    vid_writer2.write(thresh)

gt.release()
cv2.destroyAllWindows()
