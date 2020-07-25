# Jan, 2020, last week of the month
# plot for effect of the object size and number of objects on detection, tracking. in addition,  cpu and gpu usage


############################ Detection Observation ##########################
"""
from video_capture import C_VIDEO_UNIT
# from multitracking import box_iou2, correct_position
from preprocessing import C_PREPROCESSING
from Object_detection import C_DETECTION
from tracking import C_TRACKER
from utils import *
from MOT_File_Generator import C_MOT_OUTPUT_GENERATER # version 2
from Scheduler import C_SCHEDULER
from setting import *
import cv2
from sys import stdout
import copy
import matplotlib.pyplot as plt
from mylinear_assignment import linear_assignment
from psutil import virtual_memory
# from Segment_process import Process_segment
from cpu_memory_track import monitor
# import imutils, psutil
import pylab as pl
import multiprocessing as mp
import time
import numpy as np

video = C_VIDEO_UNIT(INPUT_VIDEO_SNOURCE, frame_rate=14)
detection = C_DETECTION(DETECTION_METHOD)

frame_number = 0
while True:
    ret, frame = video.get_frame()
    frame = cv2.resize(frame, (int(frame.shape[1]*0.3),int(frame.shape[0]*0.3)))
    if not ret:
        break
    if frame_number==0:
        vw = cv2.VideoWriter("detection_imagesize_observation.avi", cv2.VideoWriter_fourcc('M','J','P','G'),5,(frame.shape[1],frame.shape[0]))

    # Detection
    t1 = time.time()
    predicted_box,frame = detection.Detection_BoundingBox(frame)
    t2 = time.time()
    tables = cv2.putText(frame, str(t2-t1),(70,30),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))
    for box in predicted_box:
        frame = cv2.rectangle(frame,(box[0], box[1]),(box[0]+box[2],box[1]+box[3]), (255,0,0), 2)

    cv2.imshow("output", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break #esc to quit
    vw.write(frame)
    frame_number+=1

vw.release()
cv2.destroyAllWindows() 

"""
#################################### Tracking observation #################################

from video_capture import C_VIDEO_UNIT
# from multitracking import box_iou2, correct_position
from preprocessing import C_PREPROCESSING
from Object_detection import C_DETECTION
from tracking import C_TRACKER
from utils import *
from MOT_File_Generator import C_MOT_OUTPUT_GENERATER # version 2
from Scheduler import C_SCHEDULER
from setting import *
import cv2
from sys import stdout
import copy
import matplotlib.pyplot as plt
from mylinear_assignment import linear_assignment
from psutil import virtual_memory
# from Segment_process import Process_segment
from cpu_memory_track import monitor
# import imutils, psutil
import pylab as pl
import multiprocessing as mp
import time
import numpy as np

fp = open("tracker_statistics.txt","w")
video = C_VIDEO_UNIT(INPUT_VIDEO_SNOURCE, frame_rate=14)
ret, frame = video.get_frame()
detection = C_DETECTION(DETECTION_METHOD)
# tracker = cv2.TrackerCSRT_create()#cv2.Tracker_create('CSRT')
# bbox = cv2.selectROI(frame, False)
# ok = tracker.init(frame, bbox)

frame_number = 0
while True:
    ret, frame = video.get_frame()
    # frame = cv2.resize(frame, (int(frame.shape[1]*0.3),int(frame.shape[0]*0.3)))
    if not ret:
        break
    if (cv2.waitKey(1) & 0xFF == ord('q')) or frame_number == 0 :
        # vw = cv2.VideoWriter("1_time_observation.avi", cv2.VideoWriter_fourcc('M','J','P','G'),5,(frame.shape[1],frame.shape[0]))
        fp.write("ROI selection\n")
        tracker = None
        multiTracker = cv2.MultiTracker_create()#cv2.Tracker_create('CSRT')
        bboxes = []
        while True:
            bbox = cv2.selectROI(frame, False)
            bboxes.append(bbox)
            k = cv2.waitKey(0) & 0xFF
            if (k == 113):  # q is pressed
                break
        multiTracker = cv2.MultiTracker_create()
        for bb in bboxes[:-1]:
            multiTracker.add(cv2.TrackerCSRT_create(), frame, bb)
        # ok = tracker.init(frame, bbox) 


   
    # Detection
    timer = cv2.getTickCount()
    t1 = time.time()
    # predicted_box,frame = detection.Detection_BoundingBox(frame)
    ok, bboxes = multiTracker.update(frame)
    t2 = time.time()
    areas = []
    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
    for bbox in bboxes:
        areas.append(bbox[2]*bbox[3])
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
    
    fp.write(str(t2-t1)+", "+str(np.mean(areas))+", "+str(len(bboxes))+", "+str(fps)+"\n")
    tables = cv2.putText(frame, str(t2-t1),(200,80),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0),2)
    # for box in predicted_box:
    #     frame = cv2.rectangle(frame,(box[0], box[1]),(box[0]+box[2],box[1]+box[3]), (255,0,0), 2)
    
    # if ok:
    #     # Tracking success
        
        
    # else :
    #     # Tracking failure
    #     cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

    # Display FPS on frame
    # cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)

    cv2.imshow("Tracking", frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break #esc to quit
    # vw.write(frame)
    
    frame_number+=1

# vw.release()
fp.close()
cv2.destroyAllWindows() 


############################### Object detection- model comparison - side-by-side #################
"""
from Object_detection import C_DETECTION
from setting import *
import cv2
import os
import numpy as np
import copy
from imutils import paths 
import re 

detection1 = C_DETECTION("Deep_Yolo")#"HOG_Pedestrian"#
detection2 = C_DETECTION("Deep_Yolo_Tiny")

address = "/home/mgharasu/Videos/traffic camera/objectSize_yolo&yoloTiny/"
files = sorted(os.listdir(address))

#directory address
numbers = re.compile(r'(\d+)')

def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

files = sorted(paths.list_images(address), key = numericalSort)

for file in files:
    img1 = cv2.imread(file)
    img1 = cv2.resize(img1, (int(img1.shape[1]*0.3),int(img1.shape[0]*0.3)))
    img2 = copy.deepcopy(img1)

    pb1,img1 = detection1.Detection_BoundingBox(img1)
    img1 = cv2.putText(img1, "YOLO", (30,350), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255))
    for box in pb1:
        img1 = cv2.rectangle(img1,(box[0], box[1]),(box[0]+box[2],box[1]+box[3]), (255,0,0), 2)

    pb2,img2 = detection2.Detection_BoundingBox(img2)
    img2 = cv2.putText(img2, "YOLO_TINY", (30,350), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255))
    for box in pb2:
        img2 = cv2.rectangle(img2,(box[0], box[1]),(box[0]+box[2],box[1]+box[3]), (255,0,0), 2)

    tmp_img = np.zeros((img1.shape[0], img1.shape[1]*2,3), dtype=np.uint8)
    tmp_img[:,:img1.shape[1]] = img1
    tmp_img[:,img1.shape[1]:] = img2

    cv2.imshow("output", tmp_img)
    cv2.waitKey(0)
cv2.destroyAllWindows()
"""