import numpy as np
import cv2
import copy
import time
from video_capture import C_VIDEO_UNIT

video = C_VIDEO_UNIT('/home/mgharasu/Documents/Dataset/MOT17/train/MOT17-11-DPM/img1/', video_write=True ,output_video='/home/mgharasu/Documents/ML/tensorflow-myself/mall.avi')
while(True):
    ret, frame = video.get_frame()
    if not ret:
        break  
    video.write_output_video(frame)

# ret, old_frame = cap.read()
# old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
# sub_bg_frames1 = np.zeros(old_frame.shape)
# sub_bg_frames1_1 = np.zeros(old_frame.shape)
# one_frame = np.ones(old_frame.shape)*255
# count = 0
# while ret:
#     ret, frame = cap.read()
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     if not ret or count>60:
#         break
    
#     if count>30:
#         sub_bg_frames1_1 += np.divide(np.subtract(frame, old_frame), one_frame)    
#     else:
#         sub_bg_frames1 += np.divide(np.subtract(frame, old_frame), one_frame)

#     old_frame = copy.deepcopy(frame)
    
#     # cv2.imshow('original image', sub_bg_frames1)
    
#     # if cv2.waitKey(1) & 0xFF == ord('q'): 
#     #     break
#     # print(count)
#     count = count + 1
    
    

# a=0
# cap = cv2.VideoCapture('/home/mgharasu/Documents/Dataset/accident/Anticipating-Accidents-master/dataset/videos/training/positive/000002.mp4')
# ret, old_frame = cap.read()
# old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
# sub_bg_frames2 = np.zeros(old_frame.shape)
# sub_bg_frames2_1 = np.zeros(old_frame.shape)
# one_frame = np.ones(old_frame.shape)*255
# count = 0
# while ret:
#     ret, frame = cap.read()
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     if not ret or count>60:
#         break
    
#     if count>30:
#         sub_bg_frames2_1 += np.divide(np.subtract(frame, old_frame), one_frame)    
#     else:
#         sub_bg_frames2 += np.divide(np.subtract(frame, old_frame), one_frame)

#     old_frame = copy.deepcopy(frame)
        
#     count = count + 1

# cap = cv2.VideoCapture('/home/mgharasu/Documents/Dataset/accident/Anticipating-Accidents-master/dataset/videos/training/positive/000003.mp4')
# ret, old_frame = cap.read()
# old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
# sub_bg_frames3 = np.zeros(old_frame.shape)
# sub_bg_frames3_1 = np.zeros(old_frame.shape)
# one_frame = np.ones(old_frame.shape)*255
# count = 0
# while ret:
#     ret, frame = cap.read()
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     if not ret or count>60:
#         break
    
#     if count>30:
#         sub_bg_frames3_1 += np.divide(np.subtract(frame, old_frame), one_frame)    
#     else:
#         sub_bg_frames3 += np.divide(np.subtract(frame, old_frame), one_frame)

#     old_frame = copy.deepcopy(frame)
        
#     count = count + 1

# a=0