# import numpy as np
# import cv2
# from video_capture import C_VIDEO_UNIT

# cap = cv2.VideoCapture('/home/morteza/Videos/traffic camera/1.mkv')

# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
# fgbg = cv2.BackgroundSubtractorKNN()#createBackgroundSubtractorMOG2()
# # fgbg = cv2.bgsegm.createBackgroundSubtractorMOG2()

# while(1):
#     ret, frame = cap.read()

#     fgmask = fgbg.apply(frame)
#     fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

#     cv2.imshow('frame',fgmask)
#     k = cv2.waitKey(30) & 0xff
#     if k == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()

import numpy as np
import cv2

vdo='VA3'
cap = cv2.VideoCapture('/home/morteza/Videos/traffic camera/'+vdo+'.mp4')#'c:/2.mkv')

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fgbg = cv2.createBackgroundSubtractorKNN()
stack_frames = None
MI_Mean = list()#np.zeros(30)
frame_number = 0
MI = np.zeros(10)
frames = list()
keyframes=list()

while(1):
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame,(600,600))
    if not ret:
        print("no frame to read")
        break

    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    # if frame_number == 0:
    #     stack_frames = np.zeros((10,fgmask.shape[0],fgmask.shape[1]),np.float)
        

    # def smoothing(MI):


    if frame_number>10: 
        # calculate mean of frames
        # output=np.mean(stack_frames,axis=0)
        MI_Mean.append(np.mean(MI))#[frame_number, np.mean(MI)]) #T1=5
        # gray = cv2.cvtColor(output.astype(int), cv2.COLOR_GRAY2BGR)
        # a = np.expand_dims(output.astype(np.int8), axis = 2)
        # a=np.concatenate((a,a,a),axis=2)
        # cv2.imshow('mean_bg_subtraction',a)#fgmask)
        # k = cv2.waitKey(30) & 0xff
        # if k == 27:
        #     break
        if frame_number % 30 == 0:
            #find local maxima
            # a=0
            keyfr_idx = np.argmax(MI_Mean)
            if keyfr_idx>len(frames):
                a=0
            keyframes.append(frames[keyfr_idx])
            # cv2.imwrite('/home/morteza/Videos/traffic camera/keyframes/'+vdo+'/seg'+str(len(keyframes)-1)+'.jpg',frames[keyfr_idx])
            frames = list()
            MI_Mean = list()
            if len(MI_Mean)>30:
                a=0
        


    elems = fgmask.shape[0]*fgmask.shape[1]    
    
    cv2.imshow("subtraction",fgmask)
    cv2.imshow("frame",frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    
    tmp = fgmask.reshape(elems)
    if np.sum(tmp>20)/np.float(elems)>0.8:
        continue
    # stack_frames[frame_number%10] = fgmask
    MI[frame_number%10] = np.sum(tmp>0)
    frames.append(frame)
    frame_number += 1
    


cap.release()
cv2.destroyAllWindows()