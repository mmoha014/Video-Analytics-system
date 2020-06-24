from video_capture import C_VIDEO_UNIT
import numpy as np
import  cv2


video = C_VIDEO_UNIT('/home/morteza/Videos/traffic camera/3/')

start_frame, end_frame = [0,29]
step__frames_read = 30
frame_rate = 30
shared_cont = False
# while True:
while start_frame<=1810:
    end_frame = start_frame+frame_rate-1
    while start_frame <= end_frame:
        # ret, frame = video.get_frame()
        shared_cont, frame = video.get_frame_position(start_frame, shared_cont)
        cv2.imwrite('/home/morteza/Documents/features/DE_KFE-master/processed_frames/3/frame'+str(start_frame)+'.jpg',frame)
        start_frame = start_frame + step__frames_read
    