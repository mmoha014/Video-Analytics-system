# from Object_detection import C_DETECTION
# import matplotlib.pyplot as plt
# import cv2


# detection = C_DETECTION('LaneNet')
# image = cv2.imread('/media/mgharasu/+989127464877/project/lanenet-lane-detection-master/data/tusimple_test_image/0.jpg', cv2.IMREAD_COLOR)
# mask_image, image_vis, embedding_image, binary_seg_image = detection.Lane_Detection(image)

# plt.figure('mask_image')
# plt.imshow(mask_image[:, :, (2, 1, 0)])
# plt.figure('src_image')
# plt.imshow(image_vis[:, :, (2, 1, 0)])
# plt.figure('instance_image')
# plt.imshow(embedding_image[:, :, (2, 1, 0)])
# plt.figure('binary_image')
# plt.imshow(binary_seg_image[0] * 255, cmap='gray')
# plt.show()

from video_capture import C_VIDEO_UNIT
from Object_detection import C_DETECTION
from preprocessing import C_PREPROCESSING
from memory_profiler import profile
import time
import cv2
import numpy as np
from setting import tool_time1,tool_time2,tool_time3, tool_time4
# from cpu_memory_track import  monitor, monitor_wo_params



@profile
def main():
    
    video = C_VIDEO_UNIT('/media/mgharasu/+989127464877/project/lanenet-lane-detection-master/data/dataset2/Dataset by video recorder/Daytime/beltway')#'/media/mgharasu/+989127464877/project/lanenet-lane-detection-master/data/dataset3/video_example/05081544_0305/')   
    detection = C_DETECTION('LaneNet')
    # b=a    
    count = 0
    while True:
        count += 1
        ret, frame = video.get_frame()

        # frame = C_PREPROCESSING.resize_image(frame, (frame.shape[1]//2,frame.shape[0]//2))
        
        if not ret:
            break


        _,_, _,embedding_image = detection.Lane_Detection(frame)
        if count>20:            
            break
            
        cv2.imshow('output', np.array(embedding_image[0]*255, dtype=np.uint8))
        cv2.imshow('input',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    cv2.destroyAllWindows()
    
@profile
def main2():
    main()

if __name__ == "__main__":
    main2()
    
    # total_t0 = time.time()
    # vid_t0,vid_t1,det_t0,det_t1,get_frame_t,resiz_t,lanedet_t = main()
    # total_t1 = time.time()
    # print('video class load: %g, detection class load: %g'%((vid_t1-vid_t0),(det_t1-det_t0)))
    # print('get_frame percentiles: 0: %g, 25: %g, 50: %g, 75: %g, 100:%g '%(np.percentile(get_frame_t,0),np.percentile(get_frame_t,25),
    #                                      np.percentile(get_frame_t,50),np.percentile(get_frame_t,75),np.percentile(get_frame_t,100)))
    # print('get_frame percentiles: 0: %g, 25: %g, 50: %g, 75: %g, 100:%g '%(np.percentile(resiz_t,0),np.percentile(resiz_t,25),
    #                                             np.percentile(resiz_t,50),np.percentile(resiz_t,75),np.percentile(resiz_t,100)))
    # print('get_frame percentiles: 0: %g, 25: %g, 50: %g, 75: %g, 100:%g '%(np.percentile(lanedet_t,0),np.percentile(lanedet_t,25),
    #                                             np.percentile(lanedet_t,50),np.percentile(lanedet_t,75),np.percentile(lanedet_t,100)))
    # print('total execution: %g'%(total_t1-total_t0))
    # cpu, mem = monitor_wo_params(main)
    # print('cpu: ',np.average(cpu), ', mem: ',(np.average(mem)/1024/1024))
    
# ====================== implementation ==========================
# add function to just send info for connected components instead of sending all image
# using an index just output is send to the client
# ================================================== definition of this function should be done ======================
# compute postprocessing resource requirement in lane_detection function and consider to use this function in client or server. if it can be executed on client
# side, the implementation of the above function is not considered in project
# ===============================================  acccuracy =========
# changing the image size how effects on results 


#  while True:
#         count += 1
#         getfra_t0 = time.time()
#         ret, frame = video.get_frame()
#         getfra_t1 = time.time()
        
#         if not ret:
#             break
        
#         # preprocessing
#         # cpu, mem = monitor(C_PREPROCESSING.resize_image, args=(frame, (frame.shape[1]//2, frame.shape[0]//2)))
        
#         # print()
#         # frame = C_PREPROCESSING.resize_image(frame, (frame.shape[1]//2,frame.shape[0]//2))
#         _,_, _,embedding_image = detection.Lane_Detection(frame)
#         if count>20:
#             # cv2.imwrite('output'+str(frame.shape[1])+str(frame.shape[0])+'.png', frame)
#             # cv2.imwrite('output'+str(frame.shape[1])+str(frame.shape[0])+'_embimg.png', embedding_image)
#             break
#         # mask_image, image_vis, embedding_image, binary_seg_image = detection.Lane_Detection(frame)

#         # post processing
        
#         cv2.imshow('output', np.array(embedding_image[0]*255, dtype=np.uint8))
#         cv2.imshow('input',frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break
#     cv2.destroyAllWindows()