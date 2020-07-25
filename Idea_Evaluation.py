# import cv2
# import detectron2
# import numpy as np
# from detectron2 import model_zoo
# from detectron2.engine import DefaultPredictor
# from detectron2.config import get_cfg
# from detectron2.utils.visualizer import Visualizer
# from detectron2.data import MetadataCatalog

# # wget http://images.cocodataset.org/val2017/000000439715.jpg -O input.jpg
# im = cv2.imread("/home/mgharasu/Videos/traffic camera/keyframes/1/seg10.jpg")
# cv2.imshow("main image",im)
# im = cv2.resize(im,(960,960))
# cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
# predictor = DefaultPredictor(cfg)
# outputs = predictor(im)

# # We can use `Visualizer` to draw the predictions on the image.
# v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
# out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# cv2.imshow("output",out.get_image()[:, :, ::-1])

from video_capture import C_VIDEO_UNIT
from preprocessing import C_PREPROCESSING
from Object_detection import C_DETECTION
from tracking import C_TRACKER
from setting import *
from utils import *
import cv2
from Detection_Models.YOLO import C_DETECTION_YOLO as YOLO
from MOT_File_Generator import C_MOT_OUTPUT_GENERATER

# detection = []
# detection.append(C_DETECTION("faster_rcnn_X_101_32x8d_FPN_3x"))
# detection.append(C_DETECTION("faster_rcnn_R_101_DC5_3x"))
# detection.append(C_DETECTION("retinanet_R_101_FPN_3x"))
# detection.append(C_DETECTION("faster_rcnn_R_50_FPN_3x"))

detection = None
detection_models = {'faster_rcnn_X_101_32x8d_FPN_3x':C_DETECTION("faster_rcnn_X_101_32x8d_FPN_3x"), 'faster_rcnn_R_101_DC5_3x':C_DETECTION('faster_rcnn_R_101_DC5_3x'), 'retinanet_R_101_FPN_3x':C_DETECTION('retinanet_R_101_FPN_3x'), 'faster_rcnn_R_50_FPN_3x':C_DETECTION('faster_rcnn_R_50_FPN_3x')}

def Set_configuration(C):
    # global targetSize, detection
    #C [i,  j,   k]
    # size, rate, model 
    # print(C)
    keyframe = int(C[0])
    frame_size = int(C[1])#knobs['frame_size'][C[0]]
    frame_rate = int(C[2])#knobs['frame_rate'][C[1]]    
    # model = C[3]#knobs['detection_model'][C[2]]
    # tracker.switch_Tracker()        
    # detection[0].Switch_detection(model)
    model = detection_models[C[3][:-1]]

    return keyframe,frame_size, frame_rate, model
# 
# @track
def main():
    # groundtruth_box  = C_PREPROCESSING.VOT2013_read_groundtruth_file(FILE_ADDRESS_DEEP_GROUNDTHRUTH)
    
    #['faster_rcnn_X_101_32x8d_FPN_3x','faster_rcnn_R_101_DC5_3x','retinanet_R_101_FPN_3x','faster_rcnn_R_50_FPN_3x']
    # detection = C_DETECTION(DETECTION_METHOD)
    vdo = '1'

    video = C_VIDEO_UNIT('/home/mgharasu/Videos/traffic camera/1/'+vdo+'/')#"/media/mgharasu/+989127464877/DataSets/Virginia Traffic Videos/VB/video1")
    MOT =  C_MOT_OUTPUT_GENERATER("/home/mgharasu/Videos/traffic camera/keyframes/"+vdo+"/gt.txt")
    # groundtruth_box  = C_PREPROCESSING.MOT2017_read_groudntruth_file('/home/mgharasu/Videos/traffic camera/keyframes/gt_kyframe_'+vdo+'.txt', video.get_number_of_frames())
    config_winner_segm  = list()
    fp = open('/home/mgharasu/Videos/traffic camera/keyframes/winner_configs_1.txt','r')
    for line in fp.readlines():
        if line is not '\n':
            tmp = line.split(',')
            config_winner_segm.append(tmp)
    

    acc_per_frame = []
    # ret, frame = video.get_frame()
    # image_size = frame.shape    
    frame_number = 0
    time_sum = 0
    segm = 0

    while True:
        ret, frame = video.get_frame()

        frame_number += 1
        if not ret:
            break
        
        kyframe,frSize,frRate,detection = Set_configuration(config_winner_segm[segm])
        
        if True:#kyframe == frame_number or frame_number == 0:
            frame = cv2.resize(frame,(frSize,frSize))
            if segm == 128:
                a=0
            
            # Detection
            predicted_box,frame = detection.Detection_BoundingBox(frame)
            boxes = []
            for i, b in enumerate(predicted_box):
                    boxes.append([kyframe, b[0],b[1],b[2],b[3]])
            if len(boxes)>0:
                MOT.write(kyframe,boxes)
            else:
                MOT.write(kyframe,[[kyframe,0,0,0,0]])
            print(boxes)
            frame = draw_box(predicted_box, frame)
            cv2.imshow("output", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break #esc to quit
            # if frame_number % 30 == 0:
            segm+=1
            print(segm)
           
    
    cv2.destroyAllWindows()
    MOT.close()
    
    
if __name__ == "__main__":  
    # t1=time.time() 
    main()
    # t2=time.time()
    # print('pipeline time:', t2-t1)
    

    
