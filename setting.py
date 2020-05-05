"""======================================================
Files
========================================================="""
FOLDER = "2"#"3_2014-09-17_16-30-06"#"MOT17-05-FRCNN"#"MOT17-11-SDP" #"ADL-Rundle-6"#Venice-2"#TUD-Campus"#PETS09-S2L1"#KITTI-17"#ETH-Sunnyday"#ETH-Pedcross2"#ETH-Bahnhof"#ADL-Rundle-8"#"TUD-Stadtmitte" TUD-Stadtmitte
INPUT_VIDEO_SNOURCE = "/home/mgharasu/Videos/traffic camera/"+FOLDER+"/"#"/media/mgharasu/+989127464877/DataSets/Virginia Traffic Videos/CAM822/"+FOLDER+"/"#"/home/mgharasu/Documents/Dataset/MOT17/train/"+FOLDER+"/img1/"#/media/morteza/+989127464877/DataSets/VOT_Dataset/VOT2013/"+folder+"/color"#video.avi"#"project_video.mp4"#                      
                      #"/media/morteza/+989127464877/DataSets/2DMOT2015/train/"+FOLDER+"/img1/"
OUTPUT_VIDEO_WRITE = "/home/mgharasu/Documents/Dataset/MOT17/myTracker_output/"+FOLDER+".avi"#"/media/mgharasu/+989127464877/DataSets/Virginia Traffic Videos/CAM822/"+FOLDER+"/output.avi"#"/media/morteza/+989127464877/DataSets/2DMOT2015/output_YOLOCSRT/output_"+FOLDER+".avi"                     
FILE_ADDRESS_DEEP_GROUNDTHRUTH = "/home/mgharasu/Videos/traffic camera/gt"+FOLDER+"_960.txt"#"/home/mgharasu/Videos/traffic camera/gt.txt"#"/media/mgharasu/+989127464877/DataSets/Virginia Traffic Videos/CAM822/"+FOLDER+"/gt.txt"#"/home/mgharasu/Documents/Dataset/MOT17/train/"+FOLDER+"/gt/gt.txt"#
                                #"/media/morteza/+989127464877/DataSets/2DMOT2015/train/"+FOLDER+"/gt/gt.txt"
FILE_ADDRESS_HOG_VEHICLE = ""
FILE_ADDRESS_HOG_PEDESTRIAN = ""
FILE_ADDRESS_HAAR_VEHICLE = "haar_cascade_pretrianed_classifiers/cars.xml"
FILE_ADDRESS_HAAR_PEDESTRIAN = "haar_cascade_pretrianed_classifiers/haarcascade_fullbody.xml"
FILE_ADDRESS_DEEP_YOLO_WEIGHT = "Deep_wieghts_configs/yolov3.weights"
FILE_ADDRESS_DEEP_YOLO_CONFIG = "Deep_wieghts_configs/yolov3.cfg"
FILE_ADDRESS_DEEP_YOLO_TINY_WEIGHT = "Deep_wieghts_configs/yolov3-tiny.weights"
FILE_ADDRESS_DEEP_YOLO_TINY_CONFIG = "Deep_wieghts_configs/yolov3-tiny.cfg"
# FILE_ADDRESS_DEEP_ACCIDENT_WEIGHT = "../demo_model/"

""" ===========================================
Tracker type = 'BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT', 'Kalman_Filter'
==============================================="""
TRACKER_TYPE = "CSRT"#"Kalman_Filter"#"Kalman_Filter""CSRT"#
LARGEST_OBJECT = True
UPDATE_TRACKER = 10 #  after processing this number of frames, apply object detection

"""============================================================
Detection method
"
different methods:
                    HOG_Vehicle
                    HOG_Pedestrian
                    Haar_Vehicle
                    Haar_Pedestrian
                    Deep_Yolo  => image can not be GrayScale   
                    Deep_Yolo_Tiny               
==============================================================="""
DETECTION_METHOD = "Deep_Yolo"#"Deep_Yolo_Tiny"#"HOG_Pedestrian"#

# HOG pedestrian parameters
VEHICLE_HOG_NON_MAX_SUPPRESSION_OVERLAP_THRESHOLD = None
VEHICLE_HOG_WIN_STRIDE = None
VEHICLE_HOG_WIN_PADDING = None
VEHICLE_HOG_SCALE_FACTOR = None

PEDESTRIAN_HOG_NON_MAX_SUPPRESSION_OVERLAP_THRESHOLD = 0.65
PEDESTRIAN_HOG_WIN_STRIDE = (4, 4)
PEDESTRIAN_HOG_WIN_PADDING = (8, 8)
PEDESTRIAN_HOG_SCALE_FACTOR = 1.01

#Haar cascade parameters
VEHICLE_HAAR_SCALE_FACTOR = 1.2
VEHICLE_HAAR_MIN_NEIGHBORS = 1

PEDESTRIAN_HAAR_SCALE_FACTOR = None
PEDESTRIAN_HAAR_MIN_NEIGHBORS = None

# Deep Yolo parameters
__deep_classes={"person":0, "bicycle":1, "car":2, "motorcycle":3, "airplane":4, "bus":5, "train":6, "truck":7, "boat":8, "traffic light":9, "fire hydrant":10, "stop sign":11, "parking meter":12,
"bench":13, "bird":14, "cat":15, "dog":16, "horse":17, "sheep":18, "cow":19, "elephant":20, "bear":21, "zebra":22, "giraffe":23, "backpack":24, "umbrella":25, "handbag":26,"tie":27,
"suitcase":28, "frisbee":29, "skis":30, "snowboard":31, "sports ball":32, "kite":33, "baseball bat":34, "baseball glove":35, "skateboard":36, "surfboard":37, "tennis racket":38, 
"bottle":39, "wine glass":40, "cup":41, "fork":42, "knife":43, "spoon":44, "bowl":45, "banana":46, "apple":47, "sandwich":48,  "orange":49, "broccoli":50, "carrot":51, "hot dog":52,
"pizza":53, "donut":54, "cake":55, "chair":56, "couch":57, "potted plant":58, "bed":59, "dining table":60, "toilet":61, "tv=":62, "laptop":63, "mouse":64, "remote":65, "keyboard":66,
"cell phone":67, "microwave":68, "oven":69, "toaster":70, "sink":71, "refrigerator":72, "book":73, "clock":74, "vase":75, "scissors":76, "teddy bear":77, "hair drier":78, "toothbrush":79,}

with open("/home/mgharasu/Documents/Purified codes/Deep_wieghts_configs/yolov3.txt", 'r') as f:
    classes = [line.strip() for line in f.readlines()] 

DETECTION_CLASS = [__deep_classes['car'], __deep_classes['truck']]#[__deep_classes['person'],__deep_classes['car'], __deep_classes['truck']]
# tracker age in loss
max_age = 4
GF1 = 0.0
#======================= version 3 global variables ==========================
# detection = None
# # MOT = None
# from MOT_File_Generator import C_MOT_OUTPUT_GENERATER # version 2
# MOT = C_MOT_OUTPUT_GENERATER('./logs/'+FOLDER+'.txt')# version 2 
# groundtruth_box = None
# targetSize = 1.0
# frame_rate = 14
# F1 = []

tool_time1=[]
tool_time2=[]
tool_time3=[]
tool_time4=[]
