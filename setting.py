"""======================================================
Files
========================================================="""
FOLDER = "MOT17-13-FRCNN" #"ADL-Rundle-6"#Venice-2"#TUD-Campus"#PETS09-S2L1"#KITTI-17"#ETH-Sunnyday"#ETH-Pedcross2"#ETH-Bahnhof"#ADL-Rundle-8"#"TUD-Stadtmitte" TUD-Stadtmitte
INPUT_VIDEO_SNOURCE = "/media/morteza/+989127464877/DataSets/MOT17Det/train/"+FOLDER+"/img1/"#/media/morteza/+989127464877/DataSets/VOT_Dataset/VOT2013/"+folder+"/color"#video.avi"#"project_video.mp4"#
                      #"/media/morteza/+989127464877/DataSets/2DMOT2015/train/"+FOLDER+"/img1/"
OUTPUT_VIDEO_WRITE = "/media/morteza/+989127464877/DataSets/MOT17Det/myTracker_output/"+FOLDER+".avi"#"/media/morteza/+989127464877/DataSets/2DMOT2015/output_YOLOCSRT/output_"+FOLDER+".avi"                     
FILE_ADDRESS_DEEP_GROUNDTHRUTH = "/media/morteza/+989127464877/DataSets/MOT17Det/train/"+FOLDER+"/gt/gt.txt"
                                #"/media/morteza/+989127464877/DataSets/2DMOT2015/train/"+FOLDER+"/gt/gt.txt"
FILE_ADDRESS_HOG_VEHICLE = ""
FILE_ADDRESS_HOG_PEDESTRIAN = ""
FILE_ADDRESS_HAAR_VEHICLE = "haar_cascade_pretrianed_classifiers/cars.xml"
FILE_ADDRESS_HAAR_PEDESTRIAN = "haar_cascade_pretrianed_classifiers/haarcascade_fullbody.xml"
FILE_ADDRESS_DEEP_YOLO_WEIGHT = "Deep_wieghts_configs/yolov3.weights"
FILE_ADDRESS_DEEP_YOLO_CONFIG = "Deep_wieghts_configs/yolov3.cfg"



""" ===========================================
Tracker type = 'BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT', 'Kalman_Filter'
==============================================="""
TRACKER_TYPE = "CSRT" #"Kalman_Filter"#"Kalman_Filter"
LARGEST_OBJECT = True
UPDATE_TRACKER = 10 #  after processing this number of frames, apply object detection

"""============================================================
Detection method

different methods:
                    HOG_Vehicle
                    HOG_Pedestrian
                    Haar_Vehicle
                    Haar_Pedestrian
                    Deep_Yolo  => image can not be GrayScale                  
==============================================================="""
DETECTION_METHOD = "Deep_Yolo"#"Deep_Yolo"#"HOG_Pedestrian"#

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

DETECTION_CLASS = __deep_classes['person']
# tracker age in loss
max_age = 4