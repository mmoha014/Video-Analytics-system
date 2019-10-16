from setting import *
import cv2
import dlib
import numpy as np
from Detection_Models.YOLO import C_DETECTION_YOLO
from Detection_Models.HAAR import C_DETECTION_HAAR
from Detection_Models.HOG import C_DETECTION_HOG
from Detection_Models.YOLO_TINY import C_DETECTION_YOLO_TINY

class C_DETECTION:
    
    def __init__(self,Method):
        
        self.__Method = Method        
        
        if Method == 'HOG_Vehicle':            
            self.__detector = None
        elif Method == 'HOG_Pedestrian':
            self.__detector = C_DETECTION_HOG()#cv2.HOGDescriptor()
            # self.__detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        elif Method == "Haar_Vehicle":
            self.__detector = C_DETECTION_HAAR(FILE_ADDRESS_HAAR_VEHICLE)#cv2.CascadeClassifier(FILE_ADDRESS_HAAR_VEHICLE)
        elif Method == 'Haar_Pedestrian':
            self.__detector = C_DETECTION_HAAR(FILE_ADDRESS_HAAR_PEDESTRIAN)# do related loadings and preparations
        elif Method == 'Deep_Yolo':
            self.__detector = C_DETECTION_YOLO()#cv2.dnn.readNet(FILE_ADDRESS_DEEP_YOLO_WEIGHT, FILE_ADDRESS_DEEP_YOLO_CONFIG)
        elif Method == 'Deep_Yolo_Tiny':
            self.__detector = C_DETECTION_YOLO_TINY()#cv2.dnn.readNet(FILE_ADDRESS_DEEP_YOLO_TINY_WEIGHT, FILE_ADDRESS_DEEP_YOLO_TINY_CONFIG)

    def Switch_detection(self, Method):
        if self.__Method == Method:
            return
            
        self.__Method = Method        

        if Method == 'HOG_Vehicle':            
            self.__detector = None
        elif Method == 'HOG_Pedestrian':
            self.__detector = C_DETECTION_HOG()#cv2.HOGDescriptor()
            # self.__detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        elif Method == "Haar_Vehicle":
            self.__detector = C_DETECTION_HAAR(FILE_ADDRESS_HAAR_VEHICLE)#cv2.CascadeClassifier(FILE_ADDRESS_HAAR_VEHICLE)
        elif Method == 'Haar_Pedestrian':
            self.__detector = C_DETECTION_HAAR(FILE_ADDRESS_HAAR_PEDESTRIAN)# do related loadings and preparations
        elif Method == 'Deep_Yolo':
            self.__detector = C_DETECTION_YOLO()#cv2.dnn.readNet(FILE_ADDRESS_DEEP_YOLO_WEIGHT, FILE_ADDRESS_DEEP_YOLO_CONFIG)self.__Method = Method        
        

    def Detection_BoundingBox(self,inp_frame):  
        return self.__detector.Detection_plus_BoundingBox(inp_frame)      
    
                    
        