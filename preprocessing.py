from setting import FILE_ADDRESS_DEEP_GROUNDTHRUTH
import cv2
import numpy as np

class C_PREPROCESSING:
    
    @staticmethod
    def Color_Conversion(image, cspace):
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'GRAY':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif cspace == 'RGB2YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
            elif cspace == 'BGR2YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            elif cspace == "GRAY2RGB":
                feature_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
        return feature_image
    
    @staticmethod
    def resize(image, width):
        scale = np.divide(width, image.shape[1], dtype=np.float)
        height = int(np.round(image.shape[0]*scale))

        return cv2.resize(image, (width, height))
    @staticmethod
    def VOT2013_read_groundtruth_file(fileaddress):
        groundtruth = []
        f = tuple(open(FILE_ADDRESS_DEEP_GROUNDTHRUTH, "r"))
        for i in f:
            k = []
            for j in i.split(','):
                k.append(int(j.split('.')[0]))
            groundtruth.append(k)
        
        return groundtruth
    
    @staticmethod
    def MOT_read_groundtruth_file(fileaddress):
        frame_number_o = -1
        total_bboxes = []
        current_bbox = []
        fp = tuple(open(FILE_ADDRESS_DEEP_GROUNDTHRUTH, "r"))
        for line in fp:# line:
            temp = line.split(',')#re.findall(r'\d+', line)            
            
            frame_number = int(float(temp[0]))
            object_id = int(float(temp[1]))
            left, top, width, height = int(float(temp[2])), int(float(temp[3])),int(float(temp[4])), int(float(temp[5]))                
            
            if frame_number==frame_number_o:            
                current_bbox.append([top, left, width, height, object_id])
            else:
                if current_bbox != []:
                    total_bboxes.append(current_bbox)
                current_bbox = []      
                current_bbox.append([top, left, width, height, object_id])
                
            frame_number_o = frame_number            
        
        return total_bboxes
    
    @staticmethod
    def MOT_gt_show(frame, groundtruth):        
        font = cv2.FONT_HERSHEY_SIMPLEX        
        for gt in groundtruth:
            cv2.rectangle(frame, (gt[1],gt[0]), (gt[1]+gt[2],gt[0]+gt[3]), (255,255,255),1)
            cv2.putText(frame,str(gt[4]),(gt[1]+2,gt[0]-2), font, 1,(255,255,255),2,cv2.LINE_AA)
        cv2.resize(frame, (frame.shape[1]/2, frame.shape[0]/2))
        return frame
