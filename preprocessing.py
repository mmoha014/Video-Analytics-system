from setting import FILE_ADDRESS_DEEP_GROUNDTHRUTH
import cv2
import numpy as np
import cv2
import matplotlib.pyplot as plt

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
    def resize(imageToPredict, target_size, bboxes):    
        # print(imageToPredict.shape)

        # Note: flipped comparing to your original code!
        # x_ = imageToPredict.shape[0]
        # y_ = imageToPredict.shape[1]
        # cv2.rectangle(imageToPredict, (bboxes[0][0],bboxes[0][1]),(bboxes[0][2],bboxes[0][3]),(0,255,0),1)
        # cv2.imshow("orig", imageToPredict)
        # cv2.waitKey(0)
        y_ = imageToPredict.shape[0]
        x_ = imageToPredict.shape[1]

        targetSize = target_size
        x_scale = targetSize / x_
        y_scale = targetSize / y_
        # print(x_scale, y_scale)
        img = cv2.resize(imageToPredict, (targetSize, targetSize));
        # print(img.shape)
        img = np.array(img);

        # original frame as named values
        newbboxes = []
        for i,box in enumerate(bboxes):
            (origLeft, origTop, origRight, origBottom) = box[:-1]#(136, 164 , 237, 466)
            
            x = int(np.round(origLeft * x_scale))
            y = int(np.round(origTop * y_scale))
            xmax = int(np.round(origRight * x_scale))
            ymax = int(np.round(origBottom * y_scale))
            newbboxes.append([x, y, xmax, ymax])
            # cv2.rectangle(img, (x,y,xmax, ymax), (255,255,0),1)
            # cv2.rectangle(img, (newbboxes[i][0], newbboxes[i][1]), (newbboxes[i][2], newbboxes[i][3]), (255, 255, 0), 1)
            #print("x:", x, " y:", y,"  xmax:", xmax, "  ymax:",ymax)
            # print("i0:", newbboxes[i][0], "  i1:", newbboxes[i][1], "  i2:", newbboxes[i][2], "  i3:", newbboxes[i][3])
            #print("\n\n")
            
        # cv2.imshow("output", imageToPredict)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return img, newbboxes

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
    def MOT_read_frame_bboxes_from_groundtruth(gt, frame_number):
        res_list = [i for i in range(len(gt)) if gt[i][0][0] == gt]
        return res_list

    def MOT2017_read_groudntruth_file(fileaddress, number_of_frames):
        fp = tuple(open(FILE_ADDRESS_DEEP_GROUNDTHRUTH,"r"))
        gt_boxes = [[] for i in range(number_of_frames)]
        for line in fp:
            temp = line.split(',')
            frame_number = frame_number = int(float(temp[0]))
            if frame_number == 1:
                a=0
            object_id = int(float(temp[1]))
            left, top, width, height = int(float(temp[2])), int(float(temp[3])),int(float(temp[4])), int(float(temp[5])) 
            gt_boxes[frame_number-1].append([top, left, width, height, object_id])
        
        return gt_boxes

        
    def MOT2015_read_groundtruth_file(fileaddress):
        frame_number_o = -1
        total_bboxes = []
        current_bbox = None

        fp = tuple(open(FILE_ADDRESS_DEEP_GROUNDTHRUTH, "r"))
        for line in fp:# line:
            temp = line.split(',')#re.findall(r'\d+', line)            
            
            frame_number = int(float(temp[0]))
            object_id = int(float(temp[1]))
            left, top, width, height = int(float(temp[2])), int(float(temp[3])),int(float(temp[4])), int(float(temp[5]))                
            
            if frame_number==frame_number_o:            
                current_bbox.append([top, left, width, height, object_id])
            else:
                if current_bbox != None:
                    total_bboxes.append(current_bbox)
                current_bbox = []      
                current_bbox.append([left, top, width, height, object_id])
                
            frame_number_o = frame_number            
        
        return total_bboxes
    
    @staticmethod
    def MOT_gt_show(frame, groundtruth):        
        font = cv2.FONT_HERSHEY_SIMPLEX        
        for gt in groundtruth:
            cv2.rectangle(frame, (gt[1],gt[0]), (gt[1]+gt[2],gt[0]+gt[3]), (255,255,255),1)
            cv2.putText(frame,str(gt[4]),(gt[1]+2,gt[0]-2), font, 1,(255,255,255),2,cv2.LINE_AA)
        cv2.resize(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2)))
        cv2.imshow("out", frame)
        cv2.waitKey(0)
        return frame
