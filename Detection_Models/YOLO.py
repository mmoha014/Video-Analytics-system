from setting import *
import cv2
import numpy as np

class C_DETECTION_YOLO:
    def __init__(self):
        self.__detector = cv2.dnn.readNet(FILE_ADDRESS_DEEP_YOLO_WEIGHT, FILE_ADDRESS_DEEP_YOLO_CONFIG)


    def __get_output_layers(self,net):
            layer_names = net.getLayerNames()
            output_layers = [layer_names[i[0]-1] for i in self.__detector.getUnconnectedOutLayers()]    
            return output_layers
            
    def __deep_Yolo_detection(self, inp_frame):
        width = inp_frame.shape[1]
        height = inp_frame.shape[0]
        
        scale = 0.00392          
        
        # create input blob
        blob = cv2.dnn.blobFromImage(inp_frame, scale,  (416, 416), (0,0,0), True, crop=False)
        
        # set input blob for the network
        self.__detector.setInput(blob)
        
        

        outs = self.__detector.forward(self.__get_output_layers(self.__detector))	
        
        #initialization
        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence>0.5:
                    center_x = int(detection[0]*width)                
                    center_y = int(detection[1]*height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = center_x - w / 2
                    y = center_y -h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x,y, w, h])
        
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        return boxes, confidences, class_ids, indices
        
    def Detection_plus_BoundingBox(self,inp_frame):
        detected_boxes = []
        roi_boxes, roi_confidences, roi_class, roi_indices = self.__deep_Yolo_detection(inp_frame)
        
        # Drawing the boxes around the detected objects and saving the objects simultaneously        
        boxes = []
        bigest_idx = 0
        area = 0.0
        for idx,i in enumerate(roi_indices):
            i = i[0]
            for j, v in enumerate(roi_boxes[i]):
                if v<0:
                    roi_boxes[i][j] = 0
            
                box = roi_boxes[i]
                x = int(round(box[0]))
                y = int(round(box[1]))
                w = int(round(box[2]))
                h = int(round(box[3]))
        
            boxes.append((x,y, w, h))  
            # if area<w*h:
            #     area = w*h
            #     bigest_idx = idx
            # draw_bounding_box(frame, roi_class[i], roi_confidences[i], x,y, x+w, y+h)
            if roi_class[i] == DETECTION_CLASS:
                cv2.rectangle(inp_frame, (x,y), (x+w,y+h), [0,0,255])
                detected_boxes.append((x,y, w, h))
        
        return detected_boxes, inp_frame