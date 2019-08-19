from setting import *
import cv2
import numpy as np
import os

class C_DETECTION_FRCNN:
    def __init__(self):
        # derive the paths to the Mask R-CNN weights and model configuration
        weightsPath = os.path.sep.join(['mask-rcnn-coco', "frozen_inference_graph.pb"])
        configPath = os.path.sep.join(['mask-rcnn-coco', "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"])

        # load our Mask R-CNN trained on the COCO dataset (90 classes)
        # from disk
        print("[INFO] loading Mask R-CNN from disk...")
        net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)
    
    def __deep_FRCNN_detection(self, inp_frame):
        width = inp_frame.shape[1]
        height = inp_frame.shape[0]

        # blob = cv2.dnn.blobFromImage(inp_frame, scale,  (416, 416), (0,0,0), True, crop=False)
        blob = cv2.dnn.blobFromImage(inp_frame, swapRB=True, crop=False)
        net.setInput(blob)
        (outs, masks) = net.forward(["detection_out_final", "detection_masks"])

        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.3
        nms_threshold = 0.4

        # for out in outs:
        #     for detection in out:
        #         scores = detection[5:]
        #         class_id = np.argmax(scores)
        #         confidence = scores[class_id]
        #         if confidence > conf_threshold:
        #             center_x = int(detection[0]*width)                
        #             center_y = int(detection[1]*height)
        #             w = int(detection[2] * width)
        #             h = int(detection[3] * height)
        #             x = center_x - w / 2
        #             y = center_y -h / 2
        #             class_ids.append(class_id)
        #             confidences.append(float(confidence))
        #             boxes.append([x,y, w, h])
        for i in range(0, boxes.shape[2]):
            # extract the class ID of the detection along with the confidence
            # (i.e., probability) associated with the prediction
            classID = int(boxes[0, 0, i, 1])
            confidence = boxes[0, 0, i, 2]

            # filter out weak predictions by ensuring the detected probability
            # is greater than the minimum probability
            if confidence > _confidence:                
                box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")
                boxW = endX - startX
                boxH = endY - startY
                boxes.apend([startX, startY, boxW, boxH])
        
        return boxes


    def Detection_plus_BoundingBox(self,inp_frame):
        detected_boxes = []
        roi_boxes = self.__deep_FRCNN_detection(inp_frame)
        
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