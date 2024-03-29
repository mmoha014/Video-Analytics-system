# Video-Analytics-system
This repository is the first steps of implementing a video analytics system using mobile edge cloud computing (MECC). Currently, object detection and object tracking are implemented. The pipeline.py file is just object detection pipeline and pipeline2.py file is integration of object detection and tracking. Also, GUI_pipeline.py executes  two pipelines in one windows.

Diagram of current system is showing in the following picture.

![image](https://user-images.githubusercontent.com/54276260/63275359-65683f80-c26f-11e9-83f3-bb128a0aca72.png)

To read the YOLO weights, it is not needed to convert the weights of DarkNet YOLO model to another format because OpenCV can work with weights directly. The related files can be downloaded from [Darknet project website](https://pjreddie.com/darknet/yolo/).




### Version history
| Version  | Date |Description
| ------------- | ------------- |---|
| 1.00  | July 5, 2019  | Initial release |
| 1.01  | July 16, 2019  |  a- creating a separate module for each detection algorithm (subdirectory) <br/>b- Dynamic change of detection algorithm<br/>c - Dynamic change of tracking algorithm<br> d- MOTChallenge output generator Module - generating text file for each sequence of dataset to evaluate tracking results by means of available development kits: [matlab](https://bitbucket.org/amilan/motchallenge-devkit/src/default/) and [python](https://github.com/cheind/py-motmetrics)<br>e- Integrating all parts of implementation and using a GUI to show the output (July 21)|
|1.02| Oct 20, 2019 | a- dividing the video to profiling windows and segments <br> b- F1 score calculation <br> c- changing the pipeline to work profiling window (each profiling window is 5 segments) and process each segment differently <br> d- applying the frame rate effect on computing F1_score, memory and CPU usage <br> e- definition objective function <br> f-  profiling function on whole profiling space <br> g- extracting top-k configurations from results of previous step <br> h- function for video segment processing based on top-k configurations <br> i- applying multi-thread processing for segment processing to be simultaneously run with pipelne and sharing variables between two threads <br> j- storing the output videos based on the selected configuation in profiling of the segment (no GUI to show results) <br> k- storing the output sequence based on winner config and its filename in each segment to show results after processing video (challenging to pass frames between two frames-future improvement)|

### Integrating approaches
A sample output of integration is shown in bellow video, which makes it possible to switch among all possible approaches that are implemented in version 1.01.

![image](https://user-images.githubusercontent.com/54276260/63279637-571e2180-c277-11e9-84da-7df7023f0de9.png)

