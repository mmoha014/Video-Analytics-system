# Video-Analytics-system
This repository is the first steps of a video analytics system


# Video-Analytics-system
This repository is the first steps of implementing a video analytics system using mobile edge cloud computing (MECC). Currently, object detection and object tracking are implemented. The pipeline.py file is just object detection pipeline and pipeline2.py file is integration of object detection and tracking. Also, GUI_pipeline.py executes  two pipelines in one windows.

Diagram of current system is showing in the following picture.

![diagram_video_analytics](https://user-images.githubusercontent.com/15813546/60735555-2739e980-9f69-11e9-99a1-175f9ec166c3.jpg)

### Version history
| Version  | Date |Description
| ------------- | ------------- |---|
| 1.00  | July 5, 2019  | Initial release |
| 1.01  | July 16, 2019  |  a- creating a separate module for each detection algorithm (subdirectory) <br/>b- Dynamic change of detection algorithm<br/>c - Dynamic change of tracking algorithm<br> d- MOTChallenge output generator Module - generating text file for each sequence of dataset to evaluate tracking results by means of available development kits: [matlab](https://bitbucket.org/amilan/motchallenge-devkit/src/default/) and [python](https://github.com/cheind/py-motmetrics)<br>e- Integrating all parts of implementation and using a GUI to show the output (July 21)

### Integrating approaches
A sample output of integration is shown in bellow video, which makes it possible to switch among all possible approaches that are implemented in version 1.01.

[![](http://img.youtube.com/vi/_Wx3P0iq8ns/0.jpg)](http://www.youtube.com/watch?v=_Wx3P0iq8ns "")
