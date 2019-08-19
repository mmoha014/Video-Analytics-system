# Video-Analytics-system
This repository is the first steps of a video analytics system


# Video-Analytics-system
This repository is the first steps of implementing a video analytics system using mobile edge cloud computing (MECC). Currently, object detection and object tracking are implemented. The pipeline.py file is just object detection pipeline and pipeline2.py file is integration of object detection and tracking. Also, GUI_pipeline.py executes  two pipelines in one windows.

Diagram of current system is showing in the following picture.

![image](https://user-images.githubusercontent.com/54276260/63275359-65683f80-c26f-11e9-83f3-bb128a0aca72.png)

### Version history
| Version  | Date |Description
| ------------- | ------------- |---|
| 1.00  | July 5, 2019  | Initial release |
| 1.01  | July 16, 2019  |  a- creating a separate module for each detection algorithm (subdirectory) <br/>b- Dynamic change of detection algorithm<br/>c - Dynamic change of tracking algorithm<br> d- MOTChallenge output generator Module - generating text file for each sequence of dataset to evaluate tracking results by means of available development kits: [matlab](https://bitbucket.org/amilan/motchallenge-devkit/src/default/) and [python](https://github.com/cheind/py-motmetrics)<br>e- Integrating all parts of implementation and using a GUI to show the output (July 21)

### Integrating approaches
A sample output of integration is shown in bellow video, which makes it possible to switch among all possible approaches that are implemented in version 1.01.

[![](http://img.youtube.com/vi/_Wx3P0iq8ns/0.jpg)](http://www.youtube.com/watch?v=_Wx3P0iq8ns "")
