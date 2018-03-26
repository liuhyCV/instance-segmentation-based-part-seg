# Note
This code base framework code is from [Deformable Convolutional Networks](https://github.com/msracver/Deformable-ConvNets)
For more information about the base model, please see it's ReadMe.txt
https://github.com/liuhyCV/instance-segmentation-based-part-seg


# Model Baseline

## Object Detection
We train rcnn network on VOC2012 dataset.

(1)
traindata:voc2012train_val
testdata:voc2007test

Mean AP@0.5 = 0.7639
Mean AP@0.7 = 0.5818

(1)
traindata:voc2007train_val+voc2012train_val
testdata:voc2007test

Mean AP@0.5 = 0.7966
Mean AP@0.7 = 0.6299



# Instance Segmentation based Part Semantic Segmentation

## Part Semantic Segmentation

https://github.com/CSAILVision/sceneparsing
Scene Parsing through ADE20K Dataset. B. Zhou, H. Zhao, X. Puig, S. Fidler, A. Barriuso and A. Torralba. Computer Vision and Pattern Recognition (CVPR), 2017. (http://people.csail.mit.edu/bzhou/publication/scene-parse-camera-ready.pdf)


![image][https://github.com/liuhyCV/instance-segmentation-based-part-seg/Cascade Segmentation Module.jpg]




## Instance Segmentation with Part Semantic Segmentation and Bbox Detection

Here we use Mask RCNN bbox detection result


## Instance Segmentation with Part Semantic Segmentation and Connection Prediction


