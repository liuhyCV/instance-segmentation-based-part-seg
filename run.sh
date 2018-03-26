#!/usr/bin/env bash

# run deeplab
#python experiments/deeplab/deeplab_train_test.py --cfg experiments/deeplab/cfgs/deeplab_resnet_v1_101_voc12_segmentation_base.yaml


# run faster rcnn
#python experiments/faster_rcnn/rcnn_end2end_train_test.py --cfg experiments/faster_rcnn/cfgs/resnet_v1_101_voc0712_rcnn_end2end.yaml
python experiments/faster_rcnn/rcnn_test.py --cfg experiments/faster_rcnn/cfgs/resnet_v1_101_voc0712_rcnn_end2end.yaml

