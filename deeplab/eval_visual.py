import cv2
import sys
#sys.path.insert(0, '/home/dhc/adas-segmentation-cityscape/mxnet/python')
#sys.path.insert(0, '/home/winsty/mxnet/python')
sys.path.insert(0, '..')
import mxnet as mx
from mxnet import image

print mx.__version__
import argparse
import logging
import os
import numpy as np
import time

key_car_parts = {
    'background': 0,
    'frontside': 1,
    'leftside': 2,
    'rightside': 3,
    'backside': 4,
    'roofside': 5,
    'leftmirror': 6,
    'rightmirror': 7,
    'fliplate': 8,
    'bliplate': 9,
    'door_1': 10,
    'door_2': 11,
    'door_3': 10,
    'door_4': 11,

    'wheel_1': 12,
    'wheel_2': 13,
    'wheel_3': 14,
    'wheel_4': 15,
    'wheel_5': 14,
    'wheel_6': 15,

    'headlight_1': 16,
    'headlight_2': 17,
    'headlight_3': 16,
    'headlight_4': 17,
    'headlight_5': 17,
    'headlight_6': 16,
    'headlight_7': 17,
    'headlight_8': 16,
    'headlight_9': 17,

    'window_1': 18,
    'window_2': 19,
    'window_3': 20,
    'window_4': 20,
    'window_5': 20,
    'window_6': 20,
    'window_7': 20,
    'window_8': 20,
    'window_9': 20,
    'window_10': 20,
    'window_11': 20,
    'window_12': 20,
    'window_13': 20,
    'window_14': 20,
    'window_15': 20,
    'window_16': 20,
    'window_17': 20,
    'window_18': 20,
    'window_19': 20,
    'window_20': 20,

    'ignore': 255
}

classes = ['background','aeroplane','bicycle','bird','boat',
           'bottle','bus','car','cat','chair','cow','diningtable',
           'dog','horse','motorbike','person','potted plant',
           'sheep','sofa','train','tv/monitor']
# RGB color for each class
colormap = [[0,0,0],[128,0,0],[0,128,0], [128,128,0], [0,0,128],
            [128,0,128],[0,128,128],[128,128,128],[64,0,0],[192,0,0],
            [64,128,0],[192,128,0],[64,0,128],[192,0,128],
            [64,128,128],[192,128,128],[0,64,0],[128,64,0],
            [0,192,0],[128,192,0],[0,64,128]]

def label2image(pred):
    x = pred.astype('int32')
    cm = np.array(colormap).astype('uint8')
    return np.array(cm[x,:])



def load_params(prefix1="./enet",epoch1=140):
    save_dict1 = mx.nd.load('%s-%04d.params' % (prefix1, epoch1))
    arg_params = {}
    aux_params = {}
    for k, v in save_dict1.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
            #temp = re.findall(r"conv(.+?)_3", name)
            #if temp!=[] and temp[0] in share_weight_list:
            #arg_params['dilate1_'+name] = v
            arg_params['dilate'+name] = v
        if tp == 'aux':
            aux_params[name] = v
            #temp = re.findall(r"conv(.+?)_3", name)
            #if temp!=[] and temp[0] in share_weight_list:
             #   aux_params['dilate1_'+name] = v
            aux_params['dilate'+name] = v
    return arg_params,aux_params



def eval_IOU():

    num_class = 21
    dirc = '/home/csc302/workspace/liuhy/Deformable-ConvNets/data/VOC2010Part/VOC2012'
    eval_list_path = 'ImageSets/Main/val_seg.txt'
    model_previx = '/home/csc302/workspace/liuhy/Deformable-ConvNets/output/voc12/deeplab_resnet_v1_101_voc12_segmentation_base/2012_train_seg'
    model_name = 'deeplab_resnet_v1_101_voc12_segmentation_base'
    epoch = 12
    ctx = mx.gpu(0)

    enet, enet_args, enet_auxs = mx.model.load_checkpoint(model_previx+model_name, epoch)
    enet_args, enet_auxs = load_params(model_previx, epoch)

    lines = file(os.path.join(dirc, eval_list_path)).read().splitlines()
    num_image = len(lines)
    num = 0

    eval_height_size = 640
    eval_width_size = 800

    exector_2 = enet.simple_bind(ctx, data=(1, 3, eval_height_size, eval_width_size), softmax_label=(1, 1, eval_height_size, eval_width_size),
                                 grad_req="null")
    exector_2.copy_params_from(enet_args, enet_auxs, True)

    for line in lines:
        t = time.time()

        data_img_name = line.strip('\n').split("\t")
        data_name_index = data_img_name[0]
        img_file_path = os.path.join(dirc, 'JPEGImages', data_name_index + '.jpg')
        label_file_path = os.path.join(dirc, 'SegmentationClass', data_name_index + '.png')

        img = image.imread(img_file_path).asnumpy()
        label = image.imread(label_file_path).asnumpy()
        img = cv2.resize(img, (eval_width_size, eval_height_size), interpolation=cv2.INTER_NEAREST)
        label = cv2.resize(label, (eval_width_size, eval_height_size), interpolation=cv2.INTER_NEAREST)

        # inference
        data = np.transpose(img, [2, 0, 1])
        print data.shape
        data = data.reshape((-1, data.shape[0], data.shape[1], data.shape[2]))
        data = mx.nd.array(data, ctx)
        exector_2.forward(is_train=False, data=data)
        output = exector_2.outputs[0].asnumpy()

        # processing output
        pred = np.squeeze(output)

        score = np.argmax(pred, axis=0)
        score = cv2.resize(score, (eval_width_size, eval_height_size), interpolation=cv2.INTER_NEAREST)

        heat = np.max(pred, axis=0)
        heat *= 255
        heat_map = cv2.applyColorMap(heat.astype(np.uint8), cv2.COLORMAP_JET)
        heat_map = cv2.resize(heat_map, (eval_width_size, eval_height_size))

        pred_ids = np.argmax(pred, axis=0)
        pred_color = label2image(pred_ids)

        visual = np.zeros((eval_height_size*2, eval_width_size*2, 3))
        visual[eval_height_size:, :eval_width_size, :] = pred_color
        visual[eval_height_size:, eval_width_size:, :] = label
        visual[:eval_height_size, eval_width_size:, :] = heat_map
        visual[:eval_height_size, :eval_width_size, :] = img

        if not os.path.exists('../visual/seg_pascal_voc_part'):
            os.makedirs('../visual/seg_pascal_voc_part')
        cv2.imwrite('../visual/seg_pascal_voc_part/'+ data_name_index + '.png', visual)

        t3 = time.time() - t
        num += 1

        print 'testing {}/{}'.format(num, num_image)


if __name__ == "__main__":

    eval_IOU()
