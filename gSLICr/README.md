# gSLICr: SLIC superpixels at over 250Hz

For gSLICr install and use, please visit the project website <http://www.robots.ox.ac.uk/~victor/gslicr> 
and github website <https://github.com/carlren/gSLICr>


Here I only describe how to generate superpixels points pairs




# 1. Instruction for use gSLICr

## 1 no_segs / spixel_size: superpixels nums / size

    my_settings.img_size.x = 640;
    my_settings.img_size.y = 480;
    my_settings.no_segs = 2000;
    my_settings.spixel_size = 16;
    my_settings.coh_weight = 0.6f;
    my_settings.no_iters = 5;


## 2 image save path

    char out_name[100];
    sprintf(out_name, "/home/csc302/workspace/liuhy/gSLICr/seg_%04i.pgm", save_count);
    gSLICr_engine->Write_Seg_Res_To_PGM(out_name);
    
    
# 2. 基本原理：

gSLICr生成的pgm格式文件：H*W的矩阵，其中，矩阵值为不同superpixels标示，不同superpixels不同数值

首先，对于每一个superpixels，计算以下几个结果：
Bbox、Center coordinate、Adjacent superpixels label

Bbox：
左上顶点 + 右下顶点

Center coordinate：Bbox中心点


Adjacent superpixels label：
对每一个superpixels，对其进行膨胀操作，后减去原区域，得到其膨胀后的边界区域，该区域中存在的其他
标识，即为相邻的superpixels








