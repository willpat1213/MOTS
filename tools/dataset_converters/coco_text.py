# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import shutil
import json
import os
from functools import partial

import numpy as np
from mmengine.utils import (mkdir_or_exist, track_parallel_progress,
                            track_progress)
from PIL import Image
from scipy.io import loadmat

COCO_LEN = 10000

clsID_to_trID = {
    0: 0,
    1: 1,
    255: 255}


def convert_to_trainID(filename, in_img_dir, in_ann_dir, out_img_dir,
                       out_mask_dir, is_train):
    imgpath = filename.replace('.png', '.jpg')
    maskpath = filename
    shutil.copyfile(
        osp.join(in_img_dir, imgpath),
        osp.join(out_img_dir, 'train', imgpath) if is_train else osp.join(
            out_img_dir, 'val', imgpath))
    shutil.copyfile(
        osp.join(in_ann_dir, maskpath),
        osp.join(out_mask_dir, 'train', maskpath) if is_train else osp.join(
            out_mask_dir, 'val', maskpath))

def generate_coco_list(coco_json_path, coco_ts_path):
    datas = json.load(open(coco_json_path, 'r'))

    im_list  = [i for _, i in datas['imgs'].items()]
    im_train = list(filter(lambda x:x['set'] == 'train', im_list))
    im_train = [x['file_name'] for x in im_train]
    im_val   = list(filter(lambda x:x['set'] == 'val'  , im_list))
    im_val = [x['file_name'] for x in im_val]

    print('length of list {} & {}'.format(len(im_train), len(im_val)))

    coco_ts_train = []
    coco_ts_val = []
    for filename in os.listdir(coco_ts_path):
        if filename.replace('.png', '.jpg') in im_train:
            coco_ts_train.append(filename)
        elif filename.replace('.png', '.jpg') in im_val:
            coco_ts_val.append(filename)
        else:
            print(f'{filename} cant match!')

    return coco_ts_train, coco_ts_val

def parse_args():
    parser = argparse.ArgumentParser(
        description=\
        'Convert COCO Text annotations to mmsegmentation format')  # noqa
    parser.add_argument('coco_path', help='coco text path')
    parser.add_argument('-o', '--out_dir', help='output path')
    parser.add_argument(
        '--nproc', default=16, type=int, help='number of process')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    coco_path = args.coco_path
    coco_json_path = osp.join(coco_path, 'cocotext.v2.json')
    coco_ts_path = osp.join(coco_path, 'COCO_TS_labels')
    coco_image_path = osp.join(coco_path, 'train2014')
    nproc = args.nproc

    out_dir = args.out_dir or coco_path
    out_img_dir = osp.join(out_dir, 'images')
    out_mask_dir = osp.join(out_dir, 'annotations')

    mkdir_or_exist(osp.join(out_img_dir, 'train'))
    mkdir_or_exist(osp.join(out_img_dir, 'val'))
    mkdir_or_exist(osp.join(out_mask_dir, 'train'))
    mkdir_or_exist(osp.join(out_mask_dir, 'val'))

    train_list, test_list = generate_coco_list(coco_json_path, coco_ts_path)
    print('length of list {} & {}'.format(len(train_list), len(test_list)))

    if args.nproc > 1:
        track_parallel_progress(
            partial(
                convert_to_trainID,
                in_img_dir=coco_image_path,
                in_ann_dir=coco_ts_path,
                out_img_dir=out_img_dir,
                out_mask_dir=out_mask_dir,
                is_train=True),
            train_list,
            nproc=nproc)
        track_parallel_progress(
            partial(
                convert_to_trainID,
                in_img_dir=coco_image_path,
                in_ann_dir=coco_ts_path,
                out_img_dir=out_img_dir,
                out_mask_dir=out_mask_dir,
                is_train=False),
            test_list,
            nproc=nproc)
    else:
        track_progress(
            partial(
                convert_to_trainID,
                in_img_dir=coco_image_path,
                in_ann_dir=coco_ts_path,
                out_img_dir=out_img_dir,
                out_mask_dir=out_mask_dir,
                is_train=True), train_list)
        track_progress(
            partial(
                convert_to_trainID,
                in_img_dir=coco_image_path,
                in_ann_dir=coco_ts_path,
                out_img_dir=out_img_dir,
                out_mask_dir=out_mask_dir,
                is_train=False), test_list)

    print('Done!')


if __name__ == '__main__':
    main()
