import argparse
import json
import os.path as osp
from tqdm import tqdm

import numpy as np
import concurrent.futures
from mmengine.utils import mkdir_or_exist
from PIL import Image

def read_files_from_jsonl(jsonl_path):
    with open(jsonl_path, 'r') as file:
        data = [json.loads(line) for line in file]
    return data

def pares_args():
    parser = argparse.ArgumentParser(
        description='Convert overlaptext dataset to mmsegmentation format')
    parser.add_argument(
        '--dataset-path',
        type=str,
        default="",
        help='overlaptext dataset path.')
    parser.add_argument(
        '--anno_name',
        type=list,
        default=['diffusion_gt.jsonl'],
        help='the annotation file name.'
    )
    parser.add_argument(
        '--save-path',
        default='',
        type=str,
        help='save path of the dataset.')
    parser.add_argument(
        '--num-workers',
        default=16,
        type=int,
        help='number of worker threads for parallel processing.')
    args = parser.parse_args()
    return args

def process_image(line, dataset_path, _type, save_path):
    try:
        img = Image.open(osp.join(dataset_path, _type, line['image']))
        mask = Image.open(osp.join(dataset_path, _type, line['semseg_conditioning_image']))
        
        if line['image'].split('.')[-1] == 'jpg':
            image = line['image'].replace('jpg', 'png')
        else:
            image = line['image']

        img_save_path = osp.join(save_path, _type, 'img_dir/train', image.replace('/', '_'))
        mask_save_path = osp.join(save_path, _type, 'ann_dir/train', line['semseg_conditioning_image'].split('/', 1)[-1].replace('/', '_'))

        img.save(img_save_path)
        mask.save(mask_save_path)

        # img = Image.open(osp.join(dataset_path, line['image']))
        # mask = Image.open(osp.join(dataset_path, line['semseg_conditioning_image']))
        # if line['image'].split('.')[-1] == 'jpg':
        #     image = line['image'].replace('jpg', 'png')
        # else:
        #     image = line['image']
        # img.save(
        #     osp.join(
        #         save_path, 'img_dir/val', image.replace('/', '_')
        #     )
        # )
        # mask.save(
        #     osp.join(
        #         save_path, 'ann_dir/val', line['semseg_conditioning_image'].split('/', 1)[-1].replace('/', '_')
        #     )
        # )

    except Exception as e:
        print(f"Error processing {line['image']}: {e}")

def main():
    args = pares_args()
    dataset_path = args.dataset_path
    save_path = args.save_path
    train_anno_name = args.anno_name[0]
    # val_anno_name = args.anno_name[1]
    trainset_type_sample = [
            # "printed_en",
            # "printed_hw_en",
            # "printed_hw_zh_en",
            "printed_num",
            # "printed_zh_en_bill",
        ]
    for _type in trainset_type_sample:

        train_data = read_files_from_jsonl(osp.join(dataset_path, _type, train_anno_name))
        # val_data = read_files_from_jsonl(osp.join(dataset_path, val_anno_name))

        mkdir_or_exist(osp.join(save_path, _type, 'img_dir/train'))
        mkdir_or_exist(osp.join(save_path, _type, 'img_dir/val'))
        mkdir_or_exist(osp.join(save_path, _type, 'ann_dir/train'))
        mkdir_or_exist(osp.join(save_path, _type, 'ann_dir/val'))

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            list(tqdm(executor.map(lambda line: process_image(line, dataset_path, _type, save_path), train_data), total=len(train_data)))






if __name__ == '__main__':
    main()