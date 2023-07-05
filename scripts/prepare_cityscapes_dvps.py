# pylint: skip-file

import argparse
import multiprocessing as mp
import shutil
from pathlib import Path

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--vps_path', type=str, required=True)
parser.add_argument('--dvps_depth', type=str, required=True)
parser.add_argument('--out_path', type=str, required=True)
args = parser.parse_args()
vps_path = Path(args.vps_path).absolute()
dvps_detph = Path(args.dvps_depth).absolute()
out_path = Path(args.out_path).absolute()

for split in ['train', 'val']:
    print(f'Process {split}...')

    img_file_names = (vps_path / split / 'img').glob('*.png')
    img_file_names = [file_name for file_name in img_file_names]
    gt_file_names = (vps_path / split / 'panoptic_inst').glob('*.png')
    gt_file_names = [file_name for file_name in gt_file_names]
    depth_file_names = (dvps_detph / 'video_sequence' / split).glob('*.png')
    depth_file_names = [file_name for file_name in depth_file_names]

    def copy_image(file_name):
        city = file_name.name.split('_')[2]
        new_file_name = '_'.join(file_name.name.split('_')[2:])
        new_file_name = new_file_name.replace('new', 'left')
        dst_file = out_path / 'leftImg8bit' / split / city / new_file_name
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(file_name, dst_file)

    def copy_gt(file_name):
        city = file_name.name.split('_')[2]
        new_file_name = '_'.join(file_name.name.split('_')[2:])
        new_file_name = new_file_name.replace('final_mask', 'gtFine_instanceIds')
        new_file_name = new_file_name.replace('gtFine_color', 'gtFine_instanceIds')
        dst_file = out_path / 'gtFine' / split / city / new_file_name
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(file_name, dst_file)

    def copy_depth(file_name):
        city = file_name.name.split('_')[2]
        new_file_name = '_'.join(file_name.name.split('_')[2:])
        dst_file = out_path / 'depth' / split / city / new_file_name
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(file_name, dst_file)

    N = max(1, mp.cpu_count() - 1)
    with mp.Pool(processes=N) as p:
        with tqdm(total=len(img_file_names), desc='Copy images') as pbar:
            for _ in p.imap_unordered(copy_image, img_file_names):
                pbar.update()
        with tqdm(total=len(gt_file_names), desc='Copy annotations') as pbar:
            for _ in p.imap_unordered(copy_gt, gt_file_names):
                pbar.update()
        with tqdm(total=len(depth_file_names), desc='Copy depth') as pbar:
            for _ in p.imap_unordered(copy_depth, depth_file_names):
                pbar.update()
