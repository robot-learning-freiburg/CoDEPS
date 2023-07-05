# pylint: skip-file

import argparse
import multiprocessing as mp
import shutil
from pathlib import Path

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--in_path', type=str, required=True)
parser.add_argument('--out_path', type=str, required=True)
args = parser.parse_args()
in_path = Path(args.in_path).absolute()
out_path = Path(args.out_path).absolute()

for split in ['train', 'val']:
    print(f'Process {split}...')

    dvps_path = in_path / 'semkitti-dvps-annotations' / 'video_sequence' / split
    depth_file_names = dvps_path.glob('*depth*.png')
    depth_file_names = [file_name for file_name in depth_file_names]
    semantics_file_names = dvps_path.glob('*gtFine*.png')
    semantics_file_names = [file_name for file_name in semantics_file_names]
    kitti_path = in_path / 'dataset' / 'sequences'
    image_sequences = kitti_path.glob('*')
    image_sequences = [sequence for sequence in image_sequences]

    def copy_depth(file_name):
        sequence = int(file_name.name.split('_')[0])
        new_file_name = '_'.join(file_name.name.split('_')[1:])
        new_file_name = new_file_name.replace('_depth', '')
        dst_file = out_path / 'data_2d_depth' / f'{sequence:02}' / new_file_name
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(file_name, dst_file)

    def copy_semantics(file_name):
        sequence = int(file_name.name.split('_')[0])
        new_file_name = '_'.join(file_name.name.split('_')[1:])
        new_file_name = new_file_name.replace('_gtFine', '')
        dst_file = out_path / 'data_2d_semantics' / f'{sequence:02}' / new_file_name
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(file_name, dst_file)

    def copy_images(sequence_dir):
        sequence = int(sequence_dir.name)
        if sequence > 10:
            return
        dst_path = (out_path / 'data_2d_raw' / f'{sequence:02}')
        dst_path.mkdir(parents=True, exist_ok=True)
        for file_name in ['calib.txt', 'times.txt']:
            file = sequence_dir / file_name
            dst_file = dst_path / file.name
            shutil.copy(file, dst_file)
        images_dir = sequence_dir / 'image_2'
        dst_dir = dst_path / 'image_2'
        shutil.copytree(images_dir, dst_dir)

    N = max(1, mp.cpu_count() - 1)
    with mp.Pool(processes=N) as p:
        with tqdm(total=len(depth_file_names), desc='Copy depth') as pbar:
            for _ in p.imap_unordered(copy_depth, depth_file_names):
                pbar.update()
        with tqdm(total=len(semantics_file_names), desc='Copy annotations') as pbar:
            for _ in p.imap_unordered(copy_semantics, semantics_file_names):
                pbar.update()
        if split == 'train':
            with tqdm(total=len(image_sequences), desc='Copy images') as pbar:
                for _ in p.imap_unordered(copy_images, image_sequences):
                    pbar.update()
