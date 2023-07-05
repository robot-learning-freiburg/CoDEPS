from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from numpy.typing import ArrayLike
from PIL import Image
from tqdm import tqdm
from yacs.config import CfgNode as CN

from datasets.dataset import Dataset
from datasets.preprocessing import augment_data, prepare_for_network
from misc import CameraModel


class SemKittiDvps(Dataset):
    def __init__(
            self,
            mode: str,
            cfg: CN,
            return_depth: bool = False,
            return_only_rgb: bool = False,
            sequences: Optional[List[str]] = None,
            label_mode: str = "codeps",
    ):
        super().__init__("sem_kitti_dvps", ["train", "val", "sequence"], mode, cfg, return_depth,
                         return_only_rgb, label_mode)
        if mode == "sequence":
            assert sequences is not None and len(sequences) > 0, \
                "In 'sequence' mode, sequences have to be given."
            for seq in sequences:
                assert seq in ["00", "02", "03", "04", "05", "06", "07", "08", "09", "10"], \
                    f"Passed invalid sequence: {seq}"

        # Follow the train/val split from VIP-DeepLab
        if self.mode == "train":
            self.sequences = ["00", "02", "03", "04", "05", "06", "07", "09", "10"]
        elif self.mode == "val":
            self.sequences = ["08"]
        else:
            self.sequences = sequences
        self.frame_paths = self._get_frames()

    def _get_frames(self) -> List[Dict[str, Path]]:
        """Gather the paths of the image, annotation, and camera intrinsics files
        Returns
        -------
        frames : list of dictionaries
            List containing the file paths of the RGB image, the semantic and instance annotations,
            and the camera intrinsics
        """
        depth_files = []
        for sequence in self.sequences:
            sequence_files = sorted(
                list((self.path_base / "data_2d_depth" / sequence).glob("*.png")))
            sequence_files = sequence_files[max(self.offsets):-max(self.offsets)]
            depth_files += sequence_files

        frames = []
        for depth in tqdm(depth_files, desc=f"Collect SemKITTI-DVPS frames: Seq. {self.sequences}"):
            sequence = depth.parent.name
            frame_id = depth.name.split("_")[0]
            rgb = self.path_base / "data_2d_raw" / sequence / "image_2" / f"{frame_id}.png"
            camera = self.path_base / "data_2d_raw" / sequence / "calib.txt"
            semantic = self.path_base / "data_2d_semantics" / sequence / f"{frame_id}_class.png"
            instance = self.path_base / "data_2d_semantics" / sequence / f"{frame_id}_instance.png"
            depth = depth if self.return_depth else None
            frames.append({"rgb": rgb, "semantic": semantic, "instance": instance, "camera": camera,
                           "depth": depth})
            for path in frames[-1].values():
                if path is not None:
                    assert path.exists(), f"File does not exist: {path}"
            # if len(frames) == 10:
            #     break
        return frames

    def __getitem__(self, index: int, do_network_preparation: bool = True,
                    do_augmentation: bool = True, return_only_rgb: bool = False) -> Dict[str, Any]:
        """Collect all data for a single sample
        Parameters
        ----------
        index : int
            Will return the data sample with this index
        Returns
        -------
        output : dict
            The output contains the following data:
            1) RGB images: center and offset images (3, H, W)
            2) semantic annotations (H, W)
            3) center heatmap of the instances (1, H, W)
            4) (x,y) offsets to the center of the instances (2, H, W)
            5) loss weights for the center heatmap and the (x,y) offsets (H, W)
            6) camera intrinsics
            7) GT depth map
        """
        # Read center and offset images
        image_path = self.frame_paths[index]["rgb"]
        image = Image.open(image_path).convert("RGB")
        image_size = image.size
        images = {0: self.resize(image)}
        center_number = image_path.stem
        number_digits = len(center_number)
        for offset in self.offsets:
            # We cannot just add to the index due to the concatenated sequences.
            offset_number = int(center_number) + offset
            offset_frame_path = image_path.parent / \
                                f"{str(offset_number).zfill(number_digits)}.png"
            assert offset_frame_path.exists(), f"Offset file does not exist: {offset_frame_path}"
            images[offset] = self.resize(Image.open(offset_frame_path).convert("RGB"))

        # Read camera intrinsics of the scene and rescale to the desired output image size
        camera_path = self.frame_paths[index]["camera"]
        with open(camera_path, "r", encoding="utf-8") as f:
            camera_data = f.readlines()[2].replace("P2: ", "").split(" ")
            camera_data = [float(data) for data in camera_data]
        camera_model = CameraModel(image_size[0], image_size[1], camera_data[0], camera_data[5],
                                   camera_data[2], camera_data[6])
        height, width = self.image_size
        scaled_camera_model = camera_model.get_scaled_model_image_size(width, height)

        output = {
            "rgb": images,
            "camera_model": scaled_camera_model.to_tensor(),
        }

        if not (self.return_only_rgb or return_only_rgb):
            # Read semantic map as Cityscapes labels
            semantic_path = self.frame_paths[index]["semantic"]
            semantic = cv2.imread(str(semantic_path), cv2.IMREAD_ANYDEPTH)  # 16-bit
            semantic = cv2.resize(semantic,
                                  list(reversed(self.image_size)),
                                  interpolation=cv2.INTER_NEAREST)

            # Read instance and convert to center heatmap and offset map
            instance_path = self.frame_paths[index]["instance"]
            instance = cv2.imread(str(instance_path), cv2.IMREAD_ANYDEPTH)  # 16-bit
            instance = cv2.resize(instance,
                                  list(reversed(self.image_size)),
                                  interpolation=cv2.INTER_NEAREST)

            # Convert to Cityscapes labels
            # For instances, this does not exactly follow the convention of Cityscapes but it sets
            #  all void labels to zero
            semantic_city = self._convert_semantics(semantic)
            instance_city = instance.copy()
            instance_city[semantic_city == 255] = 0

            # Generate semantic_weights map by instance mask size
            semantic_weights = np.ones_like(instance_city, dtype=np.uint8)
            semantic_weights[semantic_city == 255] = 0

            # Set the semantic weights by instance mask size
            height, width = self.image_size
            full_res_h, full_res_w = image_size[1], image_size[0]
            small_instance_area = self.small_instance_area_full_res * (height / full_res_h) * (
                    width / full_res_w)

            inst_id, inst_area = np.unique(instance_city, return_counts=True)
            for instance_id, instance_area in zip(inst_id, inst_area):
                # Skip stuff and unlabeled pixels
                if instance_id == 0:
                    continue

                if instance_area < small_instance_area:
                    semantic_weights[instance_city == instance_id] = self.small_instance_weight

            # Compute center heatmap and (x,y) offsets to the center for each instance
            offset, center = self.get_offset_center(instance_city)

            # Generate pixel-wise loss weights
            # Unlike Panoptic-DeepLab, we do not consider the is_crowd label. Following them, we
            #  ignore stuff in the offset prediction.
            center_weights = np.ones_like(center, dtype=np.uint8)
            center_weights[0][semantic_city == 255] = 0
            offset_weights = np.expand_dims(self._make_thing_mask(semantic_city),
                                            axis=0)

            output.update({
                "semantic": semantic_city,
                "semantic_weights": semantic_weights,
                "center": center,
                "center_weights": center_weights,
                "offset": offset,
                "offset_weights": offset_weights,
                "thing_mask": offset_weights,
                "instance": instance_city.astype(np.int32),
            })

            # Read the depth
            if self.return_depth:
                depth_path = self.frame_paths[index]["depth"]
                depth = cv2.imread(str(depth_path),
                                   cv2.IMREAD_ANYDEPTH).astype(np.float32)  # 16-bit
                depth[depth > 0] /= 256  # According to README
                depth = cv2.resize(depth,
                                   list(reversed(self.image_size)),
                                   interpolation=cv2.INTER_NEAREST)
                output["depth"] = depth

        if do_augmentation:
            augment_data(output, self.augmentation_cfg)

        if do_network_preparation:
            # Convert PIL image to torch.Tensor and normalize
            prepare_for_network(output, self.normalization_cfg)

        return output

    def _convert_semantics(self, semantic: ArrayLike) -> ArrayLike:
        if self.label_mode == "cityscapes":
            # Convert to Cityscapes labels and set non-existing labels to ignore, i.e., 255
            semantic_city = 255 * np.ones_like(semantic, dtype=np.uint8)
            mapping_list = [
                (8, 0),  # road
                (10, 1),  # sidewalk
                (12, 2),  # building
                # (, 3),  # wall
                (13, 4),  # fence
                (17, 5),  # pole
                # (, 6),  # traffic light
                (18, 7),  # traffic sign
                (14, 8),  # vegetation
                (16, 9),  # terrain
                # (, 10),  # sky
                (5, 11),  # person
                (6, 12),  # bicyclist -> rider
                (7, 12),  # motorcyclist -> rider
                (0, 13),  # car
                (3, 14),  # truck
                # (, 15),  # bus
                # (, 16),  # train
                (2, 17),  # motorcycle
                (1, 18),  # bicycle
                (9, 0),  # parking -> road
                (15, 8),  # trunk -> vegetation
            ]
        elif self.label_mode == "codeps":
            # Convert to our labels and set non-existing labels to ignore, i.e., 255
            mapping_list = [
                (8, 0),  # road
                (9, 0),  # parking -> road
                (10, 1),  # sidewalk
                (12, 2),  # building
                (13, 3),  # fence
                (17, 4),  # pole
                (18, 5),  # traffic sign
                (14, 6),  # vegetation
                (15, 6),  # trunk -> vegetation
                (16, 7),  # terrain
                # (, 8),  # sky
                (5, 9),  # person
                (6, 10),  # bicyclist -> rider
                (7, 10),  # motorcyclist -> rider
                (0, 11),  # car
                (3, 12),  # truck
                (2, 13),  # motorcycle -> two-wheeler
                (1, 13),  # bicycle -> two-wheeler
            ]
        else:
            raise ValueError(f"Unsupported label mode: {self.label_mode}")

        # Remove classes as specified in the config file
        mapping_list = self._rm_classes_mapping(self.remove_classes, mapping_list)

        semantic_city = 255 * np.ones_like(semantic, dtype=np.uint8)
        for mapping in mapping_list:
            semantic_city[semantic == mapping[0]] = mapping[1]
        return semantic_city

    @property
    def ignore_classes(self) -> List[int]:
        """Returns the classes that are present in Cityscapes but not in KITTI"""
        if self.label_mode == "cityscapes":
            return [3, 6, 10, 15, 16]
        if self.label_mode == "codeps":
            # return [8]
            return []
        raise ValueError(f"Unsupported label mode: {self.label_mode}")
