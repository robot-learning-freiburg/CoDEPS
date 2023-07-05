import ctypes
import multiprocessing as mp
import pickle
from copy import deepcopy
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
import torch
from skimage.exposure import is_low_contrast
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset
from yacs.config import CfgNode as CN

from datasets.mixup import Mixup
from datasets.preprocessing import (
    augment_data,
    prepare_for_network,
    transfer_histogram_style,
)


class ReplayBuffer(TorchDataset):
    def __init__(self, adaptation_data: TorchDataset, cfg: CN, mode: str):
        assert mode in ["train", "val"], f"Unsupported mode: {mode}"

        super().__init__()
        self.target_data = adaptation_data
        self.source_data = None  # Dataset

        self.mode = mode
        offset = sum(getattr(self.target_data, "semantic_seq_mode_image_skipped", []))
        index = int((len(self.target_data) + offset) * cfg.target_dataset_adapt_ratio)
        if self.mode == "train":
            self.target_data.frame_paths = self.target_data.frame_paths[:index]
        else:  # "val"
            if getattr(self.target_data, "sequence_reference_mode", "rgb") != "rgb":
                index -= sum(self.target_data.semantic_seq_mode_image_skipped[:index])
            self.target_data.frame_paths = self.target_data.frame_paths[index:]

        self.source_num_samples = cfg.replay_sampler.nof_source_samples
        self.target_num_samples = cfg.replay_sampler.nof_target_samples
        self.buffer_indices = {}
        self.source_sampler = None
        self.source_samples_indices = []
        self.target_sampler = None
        self.samplers_seed = cfg.replay_sampler.seed
        self.buffer_seed = cfg.replay_buffer.seed

        # Workers share the same buffer indices
        MAX_TARGET_BUFFER_SIZE = len(self.target_data)  # The size has to be fixed in the beginning
        target_buffer = mp.Array(ctypes.c_int, MAX_TARGET_BUFFER_SIZE)
        shared_target_buffer = np.ctypeslib.as_array(target_buffer.get_obj())
        self.buffer_indices["target"] = torch.from_numpy(shared_target_buffer)
        self.buffer_indices["target"][:] = 0  # Samples added to the buffer will be set to 1
        self.lock = mp.Lock()

        # Buffer control
        if cfg.replay_buffer.target_size is not None:
            self.target_buffer_size = cfg.replay_buffer.target_size
        else:
            self.target_buffer_size = len(self.target_data)
        self.maximize_diversity = cfg.replay_buffer.maximize_diversity
        self.similarity_threshold = cfg.replay_buffer.similarity_threshold
        self.faiss_index = None  # Will be created when the first sample is added
        self.distance_matrix = None  # Will be created when first called
        self.distance_matrix_indices = None  # Will be created when first called
        self.buffer_remove_sampler = np.random.default_rng(seed=self.buffer_seed)

        # Data mixer
        self.cfg_mixup = cfg.mixup.clone()
        self.mixer = Mixup()
        self.mixup_sampler = np.random.default_rng(seed=self.samplers_seed)
        if cfg.mode == "off":
            self.cfg_mixup.defrost()
            self.cfg_mixup.general.active = False
            self.cfg_mixup.general.mixup_strategies = []
            self.cfg_mixup.freeze()

        # Hardcoded multi-domain adaptation
        # self.load_state()

    def __len__(self) -> int:
        return len(self.target_data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        # --------------------
        # Each worker is initialized with a different seed
        if self.source_sampler is None or self.target_sampler is None:
            self._initialize_samplers()
        # --------------------
        output = {"index": index}
        # if index < 932:
        #     return output

        # --------------------
        # Get current online data
        vanilla_adaptation_item = self.target_data.__getitem__(index, do_augmentation=False,
                                                               do_network_preparation=False)
        if self.mode == "train":
            # Delete unused fields
            for key in ["semantic_weights", "center_weights", "offset_weights"]:
                vanilla_adaptation_item.pop(key, None)
            # Used for target_augmented
            adaptation_item = {
                "rgb": deepcopy(vanilla_adaptation_item["rgb"]),
                "camera_model": deepcopy(vanilla_adaptation_item["camera_model"])
            }
        tmp_center_img = deepcopy(vanilla_adaptation_item["rgb"][0])
        image_is_low_contrast = is_low_contrast(np.array(vanilla_adaptation_item["rgb"][0]),
                                                fraction_threshold=0.2, lower_percentile=10,
                                                upper_percentile=90)
        prepare_for_network(vanilla_adaptation_item, self.target_data.normalization_cfg)
        output["target"] = vanilla_adaptation_item
        output["target_is_low_contrast"] = image_is_low_contrast
        output["target"]["rgb_original"] = tmp_center_img
        # --------------------

        if self.mode == "train":

            # --------------------
            # Sample from the source replay buffer
            if self.source_num_samples > 0:
                with self.lock:
                    # Sample the source data s.t. every item must be seen before any repetition
                    #  is allowed
                    remaining_source_samples = self.buffer_indices["source"].sum().item()
                    if self.source_num_samples < remaining_source_samples:
                        buffer_indices = torch.where(self.buffer_indices["source"])[0].tolist()
                        indices = self.source_sampler.choice(buffer_indices,
                                                             self.source_num_samples,
                                                             replace=False)
                        for i in indices:
                            self.buffer_indices["source"][i] = 0
                    elif self.source_num_samples == remaining_source_samples:
                        indices = torch.where(self.buffer_indices["source"])[0].numpy()
                        for i in self.source_samples_indices:
                            self.buffer_indices["source"][i] = 1  # Reset the buffer
                    else:  # self.source_num_samples > remaining_source_samples
                        indices_1 = torch.where(self.buffer_indices["source"])[0].numpy()
                        for i in self.source_samples_indices:
                            self.buffer_indices["source"][i] = 1  # Reset the buffer
                        buffer_indices = torch.where(self.buffer_indices["source"])[0].tolist()
                        indices_2 = self.source_sampler.choice(
                            buffer_indices, self.source_num_samples - len(indices_1), replace=False)
                        for i in indices_2:
                            self.buffer_indices["source"][i] = 0
                        indices = np.concatenate([indices_1, indices_2])

                output["source"] = []
                for i in indices:
                    source_data_item = self.source_data.__getitem__(i, do_augmentation=False,
                                                                    do_network_preparation=False)
                    # Keep a copy of the original center image
                    tmp_center_img = deepcopy(source_data_item["rgb"][0])

                    # Important to first transfer the histogram style and then augment the data
                    for k, v in source_data_item["rgb"].items():
                        source_data_item["rgb"][k] = transfer_histogram_style(v, adaptation_item[
                            "rgb"][0], "rgb")
                    augment_data(source_data_item, self.source_data.augmentation_cfg)

                    prepare_for_network(source_data_item, self.source_data.normalization_cfg)
                    source_data_item["rgb_original"] = tmp_center_img
                    output["source"].append(source_data_item)
            # --------------------

            # --------------------
            # Augmented version of the online target image
            augment_data(adaptation_item, self.target_data.augmentation_cfg)
            prepare_for_network(adaptation_item, self.target_data.normalization_cfg)
            output["target_augmented"] = [adaptation_item]
            # --------------------

            # --------------------
            # Add term if there is already a target buffer from a previous adaptation step
            extra = len(self.buffer_indices.get("prev_target", []))
            # Sample from the target replay buffer
            if self.target_num_samples > 0 and self.buffer_indices[
                "target"].sum().item() + extra > 0:
                output["target_replay"] = []
                buffer_indices = torch.where(self.buffer_indices["target"])[0].tolist()
                if extra > 0:
                    buffer_indices += self.buffer_indices["prev_target"]
                indices = self.target_sampler.choice(
                    buffer_indices, self.target_num_samples,
                    replace=self.target_num_samples > len(buffer_indices))
                for i in indices:
                    # Hardcoded continual learning
                    if i >= 10000:
                        target_item = self.prev_target_data.__getitem__(
                            i - 10000, do_augmentation=False, do_network_preparation=False,
                            return_only_rgb=True)
                    else:
                        target_item = self.target_data.__getitem__(
                            i, do_augmentation=False, do_network_preparation=False,
                            return_only_rgb=True)
                    tmp_center_img = deepcopy(target_item["rgb"][0])
                    augment_data(target_item, self.target_data.augmentation_cfg)
                    prepare_for_network(target_item, self.target_data.normalization_cfg)
                    target_item["rgb_original"] = tmp_center_img
                    output["target_replay"].append(target_item)
                    # Append another augmented version to target_augmented
                    if i >= 10000:
                        target_item = self.prev_target_data.__getitem__(
                            i - 10000, return_only_rgb=True, do_augmentation=True)
                    else:
                        target_item = self.target_data.__getitem__(
                            i, return_only_rgb=True, do_augmentation=True)
                    output["target_augmented"].append(target_item)
            # --------------------

            # --------------------
            # Create mixup samples with appropriate association
            if self.source_num_samples == 0 and self.cfg_mixup.general.active:
                raise RuntimeError("Mixup is active but source_num_samples equals to 0")
            if self.cfg_mixup.general.active:  # Active and source_num_samples > 0
                assert self.cfg_mixup.general.nof_samples == \
                       len(self.cfg_mixup.general.mixup_strategies), \
                    "The chosen number of mixup samples does not match size of specified " \
                    "mixup strategies in the config file"
                # +1 due to the online target image
                # assert self.cfg_mixup.general.nof_samples <= self.source_num_samples or \
                #        self.cfg_mixup.general.nof_samples <= (self.target_num_samples + 1)

                mixup_strategies = self.cfg_mixup.general.mixup_strategies.copy()

                indices_tgt, indices_src = [], []
                mix_counter = 0
                while mix_counter < self.cfg_mixup.general.nof_samples:
                    mix_counter += 1

                    # Initialize indices if not done yet or if all indices have been used
                    if not indices_src:
                        indices_src = list(range(self.source_num_samples))
                    if not indices_tgt:
                        indices_tgt = list(range(self.target_num_samples))

                    # Choose strategy and remove one appearance of it from the list
                    mixup_strategy = self.mixup_sampler.choice(mixup_strategies)
                    mixup_strategies.remove(mixup_strategy)

                    # Load source sample
                    idx_src = self.mixup_sampler.choice(indices_src)
                    sample_src = output["source"][idx_src]
                    indices_src.remove(idx_src)

                    # Load target sample and prioritize online image
                    if mix_counter == 1 or "target_replay" not in output:
                        sample_tgt = output["target"]
                        tgt_is_replay = 0
                    else:
                        idx_tgt = self.mixup_sampler.choice(indices_tgt)
                        sample_tgt = output["target_replay"][idx_tgt]
                        indices_tgt.remove(idx_tgt)
                        tgt_is_replay = 1

                    # Gather data required for mixup
                    mixup_item = self._get_data_for_mixup(sample_src, sample_tgt)
                    mixup_item["tgt_is_replay"] = tgt_is_replay
                    if mixup_strategy in output:
                        output[mixup_strategy].append(mixup_item)
                    else:
                        output[mixup_strategy] = [mixup_item]
            # -------------------

            # -------------------
            # Remove temporary variables
            for key in ["target", "target_replay", "source"]:
                if key in output:
                    if isinstance(output[key], list):
                        for item in output[key]:
                            item.pop("rgb_original", None)
                    else:
                        output[key].pop("rgb_original", None)
            # -------------------

        else:  # "val"
            output["target"].pop("rgb_original", None)
            output = output["target"]  # Just forward the target sample

        return output

    def _get_data_for_mixup(self, sample_src, sample_tgt):
        # Reassign variables for readability later on
        out = {}

        # Get rgb center images in window
        rgb_src = {"rgb": {
            0: transfer_histogram_style(sample_src["rgb_original"], sample_tgt["rgb_original"],
                                        "rgb")}}
        augment_data(rgb_src, self.source_data.augmentation_cfg)
        prepare_for_network(rgb_src, self.source_data.normalization_cfg)
        out["rgb_src"] = {0: rgb_src["rgb"][0]}
        out["rgb_tgt"] = {0: sample_tgt["rgb"][0]}

        # Get semantic source labels (target will be predicted later)
        semantic_src = torch.from_numpy(sample_src["semantic"]).unsqueeze(0)
        out["semantic_src"] = semantic_src

        # Get instance source labels (target will be predicted later)
        instance_src = torch.from_numpy(sample_src["instance"]).unsqueeze(0)
        out["instance_src"] = instance_src

        # Get source and target camera models
        out["camera_model_src"] = sample_src["camera_model"]
        out["camera_model_tgt"] = sample_tgt["camera_model"]

        return out

    def _initialize_samplers(self):
        if torch.utils.data.get_worker_info() is None:
            worker_id = 0
        else:
            worker_id = torch.utils.data.get_worker_info().id
        seed = self.samplers_seed + worker_id
        self.source_sampler = np.random.default_rng(seed=seed)
        self.target_sampler = np.random.default_rng(seed=seed)

    def initialize_source_buffer(self, source_data: TorchDataset, number_samples: Optional[int]):
        if self.mode != "train":
            print("\033[91m" + "WARNING: unable to initialize base data in val mode" + "\033[0m")
            return
        if number_samples is not None:
            assert number_samples >= self.source_num_samples
        assert source_data.stuff_classes == self.stuff_classes
        assert source_data.thing_classes == self.thing_classes

        self.source_data = source_data

        # Workers share the same buffer indices
        MAX_SOURCE_BUFFER_SIZE = len(self.source_data)  # The size has to be fixed in the beginning
        source_buffer = mp.Array(ctypes.c_int, MAX_SOURCE_BUFFER_SIZE)
        shared_source_buffer = np.ctypeslib.as_array(source_buffer.get_obj())
        self.buffer_indices["source"] = torch.from_numpy(shared_source_buffer)
        self.buffer_indices["source"][:] = 0  # Samples added to the buffer will be set to 1

        # Add to source buffer
        if number_samples is not None:
            if self.maximize_diversity:
                with open(self.source_data.class_distribution_file, "rb") as f:
                    data = pickle.load(f)
                    number_pixels = data["num_pixels"]
                    class_images = data["class_images"]
                    images_numbers_pixels = data["images_num_pixels"]

                # Compute frequency of each class in the dataset
                frequency = {c: number_pixels[c] / sum(number_pixels.values()) for c in
                             number_pixels.keys()}

                # Compute the sampling probability
                temperature = 0.01
                probability = {c: np.exp((1 - freq) / temperature) for c, freq in frequency.items()}
                probability = {c: prob / sum(probability.values()) for c, prob in
                               probability.items()}

                sampler = np.random.default_rng(seed=self.buffer_seed)
                while len(self.source_samples_indices) < number_samples:
                    sampled_class = sampler.choice(list(probability.keys()),
                                                   p=list(probability.values()))
                    possible_images = class_images[sampled_class]

                    # Sample based on number of pixels with this class
                    number_pixels_class = {image_id: images_numbers_pixels[image_id][sampled_class]
                                           for image_id in possible_images}
                    sampled_image_index = sampler.choice(possible_images,
                                                         p=list(number_pixels_class.values()) / sum(
                                                             number_pixels_class.values()))

                    # Random sampling
                    if sampled_image_index not in self.source_samples_indices:
                        self.source_samples_indices.append(sampled_image_index)

            else:
                # Add random samples
                rng = np.random.default_rng(seed=self.buffer_seed)
                self.source_samples_indices = rng.choice(len(self.source_data), number_samples,
                                                         replace=False).tolist()
            self.source_samples_indices.sort()

        else:
            # Add all samples
            self.source_samples_indices = list(range(len(self.source_data)))
        for i in self.source_samples_indices:
            self.buffer_indices["source"][i] = 1

    def add(self, index: int, image_features: Optional[Tensor] = None):
        if self.maximize_diversity:
            # pylint: disable=no-value-for-parameter
            assert image_features is not None

            flattened_features = image_features.mean(-1).mean(-1).cpu().numpy()
            if self.faiss_index is None:
                # Cosine similarity
                self.faiss_index = faiss.IndexIDMap(
                    faiss.index_factory(flattened_features.shape[1], "Flat",
                                        faiss.METRIC_INNER_PRODUCT))
            faiss.normalize_L2(flattened_features)  # The inner product becomes cosine similarity

            # Add term if there is already a target buffer from a previous adaptation step
            extra = len(self.buffer_indices.get("prev_target", []))

            # Only add if sufficiently different to existing samples
            if self.faiss_index.ntotal == 0:
                similarity = [[0]]
            else:
                similarity, _ = self.faiss_index.search(flattened_features, 1)
            if similarity[0][0] < self.similarity_threshold:
                self.faiss_index.add_with_ids(flattened_features, np.array([index]))
                self.buffer_indices["target"][index] = 1
                print(f"Added sample {index} to the target buffer | similarity {similarity[0][0]}")

                if self.buffer_indices["target"].sum() + extra > self.target_buffer_size:
                    # Maximize the diversity in the replay buffer
                    if self.distance_matrix is None:
                        features = self.faiss_index.index.reconstruct_n(0, self.faiss_index.ntotal)
                        dist_mat, matching = self.faiss_index.search(features,
                                                                     self.faiss_index.ntotal)
                        for i in range(self.faiss_index.ntotal):
                            dist_mat[i, :] = dist_mat[i, matching[i].argsort()]
                        self.distance_matrix = dist_mat
                        self.distance_matrix_indices = faiss.vector_to_array(
                            self.faiss_index.id_map)
                    else:
                        # Only update the elements that actually change
                        fill_up_index = np.argwhere(self.distance_matrix_indices < 0)[0, 0]
                        a, b = self.faiss_index.search(flattened_features, self.faiss_index.ntotal)
                        self.distance_matrix_indices[fill_up_index] = index
                        # ToDo: This is not correct, but it was used in the initial submission
                        # a = a[:, b.argsort()[:, self.distance_matrix_indices.argsort()][0]][0]
                        # This is the correct code
                        sorter = np.argsort(b[0])
                        sorter_idx = sorter[
                            np.searchsorted(b[0], self.distance_matrix_indices, sorter=sorter)]
                        a = a[:, sorter_idx][0]
                        self.distance_matrix[fill_up_index, :] = self.distance_matrix[:,
                                                                 fill_up_index] = a
                    # Subtract self-similarity
                    remove_index_tmp = np.argmax(
                        self.distance_matrix.sum(0) - self.distance_matrix.diagonal())
                    self.distance_matrix[:, remove_index_tmp] = self.distance_matrix[
                                                                remove_index_tmp, :] = -1
                    remove_index = self.distance_matrix_indices[remove_index_tmp]
                    self.distance_matrix_indices[remove_index_tmp] = -1
                    self.faiss_index.remove_ids(np.array([remove_index]))

                    # Hardcoded continual learning
                    if remove_index >= 10000:
                        self.buffer_indices["prev_target"].remove(remove_index)
                    else:
                        self.buffer_indices["target"][remove_index] = 0

                    print(f"Removed sample {remove_index} from the target buffer")

        else:
            self.buffer_indices["target"][index] = 1
            if self.buffer_indices["target"].sum() > self.target_buffer_size:
                buffer_indices = torch.where(self.buffer_indices["target"])[0].tolist()
                remove_index = self.buffer_remove_sampler.choice(buffer_indices)
                self.buffer_indices["target"][remove_index] = 0

    def save_state(self):
        target_samples_indices = []
        features = []
        for i in range(self.faiss_index.ntotal):
            target_samples_indices.append(self.faiss_index.id_map.at(i))
            features.append(self.faiss_index.index.reconstruct(i))
        with open("buffer_state.pkl", "wb") as f:
            pickle.dump({
                "target_samples_indices": target_samples_indices,
                "faiss_features": features,
                "target_data": self.target_data,
            }, f)

    def load_state(self):
        # pylint: disable=no-value-for-parameter, attribute-defined-outside-init
        with open("buffer_state.pkl", "rb") as f:
            state = pickle.load(f)
        self.prev_target_data = state["target_data"]
        target_samples_indices = state["target_samples_indices"]
        features = state["faiss_features"]

        self.buffer_indices["prev_target"] = [10000 + idx for idx in target_samples_indices]
        self.faiss_index = faiss.IndexIDMap(
            faiss.index_factory(features[0].size, "Flat", faiss.METRIC_INNER_PRODUCT))
        for idx, feature in zip(target_samples_indices, features):
            idx = 10000 + idx
            self.faiss_index.add_with_ids(feature.reshape(1, feature.size), np.array([idx]))

        if self.faiss_index.ntotal == self.target_buffer_size:
            features = self.faiss_index.index.reconstruct_n(0, self.faiss_index.ntotal)
            dist_mat, matching = self.faiss_index.search(features, self.faiss_index.ntotal)
            for i in range(self.faiss_index.ntotal):
                dist_mat[i, :] = dist_mat[i, matching[i].argsort()]
            self.distance_matrix_indices = np.append(faiss.vector_to_array(self.faiss_index.id_map),
                                                     [-1])
            self.distance_matrix = -np.ones(
                (self.distance_matrix_indices.size, self.distance_matrix_indices.size))
            self.distance_matrix[:self.distance_matrix_indices.size - 1,
            :self.distance_matrix_indices.size - 1] = dist_mat

    @property
    def stuff_classes(self) -> List[int]:
        return self.target_data.stuff_classes

    @property
    def thing_classes(self) -> List[int]:
        return self.target_data.thing_classes

    @property
    def ignore_classes(self) -> List[int]:
        return self.target_data.ignore_classes

    @property
    def num_classes(self) -> int:
        return self.target_data.num_classes

    @property
    def num_things(self) -> int:
        return self.target_data.num_things

    @property
    def num_stuff(self) -> int:
        return self.target_data.num_stuff
