import random
import time

import numpy as np
from skimage.transform import resize
from torch.utils.data import Dataset
from utils import utils


class TestDatasetWrapper(Dataset):
    def __init__(self,
                 dataset,
                 trackers,
                 dataset_part=5,
                 start_idx=0,
                 end_idx=None,
                 just_ids=None):
        super(TestDatasetWrapper, self).__init__()
        self.name = dataset.__class__.__name__
        self.sequence_list = dataset.get_sequence_list()

        if dataset_part < 5:
            part_start_idx = int(
                len(self.sequence_list) * (dataset_part - 1) / 4)
            part_end_idx = int(len(self.sequence_list) * dataset_part / 4)
            self.sequence_list = self.sequence_list[
                part_start_idx:part_end_idx]

        if just_ids is not None:
            self.sequence_list = self.sequence_list[just_ids]
        elif end_idx is not None:
            self.sequence_list = self.sequence_list[:end_idx]
        self.sequence_list = self.sequence_list[start_idx:]
        self.trackers = trackers

        print("[%s] loaded" % (self.name))

    def __getitem__(self, index):
        video_info = self.sequence_list[index]
        return video_info

    def __len__(self):
        return len(self.sequence_list)


class MultiDataset(Dataset):
    def __init__(self, multi_datasets):
        super(MultiDataset, self).__init__()
        self.multi_datasets = multi_datasets
        self.ds_lenghts = [len(ds) for ds in multi_datasets]
        self.cumulative_sum = np.cumsum(self.ds_lenghts)

    def __getitem__(self, index):
        for i in range(len(self.cumulative_sum)):
            if self.cumulative_sum[i] <= index:
                continue
            ds_index = i
            ds = self.multi_datasets[ds_index]
            if i > 0:
                in_ds_index = index - self.cumulative_sum[i - 1]
            else:
                in_ds_index = index
            break
        return ds[in_ds_index]

    def __len__(self):
        return self.cumulative_sum[-1]


class TrainDatasetWrapper(Dataset):
    def __init__(self,
                 dataset,
                 trackers,
                 M0,
                 start_idx=0,
                 end_idx=None,
                 samples_per_video=-1):
        super(TrainDatasetWrapper, self).__init__()
        self.name = dataset.__class__.__name__
        # s = time.time()
        self.sequence_list = dataset.sequence_list
        # e = time.time()
        # print("time get_sequence_list", e - s)
        if end_idx is not None:
            self.sequence_list = self.sequence_list[:end_idx]
        self.sequence_list = self.sequence_list[start_idx:]

        dataset.sequence_list = self.sequence_list
        self.sequence_list = dataset.get_sequence_list()

        self.trackers = trackers
        self.samples_per_video = samples_per_video
        self.seq_frames_count = []
        self.preproc_path = "/data/sftrack-preprocessed/datasets/tracking/train/"

        # temporal margin (number of frames : -M0, 0, +M0)
        self.M0 = M0
        self.center_frame_indexes = []

        print("[%s] loaded" % (self.name), len(self.sequence_list))
        # s = time.time()
        for seq in self.sequence_list:
            if samples_per_video == -1:
                self.seq_frames_count.append(len(seq.frames) - 2 * self.M0)
            else:
                self.seq_frames_count.append(samples_per_video)

                indexes_fname = "%s/gt/bbox_img/%s_object_1/frames_saved.npy" % (
                    self.preproc_path, seq.name)
                save_indexes = np.load(indexes_fname)
                self.center_frame_indexes.append(save_indexes.tolist())
        self.frames_cumulative_sum = np.cumsum(self.seq_frames_count)
        # e = time.time()
        # print("time lists", e - s)

    def __getitem__(self, index):
        for i in range(len(self.frames_cumulative_sum)):
            if self.frames_cumulative_sum[i] <= index:
                continue
            seq_index = i
            seq = self.sequence_list[seq_index]
            if self.samples_per_video == -1:
                if i > 0:
                    central_frame_idx = index - self.frames_cumulative_sum[
                        i - 1] + self.M0
                else:
                    central_frame_idx = index + self.M0
            else:
                # central_frame_idx = random.randint(
                #     self.M0,
                #     len(seq.frames) - self.M0 - 1)
                pos_in_array = random.randint(
                    0,
                    len(self.center_frame_indexes[seq_index]) - 1)
                central_frame_idx = self.center_frame_indexes[seq_index][
                    pos_in_array]
            break

        rgb_frames = []
        gt_bbox_imgs = []
        tracker_bbox_imgs = []
        scale_factor = 1
        for i, frame_idx in enumerate(
                range(central_frame_idx - self.M0,
                      central_frame_idx + self.M0 + 1)):

            # 1. rgb & scale factor
            frame_path = seq.frames[frame_idx]
            frame_rgb = utils.jpeg4py_loader(frame_path).transpose(2, 0, 1)
            if i == 0:
                scale_fname = "%s/gt/bbox_img/%s_object_1/scale_factor.npy" % (
                    self.preproc_path, seq.name)
                scale_factor = np.load(scale_fname).item()
                shape = frame_rgb.shape[1:]
                new_shape = [int(dim / scale_factor) for dim in shape]

            if scale_factor != 1:
                frame_rgb_resized = resize(frame_rgb,
                                           (3, new_shape[0], new_shape[1]))
            else:
                frame_rgb_resized = frame_rgb
            frame_rgb_npy = (np.asarray(frame_rgb_resized) / 255).astype(
                np.float32)
            rgb_frames.append(frame_rgb_npy)

            # 2. gt_bbox_img
            gt_path = "%s/gt/bbox_img/%s_object_1/%05d.npy" % (
                self.preproc_path, seq.name, frame_idx)
            gt_bbox_img = utils.load_mask_np(gt_path)
            gt_bbox_imgs.append(gt_bbox_img)

            # 3. trackers_bbox_imgs
            tracker_bboxes = []
            for tracker_name in self.trackers:
                tracker_path = "%s/trackers/%s/bbox_img/%s_object_1/%05d.npy" % (
                    self.preproc_path, tracker_name, seq.name, frame_idx)
                tracker_pred = utils.load_mask_np(tracker_path)
                tracker_bboxes.append(tracker_pred)
            tracker_bbox_imgs.append(tracker_bboxes)

        rgb_frames = np.array(rgb_frames)
        gt_bbox_imgs = np.array(gt_bbox_imgs)[:, None]
        tracker_bbox_imgs = np.array(tracker_bbox_imgs)

        return rgb_frames, gt_bbox_imgs, tracker_bbox_imgs

    def __len__(self):
        return self.frames_cumulative_sum[-1]
