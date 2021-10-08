import glob
import os
import random
import sys

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

# from our project
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from utils import utils

PREPROC_PATH = "/data/sftrack-preprocessed/datasets/davis2017/full_size/"
GT_PREPROC_PATH = "%s/gt/" % PREPROC_PATH

LOW_RES_PREPROC_PATH = "/data/sftrack-preprocessed/datasets/davis2017/240/"
LOW_RES_GT_PREPROC_PATH = "%s/gt/" % LOW_RES_PREPROC_PATH

FRAME_W, FRAME_H = 854, 480


class Davis17DatasetRaw(Dataset):
    def __init__(self, ds_path, split_name, save_it):
        super(Davis17DatasetRaw, self).__init__()
        assert (split_name in ["val", "train"])
        videos = utils.txt_to_array("%s/ImageSets/2017/%s.txt" %
                                    (ds_path, split_name))
        self.frames_info = []
        self.save_it = save_it
        self.split_name = split_name

        # videos = ["india"]
        for video_name in videos:
            frames_path_pattern = "%s/JPEGImages/480p/%s/*.jpg" % (ds_path,
                                                                   video_name)
            frames_path = sorted(glob.glob(frames_path_pattern))

            labels_path_pattern = "%s/Annotations_unsupervised/480p/%s/*.png" % (
                ds_path, video_name)
            labels_path = sorted(glob.glob(labels_path_pattern))

            if save_it:
                save_folder = "%s/rgb/%s/%s" % (GT_PREPROC_PATH, split_name,
                                                video_name)
                os.system("mkdir -p %s" % save_folder)

            for small_idx, frame_path in enumerate(frames_path):
                label_path = labels_path[small_idx]
                large_idx = int(frame_path.split("/")[-1].split(".")[0])
                self.frames_info.append(
                    (frame_path, label_path, video_name, small_idx, large_idx))

        print("[Davis17DatasetRaw]", ds_path, split_name,
              len(self.frames_info))

    def __getitem__(self, index):
        frame_path, label_path, video_name, small_idx, large_idx = self.frames_info[
            index]
        frame_npy = (np.asarray(Image.open(frame_path)) / 255).astype(
            np.float32)
        labels_npy = np.asarray(Image.open(label_path))

        # skip 0 - background
        objects_ids = sorted(np.unique(labels_npy))[1:]

        # crop to one size
        w_dim = frame_npy.shape[1]
        if w_dim > FRAME_W:
            start = (w_dim - FRAME_W) // 2
            frame_npy = frame_npy[:, start:start + FRAME_W]
            labels_npy = labels_npy[:, start:start + FRAME_W]

        if self.save_it:
            save_path = "%s/rgb/%s/%s/%05d.npy" % (
                GT_PREPROC_PATH, self.split_name, video_name, small_idx)
            np.save(save_path, frame_npy.transpose(2, 0, 1))
            # print("objects_ids", objects_ids)
            for object_id in objects_ids:
                segm_save_folder = "%s/segm_img/%s/%s_object_%d" % (
                    GT_PREPROC_PATH, self.split_name, video_name, object_id)
                segm_save_path = "%s/%05d.npy" % (segm_save_folder, small_idx)

                if not os.path.exists(segm_save_folder):
                    os.system("mkdir -p %s" % segm_save_folder)

                np.save(segm_save_path,
                        (labels_npy == object_id).astype(np.uint8))
        return frame_npy, labels_npy, video_name, small_idx, large_idx

    def __len__(self):
        return len(self.frames_info)


class Davis17DatasetMetaData(Dataset):
    def __init__(self, split_name):
        super(Davis17DatasetMetaData, self).__init__()
        assert (split_name in ["val", "train"])
        self.frames_info = []
        self.split_name = split_name

        rgb_path = "%s/rgb/%s/" % (GT_PREPROC_PATH, split_name)
        video_names = sorted([name for name in os.listdir(rgb_path)])

        for video_name in video_names:
            segm_objects_pattern = "%s/segm_img/%s/%s_object_*" % (
                GT_PREPROC_PATH, split_name, video_name)
            segm_objects_path = sorted(glob.glob(segm_objects_pattern))

            for obj_idx_path in segm_objects_path:
                obj_idx = int(obj_idx_path.split("/")[-1].split("_")[-1])
                self.frames_info.append((video_name, obj_idx))

        print("[Davis17DatasetMetaData] loaded", split_name,
              len(self.frames_info))

    def __getitem__(self, index):
        video_name, obj_idx = self.frames_info[index]

        return video_name, obj_idx

    def __len__(self):
        return len(self.frames_info)


class Davis17DatasetSegmOutput(Dataset):
    def __init__(self, ds_path, split_name):
        super(Davis17DatasetSegmOutput, self).__init__()
        assert (split_name in ["val", "train"])
        videos = utils.txt_to_array(
            "/data/saliency/davis2017/ImageSets/2017/%s.txt" % (split_name))
        self.frames_info = []
        self.split_name = split_name

        # videos = ["india"]
        for video_name in videos:
            labels_path_pattern = "%s/%s/*.png" % (ds_path, video_name)
            labels_path = sorted(glob.glob(labels_path_pattern))

            for small_idx, label_path in enumerate(labels_path):
                self.frames_info.append((label_path, video_name, small_idx))

        print("[Davis17DatasetRaw]", ds_path, split_name,
              len(self.frames_info))

    def __getitem__(self, index):
        label_path, video_name, small_idx = self.frames_info[index]
        labels_npy = np.asarray(Image.open(label_path))

        # crop to one size
        w_dim = labels_npy.shape[1]
        if w_dim > FRAME_W:
            start = (w_dim - FRAME_W) // 2
            labels_npy = labels_npy[:, start:start + FRAME_W]

        return labels_npy, video_name, small_idx

    def __len__(self):
        return len(self.frames_info)


##### DAVIS training
class Davis17AllTrackersDataset(Dataset):
    def __init__(self, trackers, split_name, M0, samples_per_video=-1):
        super(Davis17AllTrackersDataset, self).__init__()
        self.M0 = M0
        self.trackers = trackers
        self.preproc_path = LOW_RES_PREPROC_PATH
        videos_path = "%s/gt/segm_img/%s/" % (self.preproc_path, split_name)
        self.seq_names = sorted([name for name in os.listdir(videos_path)])
        self.seq_used_frames_count = []
        self.seq_frames = []
        self.samples_per_video = samples_per_video
        self.split_name = split_name

        for video_name in self.seq_names:
            orig_rgb_name = video_name[:video_name.find("object") - 1]
            frames_path_pattern = "%s/gt/rgb/%s/%s/*.npy" % (
                self.preproc_path, split_name, orig_rgb_name)
            frames_path = sorted(glob.glob(frames_path_pattern))

            self.seq_frames.append(frames_path)

            if samples_per_video == -1:
                self.seq_used_frames_count.append(
                    len(frames_path) - 2 * self.M0)
            else:
                self.seq_used_frames_count.append(samples_per_video)
        self.frames_cumulative_sum = np.cumsum(self.seq_used_frames_count)

        print("[Davis17AllTrackersDataset] loaded", split_name,
              self.frames_cumulative_sum[-1])

    def __getitem__(self, index):
        for i in range(len(self.frames_cumulative_sum)):
            if self.frames_cumulative_sum[i] <= index:
                continue
            seq_index = i
            seq_name = self.seq_names[seq_index]
            seq_frames = self.seq_frames[seq_index]
            if self.samples_per_video == -1:
                if i > 0:
                    central_frame_idx = index - self.frames_cumulative_sum[
                        i - 1] + self.M0
                else:
                    central_frame_idx = index + self.M0
            else:
                central_frame_idx = random.randint(
                    self.M0,
                    len(seq_frames) - self.M0 - 1)
            break

        rgb_frames = []
        gt_segm_imgs = []
        tracker_bbox_imgs = []

        # print(seq_name, central_frame_idx, len(seq_frames))
        for frame_idx in range(central_frame_idx - self.M0,
                               central_frame_idx + self.M0 + 1):
            # 1. rgb
            frame_path = seq_frames[frame_idx]
            # frame_rgb = jpeg4py_loader(frame_path).transpose(2, 0, 1)
            # frame_rgb_npy = (np.asarray(frame_rgb) / 255).astype(np.float32)
            frame_rgb_npy = np.load(frame_path)
            rgb_frames.append(frame_rgb_npy)
            mask_shape = frame_rgb_npy.shape[1:]

            # 2. gt_segm_imgs
            gt_path = "%s/gt/segm_img/%s/%s/%05d.npy" % (
                self.preproc_path, self.split_name, seq_name, frame_idx)
            gt_segm_img = utils.load_mask_np(gt_path, mask_shape)
            gt_segm_imgs.append(gt_segm_img)

            # 3. trackers_bbox_imgs
            tracker_bboxes = []
            for tracker_name in self.trackers:
                tracker_path = "%s/trackers/%s/bbox_img/%s/%s/%05d.npy" % (
                    self.preproc_path, tracker_name, self.split_name, seq_name, frame_idx)
                tracker_pred = utils.load_mask_np(tracker_path)
                tracker_bboxes.append(tracker_pred)
            tracker_bbox_imgs.append(tracker_bboxes)

        rgb_frames = np.array(rgb_frames)
        gt_segm_imgs = np.array(gt_segm_imgs)[:, None]
        tracker_bbox_imgs = np.array(tracker_bbox_imgs)

        return rgb_frames, gt_segm_imgs, tracker_bbox_imgs

    def __len__(self):
        return self.frames_cumulative_sum[-1]


class Davis17SampledTrackerDataset(Dataset):
    def __init__(self, trackers, split_name):
        super(Davis17SampledTrackerDataset, self).__init__()
        assert (split_name in ["val", "train"])
        self.frames_info = []
        self.split_name = split_name

        rgb_path = "%s/rgb/%s/" % (LOW_RES_GT_PREPROC_PATH, split_name)
        video_names = sorted([name for name in os.listdir(rgb_path)])

        for video_name in video_names:
            frames_path_pattern = "%s/rgb/%s/%s/*.npy" % (
                LOW_RES_GT_PREPROC_PATH, split_name, video_name)
            frames_path = sorted(glob.glob(frames_path_pattern))

            segm_objects_pattern = "%s/segm_img/%s/%s_object_*" % (
                LOW_RES_GT_PREPROC_PATH, split_name, video_name)
            segm_objects_path = sorted(glob.glob(segm_objects_pattern))

            bbox_objects_pattern = "%s/bbox_img/%s/%s_object_*" % (
                LOW_RES_GT_PREPROC_PATH, split_name, video_name)
            bbox_objects_path = sorted(glob.glob(bbox_objects_pattern))

            trackers_bbox_path = []
            for tracker_name in trackers:
                tracker_bbox_objects_pattern = "%s/trackers/%s/bbox_img/%s/%s_object_*" % (
                    LOW_RES_PREPROC_PATH, tracker_name, split_name, video_name)
                tracker_bbox_objects_path = sorted(
                    glob.glob(tracker_bbox_objects_pattern))
                trackers_bbox_path.append(tracker_bbox_objects_path)
            trackers_bbox_path = np.array(trackers_bbox_path)

            for small_idx, frame_path in enumerate(frames_path):
                for obj_idx, _ in enumerate(segm_objects_path):
                    gt_segm_img = "%s/%05d.npy" % (segm_objects_path[obj_idx],
                                                   small_idx)
                    gt_bbox_img = "%s/%05d.npy" % (bbox_objects_path[obj_idx],
                                                   small_idx)

                    trackers_bbox_img = []
                    trackers_path = trackers_bbox_path[:, obj_idx]
                    for tracker_path in trackers_path:
                        tracker_bbox_img_path = "%s/%05d.npy" % (tracker_path,
                                                                 small_idx)
                        trackers_bbox_img.append(tracker_bbox_img_path)
                    trackers_bbox_img = np.array(trackers_bbox_img)

                    # if os.path.exists(gt_segm_img):
                    self.frames_info.append((frame_path, gt_segm_img,
                                             gt_bbox_img, trackers_bbox_img))

        print("[Davis17SampledTrackerDataset] loaded", split_name,
              len(self.frames_info))

    def __getitem__(self, index):
        frame_path, segm_img_path, bbox_img_path, trackers_bbox_img = self.frames_info[
            index]
        frame_rgb = np.load(frame_path)
        mask_shape = frame_rgb.shape[1:]

        gt_segm_img = utils.load_mask_np(segm_img_path, mask_shape)
        gt_bbox_img = utils.load_mask_np(bbox_img_path, mask_shape)

        rand_tracker_idx = random.randint(0, trackers_bbox_img.shape[0] - 1)
        tracker_bbox_img = utils.load_mask_np(
            trackers_bbox_img[rand_tracker_idx], mask_shape)

        return frame_rgb, gt_segm_img, gt_bbox_img, tracker_bbox_img

    def __len__(self):
        return len(self.frames_info)
