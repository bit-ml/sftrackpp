import os
import random
import sys

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# from our project
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from datasets import (got10kdataset, lasotdataset, nfsdataset, otbdataset,
                      trackingnetdataset, uavdataset)
from datasets.dataset import TestDatasetWrapper as wrap
from utils import utils


def is_sequence_generated(seq, preproc_path):
    save_dir_path = "%s/gt/bbox_img/%s_object_1" % (preproc_path, seq.name)
    if not os.path.exists(save_dir_path):
        return False
    return True


def preprocess_sequence(seq, trackers, preproc_path, num_samples, max_dim,
                        with_GT):
    run_id = 0
    M0 = 2

    frame_shape = utils.jpeg4py_loader(seq.frames[0]).shape[:2]

    # save SCALE
    scale_factor = max(frame_shape) / max_dim
    if scale_factor < 1:
        scale_factor = 1

    gt_dir_name = "%s/gt/bbox_img/%s_object_1/" % (preproc_path, seq.name)
    if not os.path.exists(gt_dir_name):
        os.system("mkdir -p %s" % gt_dir_name)
    scale_fname = "%s/scale_factor.npy" % (gt_dir_name)
    np.save(scale_fname, scale_factor)
    # print("Num frames", len(seq.frames[0]), "shape",
    #       int(frame_shape[0] / scale_factor),
    #       int(frame_shape[1] / scale_factor))
    # return
    if num_samples == -1:
        save_indexes = range(0, len(seq.frames))
        # save_indexes = range(len(seq.frames) -2* M0 - 1, len(seq.frames)-M0-1)
        # save_indexes = range(0, M0)
        M0 = 0
    else:
        indexes_fname = "%s/frames_saved.npy" % (gt_dir_name)
        if os.path.exists(indexes_fname):
            save_indexes = np.load(indexes_fname)
        else:
            save_indexes = np.array([
                random.randint(M0,
                               len(seq.frames) - M0 - 1)
                for i in range(num_samples)
            ])
            np.save(indexes_fname, save_indexes)

    # gt bbox
    if with_GT:
        gt_bbox_img_save_path = "%s/gt/bbox_img/" % (preproc_path)
        utils.bboxarray_xywh_2segm_map(
            seq.ground_truth_rect,
            seq.name,
            object_id=1,
            bbox_img_save_path=gt_bbox_img_save_path,
            save_indexes=save_indexes,
            M0=M0,
            scale_factor=scale_factor,
            frame_shape=frame_shape)

    for tracker_name, tracker_config in trackers:
        tracker_source = "/data/tracking-vot/experts/pytracking/pytracking/tracking_results/%s/%s_%03d" % (
            tracker_name, tracker_config, run_id)
        bboxes_file = "%s/%s.txt" % (tracker_source, seq.name)
        # if not os.path.exists(bboxes_file):
        #     print("Skip", bboxes_file)
        #     continue

        bbox_img_save_path = "%s/trackers/%s/bbox_img/" % (preproc_path,
                                                           tracker_name)
        utils.bboxfile_xywh_2segm_map(bboxes_file,
                                      seq.name,
                                      object_id=1,
                                      bbox_img_save_path=bbox_img_save_path,
                                      split_token="\t",
                                      save_indexes=save_indexes,
                                      M0=M0,
                                      scale_factor=scale_factor,
                                      frame_shape=frame_shape)


def preproc_train_datasets():
    print("TRAIN datasets")
    trackers = [("dimp", "prdimp18"), ("atom", "default"),
                ("segm", "default_params"), ("siamban", "default"),
                ("siamrpnpp", "default")]

    train_datasets = []

    # train_datasets.append(
    #     wrap(got10kdataset.GOT10KDataset(split="train_few"),
    #          trackers))
    # train_datasets.append(
    #     wrap(lasotdataset.LaSOTDataset(split="train_few"),
    #          trackers,
    #          start_idx=221))
    # train_datasets.append(
    #     wrap(trackingnetdataset.TrackingNetDataset(split="train_few"),
    #          trackers))

    preproc_path = "/data/sftrack-preprocessed5/datasets/tracking/train/"

    for crt_dataset in train_datasets:
        print("[%s]..." % crt_dataset.name)
        for seq in tqdm(crt_dataset):
            preprocess_sequence(seq,
                                trackers,
                                preproc_path,
                                num_samples=2,
                                max_dim=480,
                                with_GT=True)


def preproc_test_datasets():
    print("TEST datasets")
    trackers = [("dimp", "prdimp18"), ("atom", "default"),
                ("segm", "default_params"), ("siamban", "default"),
                ("siamrpnpp", "default")]
    test_datasets = []
    # test_datasets.append(wrap(otbdataset.OTBDataset(), trackers))
    # test_datasets.append(wrap(nfsdataset.NFSDataset(), trackers))
    # test_datasets.append(wrap(uavdataset.UAVDataset(), trackers))

    # test_datasets.append(
    #     wrap(got10kdataset.GOT10KDataset(split="test"), trackers))
    test_datasets.append(
        wrap(lasotdataset.LaSOTDataset(split="test"), trackers))
    # test_datasets.append(
    #     wrap(trackingnetdataset.TrackingNetDataset(split="test"),
    #          trackers))
    preproc_path = "/data/sftrack-preprocessed/datasets/tracking/test/"

    for crt_dataset in test_datasets:
        print("[%s]..." % crt_dataset.name)
        for seq in tqdm(crt_dataset):
            preprocess_sequence(seq,
                                trackers,
                                preproc_path,
                                num_samples=-1,
                                max_dim=480,
                                with_GT=False)


# def check_missing_datasets():
#     print("TEST datasets")
#     trackers = [("dimp", "prdimp18"), ("atom", "default"),
#                 ("segm", "default_params")]
# ("siamban", "default")
#     test_datasets = []
#     # test_datasets.append(wrap(otbdataset.OTBDataset(), trackers))
#     # test_datasets.append(wrap(nfsdataset.NFSDataset(), trackers))

#     # test_datasets.append(wrap(uavdataset.UAVDataset(), trackers))
#     # test_datasets.append(
#     #     wrap(got10kdataset.GOT10KDataset(split="test"), trackers))

#     # test_datasets.append(
#     #     wrap(lasotdataset.LaSOTDataset(split="test"), trackers))
#     # test_datasets.append(
#     #     wrap(trackingnetdataset.TrackingNetDataset(split="test"), trackers))
#     preproc_path = "/data/sftrack-preprocessed/datasets/tracking/test/"

#     for crt_dataset in test_datasets:
#         print("[%s]..." % crt_dataset.name)
#         for seq in tqdm(crt_dataset):
#             is_generated = is_sequence_generated(seq, preproc_path)
#             if not is_generated:
#                 break

#     print("TRAIN datasets")
#     train_datasets = []
#     # train_datasets.append(
#     #     wrap(got10kdataset.GOT10KDataset(split="train_few"), trackers))
#     # train_datasets.append(
#     #     wrap(lasotdataset.LaSOTDataset(split="train_few"), trackers))
#     # train_datasets.append(
#     #     wrap(trackingnetdataset.TrackingNetDataset(split="train_few"),
#     #          trackers))
#     preproc_path = "/data/sftrack-preprocessed/datasets/tracking/train/"

#     for crt_dataset in train_datasets:
#         print("[%s]..." % crt_dataset.name)
#         for seq in tqdm(crt_dataset):
#             is_generated = is_sequence_generated(seq, preproc_path)
#             if not is_generated:
#                 break


def main():
    preproc_test_datasets()
    # preproc_train_datasets()
    # check_missing_datasets()


if __name__ == '__main__':
    main()
