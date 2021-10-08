import glob
import os
import sys

import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# from our project
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from datasets import davis17
from utils import utils

RAW_DATA_PATH = "/data/saliency/davis2017"

HIGH_RES_PREPROCESS_PATH = "/data/sftrack-preprocessed/datasets/davis2017/full_size"
LOW_RES_PREPROCESS_PATH = "/data/sftrack-preprocessed/datasets/davis2017/240"


def process_gt_rgb_segm(raw_dataset_path):
    # read&save images
    for split_name in ["val", "train"]:
        ds = davis17.Davis17DatasetRaw(raw_dataset_path,
                                       split_name,
                                       save_it=True)
        dl = DataLoader(ds, batch_size=50, shuffle=False, num_workers=8)

        for idx, batch_data in enumerate(tqdm(dl)):
            # load dataloader to trigger the SAVE action
            frame_npy, label_npy, video_name, small_idx, large_idx = batch_data


def process_gt_bboxes(raw_dataset_path):
    for split_name in ["val", "train"]:
        ds = davis17.Davis17DatasetRaw(raw_dataset_path,
                                       split_name,
                                       save_it=False)
        dl = DataLoader(ds, batch_size=50, shuffle=False, num_workers=8)

        for batch_data in tqdm(dl):
            frame_npy, label_npy, video_name, small_idx, large_idx = batch_data
            bbox_coord_save_path = "%s/gt/bbox_coord/%s" % (
                HIGH_RES_PREPROCESS_PATH, split_name)
            bbox_img_save_path = "%s/gt/bbox_img/%s" % (
                HIGH_RES_PREPROCESS_PATH, split_name)
            utils.label2bbox(label_npy, video_name, small_idx,
                             bbox_coord_save_path, bbox_img_save_path)


def pytracking_pipeline(tracker_results,
                        tracker_name,
                        split_token,
                        suffix="object_"):
    for split_name in ["val", "train"]:
        # for split_name in ["val"]:
        ds = davis17.Davis17DatasetMetaData(split_name)
        dl = DataLoader(ds, batch_size=20, shuffle=False, num_workers=8)

        # bbox_coord_save_path = "%s/trackers/%s/bbox_coord/%s" % (
        #     HIGH_RES_PREPROCESS_PATH, tracker_name, split_name)
        bbox_img_save_path = "%s/trackers/%s/bbox_img/%s" % (
            HIGH_RES_PREPROCESS_PATH, tracker_name, split_name)

        for batch_data in tqdm(dl):
            video_name, object_id = batch_data
            print("video_name", video_name, object_id)
            for idx in range(len(video_name)):
                bboxes_file = "%s/%s_%s%d.txt" % (
                    tracker_results, video_name[idx], suffix, object_id[idx])

                utils.bbox2segm_map(bboxes_file,
                                    video_name[idx],
                                    object_id[idx],
                                    bbox_img_save_path,
                                    split_token=split_token)


def process_atom():
    tracker_results = "/data/tracking-vot/experts/pytracking/pytracking/tracking_results/atom/default_000"
    pytracking_pipeline(tracker_results, "atom", split_token="\t", suffix="")


def process_prdimp():
    tracker_results = "/data/tracking-vot/experts/pytracking/pytracking/tracking_results/dimp/dimp18_000/"
    pytracking_pipeline(tracker_results, "dimp", split_token="\t", suffix="")


def process_oceanplus():
    tracker_results = '/data/tracking-vot/experts/Ocean/TracKit/result/DAVIS2017/OceanPlusMMS/'

    for split_name in ["val", "train"]:
        ds = davis17.Davis17DatasetSegmOutput(tracker_results, split_name)
        dl = DataLoader(ds, batch_size=20, shuffle=False, num_workers=8)

        for batch_data in tqdm(dl):
            label_npy, video_name, small_idx = batch_data
            bbox_coord_save_path = "%s/trackers/oceanplus/bbox_coord/%s" % (
                HIGH_RES_PREPROCESS_PATH, split_name)
            bbox_img_save_path = "%s/trackers/oceanplus/bbox_img/%s" % (
                HIGH_RES_PREPROCESS_PATH, split_name)
            utils.label2bbox(label_npy, video_name, small_idx,
                             bbox_coord_save_path, bbox_img_save_path)


def process_siamban():
    # tracker_results = "/data/tracking-vot/experts/SiamBAN/siamban/results/davis2017/model"
    tracker_results = "/data/tracking-vot/experts/pytracking/pytracking/tracking_results/siamban/default_000"
    pytracking_pipeline(tracker_results,
                        "siamban",
                        split_token="\t",
                        suffix="")


def process_siamrpnpp():
    # tracker_results = "/data/tracking-vot/experts/SiamRPN++/results/davis2017/model"
    tracker_results = "/data/tracking-vot/experts/pytracking/pytracking/tracking_results/siamrpnpp/default_000"
    pytracking_pipeline(tracker_results,
                        "siamrpnpp",
                        split_token="\t",
                        suffix="")


def process_siamfcpp():
    tracker_results = "/data/tracking-vot/experts/SiamFC++/video_analyst/logs/DAVIS2017/sat_res18_davis17/baseline/results_multi"
    for split_name in ["val", "train"]:
        ds = davis17.Davis17DatasetSegmOutput(tracker_results, split_name)
        dl = DataLoader(ds, batch_size=20, shuffle=False, num_workers=8)

        for batch_data in tqdm(dl):
            label_npy, video_name, small_idx = batch_data
            bbox_coord_save_path = "%s/trackers/siamfcpp/bbox_coord/%s" % (
                HIGH_RES_PREPROCESS_PATH, split_name)
            bbox_img_save_path = "%s/trackers/siamfcpp/bbox_img/%s" % (
                HIGH_RES_PREPROCESS_PATH, split_name)
            utils.label2bbox(label_npy, video_name, small_idx,
                             bbox_coord_save_path, bbox_img_save_path)


def processd3s():
    tracker_results = "/data/tracking-vot/experts/pytracking/pytracking/tracking_results/segm/default_params_000"
    pytracking_pipeline(tracker_results, "segm", split_token="\t", suffix="")


def process_trackers():
    # # ok
    process_atom()
    # process_prdimp()
    # processd3s()
    # process_siamban()
    # process_siamrpnpp()

    # process_oceanplus()
    # process_siamfcpp()

    # not working
    # process_kys() - module cuda old
    # SiamR-CNN - tf old
    # LTMU - tf old
    pass


def process_gt():
    # process_gt_rgb_segm(RAW_DATA_PATH)
    # process_gt_bboxes(RAW_DATA_PATH)
    pass


def to_low_res():
    print("Resize to half")

    trackers = ["atom"]
    # trackers = ["atom", "dimp", "segm", "siamrpnpp", "siamban"]
    # , "oceanplus", "siamban", "siamfcpp", "siamrpnpp"

    # to_resize_folders = ["gt/bbox_img", "gt/rgb", "gt/segm_img"]
    to_resize_folders = []
    for tracker_name in trackers:
        to_resize_folders.append("trackers/%s/bbox_img" % tracker_name)

    for split_name in ["val", "train"]:
        # for split_name in ["val"]:
        print("Split", split_name)
        for sub_path in to_resize_folders:
            print("\tsub_path", sub_path)

            inp_folder = "%s/%s/%s/" % (HIGH_RES_PREPROCESS_PATH, sub_path,
                                        split_name)
            all_videos = next(os.walk(inp_folder))[1]
            for video_name in tqdm(all_videos):
                outp_folder = "%s/%s/%s/%s" % (
                    LOW_RES_PREPROCESS_PATH, sub_path, split_name, video_name)
                if not os.path.exists(outp_folder):
                    os.system("mkdir -p %s" % outp_folder)

                # all_npys = glob.glob("%s/%s/*.npy" % (inp_folder, video_name))
                all_npys = next(os.walk("%s/%s/" %
                                        (inp_folder, video_name)))[2]
                for npy_path in all_npys:
                    inp_npy = np.load("%s/%s/%s" %
                                      (inp_folder, video_name, npy_path))
                    out_npy = utils.resize_npy(inp_npy)
                    out_path = "%s/%s" % (outp_folder, npy_path)
                    np.save(out_path, out_npy)
                    # from PIL import Image
                    # Image.fromarray((out_npy*
                    #      255).astype(np.uint8)).save("abc_davis_preproc.png")
                    # Image.fromarray((inp_npy*
                    #      255).astype(np.uint8)).save("abc_large_scale_davis_preproc.png")


def main():
    # process_gt()
    # process_trackers()

    to_low_res()
    pass


if __name__ == '__main__':
    main()
    # for arg in sys.argv:
    #     print(arg)
