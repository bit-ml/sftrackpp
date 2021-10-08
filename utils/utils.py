import math
import os

import jpeg4py
import numpy as np
import pi
import torch
from PIL import Image
from skimage.measure import regionprops

FRAME_W, FRAME_H = 854, 480


def jpeg4py_loader(path):
    """ Image reading using jpeg4py https://github.com/ajkxyz/jpeg4py"""
    try:
        return jpeg4py.JPEG(path).decode()
    except Exception as e:
        print('ERROR: Could not read image "{}"'.format(path))
        print(e)
        return None


def normalize(mask):
    maxe = mask.max()
    mine = mask.min()
    return (mask - mine) / (maxe - mine)


def iou(soft_mask, gt_mask, th):
    # pred_mask = torch.sigmoid(soft_mask)
    pred_mask = normalize(soft_mask)
    pred_mask_binary = pred_mask.ge(th).cuda().byte()
    gt_mask = gt_mask.byte()

    inters = torch.sum((gt_mask & pred_mask_binary))
    union = torch.sum((gt_mask | pred_mask_binary))
    J_davis = inters / (union.float() + 0.00001)
    return J_davis


def txt_to_array(txt_path):
    with open(txt_path, 'r') as fd:
        lines = fd.readlines()
    for i in range(len(lines)):
        lines[i] = lines[i].strip()
    return lines


def chain_generators(*iterables):
    for it in iterables:
        for element in it:
            yield element


def bbox2segm_map(bboxes_file,
                  video_name,
                  object_id,
                  bbox_img_save_path,
                  split_token='\t'):
    # bbox to map
    save_dir_imgs = "%s/%s_object_%d" % (bbox_img_save_path, video_name,
                                         object_id)
    if not os.path.exists(save_dir_imgs):
        os.system("mkdir -p %s" % save_dir_imgs)

    with open(bboxes_file, "r") as fd:
        lines_str = fd.readlines()
        for small_idx, line_str in enumerate(lines_str):
            x, y, w, h = [
                int(float(token.strip()))
                for token in line_str.split(split_token)
            ]

            # bbox to map
            segm_map = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)

            segm_map[y:y + h, x:x + w] = 1.
            np.save("%s/%05d.npy" % (save_dir_imgs, small_idx), segm_map)


def bboxfile_xywh_2segm_map(bboxes_file, video_name, object_id,
                            bbox_img_save_path, split_token, save_indexes, M0,
                            scale_factor, frame_shape):

    frame_h, frame_w = frame_shape
    frame_h = int(frame_h / scale_factor)
    frame_w = int(frame_w / scale_factor)

    # bbox to map
    save_dir_imgs = "%s/%s_object_%d" % (bbox_img_save_path, video_name,
                                         object_id)
    if not os.path.exists(save_dir_imgs):
        os.system("mkdir -p %s" % save_dir_imgs)

    with open(bboxes_file, "r") as fd:
        lines_str = fd.readlines()
        for central_frame_idx in save_indexes:
            for small_idx in range(central_frame_idx - M0,
                                   central_frame_idx + M0 + 1):
                line_str = lines_str[small_idx]
                x, y, w, h = [
                    int(float(token.strip()) / scale_factor)
                    for token in line_str.split(split_token)
                ]

                # bbox to map
                segm_map = np.zeros((frame_h, frame_w), dtype=np.uint8)

                segm_map[y:y + h, x:x + w] = 1.
                np.save("%s/%05d.npy" % (save_dir_imgs, small_idx), segm_map)


def bboxarray_xywh_2segm_map(bboxes_array, video_name, object_id,
                             bbox_img_save_path, save_indexes, M0,
                             scale_factor, frame_shape):
    frame_h, frame_w = frame_shape
    frame_h = int(frame_h / scale_factor)
    frame_w = int(frame_w / scale_factor)

    # bbox to map
    save_dir_imgs = "%s/%s_object_%d" % (bbox_img_save_path, video_name,
                                         object_id)
    if not os.path.exists(save_dir_imgs):
        os.system("mkdir -p %s" % save_dir_imgs)

    for central_frame_idx in save_indexes:
        for small_idx in range(central_frame_idx - M0,
                               central_frame_idx + M0 + 1):
            bbox = bboxes_array[small_idx]
            try:
                # x, y, w, h = int(bbox[0] / scale_factor), int(
                #     bbox[1] / scale_factor), int(bbox[2] / scale_factor), int(
                #         bbox[3] / scale_factor)
                x, y, w, h = [int(dim / scale_factor) for dim in bbox]
            except:
                if math.isnan(bbox[0]):
                    # bbox to map
                    segm_map = np.zeros((frame_h, frame_w), dtype=np.uint8)
                    np.save("%s/%05d.npy" % (save_dir_imgs, small_idx),
                            segm_map)
                    continue

            # bbox to map
            segm_map = np.zeros((frame_h, frame_w), dtype=np.uint8)

            segm_map[y:y + h, x:x + w] = 1.
            np.save("%s/%05d.npy" % (save_dir_imgs, small_idx), segm_map)


def label2bbox(labels_npy, video_names, small_indexes, bbox_coord_save_path,
               bbox_img_save_path):
    '''
        labels_npy: BS x H x W
    '''

    for idx in range(labels_npy.shape[0]):
        label_npy = labels_npy[idx]
        video_name = video_names[idx]
        small_idx = small_indexes[idx].item()

        # skip 0 - background
        objects_ids = sorted(np.unique(label_npy))[1:]

        for object_id in objects_ids:
            one_obj_mask = (label_npy == object_id).byte().numpy()
            props = regionprops(one_obj_mask)
            assert (len(props) == 1)
            sy, sx, ey, ex = props[0].bbox

            # bbox coords
            save_dir = "%s/%s_object_%d" % (bbox_coord_save_path, video_name,
                                            object_id)
            if not os.path.exists(save_dir):
                os.system("mkdir -p %s" % save_dir)
            with open("%s/%05d.txt" % (save_dir, small_idx), "w") as fd:
                fd.write("%d,%d,%d,%d,%d,%d" %
                         (sy, sx, ey, ex, FRAME_H, FRAME_W))

            # bbox to map
            save_dir = "%s/%s_object_%d" % (bbox_img_save_path, video_name,
                                            object_id)
            if not os.path.exists(save_dir):
                os.system("mkdir -p %s" % save_dir)
            segm_map = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)

            segm_map[sy:ey, sx:ex] = 1.
            np.save("%s/%05d.npy" % (save_dir, small_idx), segm_map)


def sfsegpp(sfsegppnet,
            input_masks,
            trackers_output,
            sfseg_params,
            num_iters=1):
    # trackers_output shape: BS x DTime x N_trackers X H x W
    bs, num_frames, n_trackers, h, w = trackers_output.shape
    if n_trackers > 1:
        nn_inp = trackers_output.view(bs * num_frames, n_trackers, h, w)
        input_masks = sfsegppnet(2 * nn_inp - 1).view(bs, num_frames, 1, h, w)
        input_masks = input_masks[:, :, 0]
    else:
        input_masks = 0.3308 - (2 * trackers_output[:, :, 0] - 1)

    # input_masks shape: BS X DTime X H X W
    torch.sigmoid_(input_masks)

    # initial binarization
    preds_pi = input_masks.clone()
    computed_features = input_masks.clone()

    # preds_pi shape: BS x DTime X H X W
    for i in range(num_iters):
        pi.one_iter_pi_batch(preds_pi, input_masks, computed_features,
                             sfseg_params)

    return preds_pi


def save_model(model, path):
    dir_name = os.path.dirname(path)
    if not os.path.exists(dir_name):
        os.system("mkdir -p %s" % dir_name)
    torch.save(model.state_dict(), path)
    print("Model saved at %s" % path)


def load_model(net_phase1, path):
    if os.path.exists(path):
        net_phase1.load_state_dict(torch.load(path))
    else:
        print("Could NOT load from path", path)
        import sys
        sys.exit(-1)


def load_mask_np(path, shape=None):
    if os.path.exists(path):
        mask = np.load(path)
    else:
        if shape is None:
            print(path, "EMPTY!")
            assert (False)
        else:
            mask = np.zeros(shape, dtype=np.float32)
    return mask


def resize_npy(inp_npy):
    shape = inp_npy.shape
    if len(shape) == 2:
        h, w = shape
        resized = np.array(
            Image.fromarray((inp_npy * 255).astype(np.uint8)).resize(
                (w // 2, h // 2)))
        return resized.astype(np.float32) / 255.
    if len(shape) == 3:
        c, h, w = shape
        assert (c == 3)
        resized = np.array(
            Image.fromarray(
                (inp_npy.transpose(1, 2, 0) * 255).astype(np.uint8)).resize(
                    (w // 2, h // 2)))
        return resized.astype(np.float32).transpose(2, 0, 1) / 255.

    assert (False)


# def first_non_zero_yx(scores2d):
#     non_zeros_yx = np.nonzero(scores2d)
#     y_max, x_max = np.unravel_index(flatten_max, scores2d.shape)
#     return y_max, x_max

# def last_non_zero_yx(scores2d):
#     flatten_max = np.argmax((scores2d!=0).cpu())
#     y_max, x_max = np.unravel_index(flatten_max, scores2d.shape)
#     return y_max, x_max
