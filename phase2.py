import os
import sys

import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

# our code
import pi
from datasets import davis17
from models.unet_model import UNetMedium, UNetSmall

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from utils import utils

CHECKPOINTS_FOLDER = "checkpoints_sftrackpp"


def forward_batch(num_trackers, models, sfseg_params, batch, batch_idx,
                  device):
    rgb_frames, gt_segm_imgs, trackers_bbox_imgs = batch
    bs, num_frames, chan, h, w = rgb_frames.shape
    M0 = sfseg_params["M0"]

    rgb_frames = rgb_frames.to(device=device)
    gt_segm_imgs = gt_segm_imgs.to(device=device).float()
    trackers_bbox_imgs = trackers_bbox_imgs.to(device=device).float()
    # rgb_frames:           BS x 2*M0 + 1 x channels     x H x W
    # trackers_bbox_imgs:                 x num_trackers x
    # gt_segm_imgs:                       x 1            x

    # Phase1. Output from all trackers
    phase1_segms = []
    for tr_idx in range(num_trackers):
        # concatenate on channels axis
        tracker_inp = torch.cat(
            [rgb_frames, trackers_bbox_imgs[:, :, tr_idx:tr_idx + 1]], axis=2)
        segm_pred1 = models[0](tracker_inp.view(bs * num_frames, 4, h,
                                                w)).view(
                                                    bs, num_frames, 1, h, w)
        phase1_segms.append(segm_pred1)

    # Phase 2. SFSeg
    input_masks = torch.cat(phase1_segms, axis=2)
    phase2_segm_interm = utils.sfsegpp(models[1],
                                       input_masks=input_masks,
                                       trackers_output=input_masks,
                                       sfseg_params=sfseg_params)[:, M0:M0 + 1]

    # phase2_segm_interm shape: BS x 1 x H x W
    phase2_segm = models[2](phase2_segm_interm)
    return phase2_segm, gt_segm_imgs


def train_phase2(epoch, trackers, models, data_loader, sfseg_params, optimizer,
                 scheduler, loss_fcn, device):
    for model in models:
        model.train()

    training_loss = 0
    num_trackers = len(trackers)
    M0 = sfseg_params["M0"]

    for batch_idx, batch in enumerate(tqdm(data_loader)):
        phase2_segm, gt_segm_imgs = forward_batch(num_trackers, models,
                                                  sfseg_params, batch,
                                                  batch_idx, device)
        batch_loss = loss_fcn(phase2_segm, gt_segm_imgs[:, M0])

        crt_batch_loss = batch_loss.item()
        training_loss += crt_batch_loss
        batch_loss.backward()

        # Optimizer
        optimizer.step()
        optimizer.zero_grad()

        scheduler.step(crt_batch_loss)
    training_loss /= len(data_loader)
    return training_loss


def val_phase2(epoch, trackers, models, data_loader, sfseg_params, loss_fcn,
               device):
    for model in models:
        model.eval()

    val_loss = 0
    num_trackers = len(trackers)
    M0 = sfseg_params["M0"]

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            phase2_segm, gt_segm_imgs = forward_batch(num_trackers, models,
                                                      sfseg_params, batch,
                                                      batch_idx, device)
            batch_loss = loss_fcn(phase2_segm, gt_segm_imgs[:, M0])

            crt_batch_loss = batch_loss.item()
            val_loss += crt_batch_loss

    val_loss /= len(data_loader)

    return val_loss


def main():
    n_epochs = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trackers = ["dimp", "atom", "segm", "siamban", "siamrpnpp"]
    trackers = ["dimp"]

    kernel_size = (3, 5, 5)
    M0 = kernel_size[0] // 2

    ds_train = davis17.Davis17AllTrackersDataset(trackers,
                                                 "train",
                                                 M0=M0,
                                                 samples_per_video=2)
    ds_val = davis17.Davis17AllTrackersDataset(trackers,
                                               "val",
                                               M0=M0,
                                               samples_per_video=2)

    dl_train = DataLoader(ds_train, batch_size=7, shuffle=True, num_workers=20)
    dl_val = DataLoader(ds_val, batch_size=30, shuffle=True, num_workers=20)

    # models
    net_phase1 = UNetMedium(n_inp=4, n_outp=1, with_dropout=False)
    utils.load_model(net_phase1,
                     "%s/phase1_net1_basic.pth" % CHECKPOINTS_FOLDER)
    net_phase1.to(device)
    net_phase1 = nn.DataParallel(net_phase1)
    net_phase2 = nn.Conv2d(in_channels=len(trackers),
                           out_channels=1,
                           kernel_size=1,
                           bias=True)
    net_phase2.to(device)
    net_phase2_1 = UNetSmall(n_inp=1, n_outp=1, with_dropout=False)
    net_phase2_1.to(device)
    net_phase2_1 = nn.DataParallel(net_phase2_1)

    models = [net_phase1, net_phase2, net_phase2_1]

    all_params = utils.chain_generators(net_phase1.parameters(),
                                        net_phase2.parameters(),
                                        net_phase2_1.parameters())
    optimizer = optim.SGD(all_params,
                          lr=0.02,
                          weight_decay=1e-4,
                          nesterov=True,
                          momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer,
                                  patience=10,
                                  factor=0.1,
                                  threshold=0.005,
                                  min_lr=1e-4,
                                  verbose=True)

    loss_fcn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(5.)).to(device)

    sfseg_params = {}
    M0 = kernel_size[0] // 2
    sfseg_params["M0"] = M0
    sfseg_params["filter"] = pi.init_cnn_filter(kernel_size)
    sfseg_params["p"] = 0.1
    sfseg_params["alpha"] = 0.5

    for epoch in range(n_epochs):
        train_phase2(epoch, trackers, models, dl_train, sfseg_params,
                     optimizer, scheduler, loss_fcn, device)
        val_phase2(epoch, trackers, models, dl_val, sfseg_params, loss_fcn,
                   device)

        utils.save_model(net_phase1.module,
                         "%s/phase2_net1.pth" % CHECKPOINTS_FOLDER)
        utils.save_model(net_phase2, "%s/phase2_net2.pth" % CHECKPOINTS_FOLDER)
        utils.save_model(net_phase2_1.module,
                         "%s/phase2_net2_1.pth" % CHECKPOINTS_FOLDER)


if __name__ == '__main__':
    main()
