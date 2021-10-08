import os
import sys

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

# our code
from datasets import davis17
from models.unet_model import UNetMedium

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from utils import utils

CHECKPOINTS_FOLDER = "checkpoints_sftrackpp"


def forward_batch(epoch, model, batch, batch_idx, device):
    # tracker_bbox_imgs: N trackers list: BS x H x W
    frame_rgb, gt_segm_img, gt_bbox_img, tracker_bbox_img = batch
    frame_rgb = frame_rgb.to(device=device)
    gt_segm_img = gt_segm_img.to(device=device).float()

    if epoch == 0 and batch_idx < 20:
        pseudogt_bbox_img = gt_bbox_img.to(device=device).float()
    else:
        pseudogt_bbox_img = tracker_bbox_img.to(device=device).float()

    netinp = torch.cat([frame_rgb, pseudogt_bbox_img[:, None]], axis=1)
    phase1_segm = model(netinp)

    # del netinp, frame_rgb, gt_bbox_img, tracker_bbox_img
    return phase1_segm, gt_segm_img


def train_phase1(epoch, model, data_loader, optimizer, scheduler, loss_fcn,
                 device):
    model.train()
    training_loss = 0

    for batch_idx, batch in enumerate(tqdm(data_loader)):
        phase1_segm, gt_segm_img = forward_batch(epoch, model, batch,
                                                 batch_idx, device)

        batch_loss = loss_fcn(phase1_segm.squeeze(1), gt_segm_img)

        crt_batch_loss = batch_loss.item()
        training_loss += crt_batch_loss
        batch_loss.backward()

        # Optimizer
        optimizer.step()
        optimizer.zero_grad()

        scheduler.step(crt_batch_loss)

    training_loss /= len(data_loader)
    return training_loss


def val_phase1(epoch, model, data_loader, loss_fcn, device):
    model.eval()
    val_loss = 0

    for batch_idx, batch in enumerate(tqdm(data_loader)):
        phase1_segm, gt_segm_img = forward_batch(epoch, model, batch,
                                                 batch_idx, device)

        batch_loss = loss_fcn(phase1_segm.squeeze(1), gt_segm_img)

        crt_batch_loss = batch_loss.item()
        val_loss += crt_batch_loss
        batch_loss.backward()

    val_loss /= len(data_loader)
    return val_loss


def main():
    n_epochs = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # trackers = ["dimp", "atom", "segm", "siamban", "siamrpnpp"]
    trackers = ["dimp"]

    net = UNetMedium(n_inp=4, n_outp=1, with_dropout=False)
    net = nn.DataParallel(net)
    net.train()
    net.to(device)

    ds_train = davis17.Davis17SampledTrackerDataset(trackers,
                                                    split_name="train")
    ds_val = davis17.Davis17SampledTrackerDataset(trackers, split_name="val")

    dl_train = DataLoader(ds_train,
                          batch_size=80,
                          shuffle=True,
                          num_workers=30)
    dl_val = DataLoader(ds_val, batch_size=80, shuffle=False, num_workers=20)

    optimizer = optim.SGD(net.parameters(),
                          lr=0.02,
                          weight_decay=1e-4,
                          nesterov=True,
                          momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer,
                                  patience=80,
                                  factor=.1,
                                  threshold=0.005,
                                  min_lr=1e-7,
                                  verbose=True)

    loss_fcn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(5.)).to(device)

    # train
    for epoch in range(n_epochs):
        train_phase1(epoch, net, dl_train, optimizer, scheduler, loss_fcn,
                     device)
        val_phase1(epoch, net, dl_val, loss_fcn, device)
        utils.save_model(net.module,
                         "%s/phase1_net1_basic.pth" % CHECKPOINTS_FOLDER)


if __name__ == '__main__':
    main()
