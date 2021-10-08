import os
import sys

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

# our code
import pi
from datasets import (got10kdataset, lasotdataset, nfsdataset, otbdataset,
                      trackingnetdataset, uavdataset)
from datasets.dataset import MultiDataset
from datasets.dataset import TrainDatasetWrapper as wrap_train
from models.unet_model import UNetMedium, UNetSmall

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from utils import utils

CHECKPOINTS_FOLDER = "checkpoints_sftrackpp"


def save_all_nets(models, epoch):
    net_phase1, net_phase2, net_phase2_1, net_phase3 = models
    if not os.path.exists(CHECKPOINTS_FOLDER):
        os.system("mkdir -p %s" % CHECKPOINTS_FOLDER)

    utils.save_model(net_phase1.module,
                     "%s/phase3_net1_e%d.pth" % (CHECKPOINTS_FOLDER, epoch))
    utils.save_model(net_phase2,
                     "%s/phase3_net2_e%d.pth" % (CHECKPOINTS_FOLDER, epoch))
    utils.save_model(net_phase2_1.module,
                     "%s/phase3_net2_1_e%d.pth" % (CHECKPOINTS_FOLDER, epoch))
    utils.save_model(net_phase3.module,
                     "%s/phase3_net3_e%d.pth" % (CHECKPOINTS_FOLDER, epoch))


def forward_batch(num_trackers, models, sfseg_params, batch, batch_idx,
                  device):
    rgb_frames, gt_bbox_imgs, trackers_bbox_imgs = batch
    bs, num_frames, _, h, w = rgb_frames.shape
    M0 = sfseg_params["M0"]

    rgb_frames = rgb_frames.to(device=device)
    gt_bbox_imgs = gt_bbox_imgs.to(device=device).float()
    trackers_bbox_imgs = trackers_bbox_imgs.to(device=device).float()
    # rgb_frames:        BS x 2*M0 + 1 x channels     x H x W
    # trackers_bbox_imgs:              x num_trackers x
    # gt_bbox_imgs:                    x 1            x

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

    # Phase 3
    phase3_bbox = models[3](phase2_segm)
    return phase3_bbox, gt_bbox_imgs


def train_phase3(epoch, trackers, models, train_loader, sfseg_params,
                 optimizer, scheduler, loss_fcn, device):
    for model in models:
        model.train()

    training_loss = 0
    iou_all = 0

    num_trackers = len(trackers)
    M0 = sfseg_params["M0"]
    batch_sim_loss = 0
    iou_batch = 0

    num_train_samples = 200

    for batch_idx, batch in enumerate(tqdm(train_loader)):
        phase3_bbox, gt_bbox_imgs = forward_batch(num_trackers, models,
                                                  sfseg_params, batch,
                                                  batch_idx, device)

        batch_loss = loss_fcn(phase3_bbox, gt_bbox_imgs[:, M0])
        crt_batch_loss = batch_loss.item()
        training_loss += crt_batch_loss
        batch_loss.backward()

        batch_sim_loss += crt_batch_loss

        with torch.no_grad():
            iou = utils.iou(phase3_bbox[0, 0], gt_bbox_imgs[0, M0, 0], th=0.75)
            iou_batch += iou
            iou_all += iou

        # Optimizer
        if batch_idx % 10 == 9:
            batch_sim_loss /= 10
            iou_batch /= 10
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step(batch_sim_loss)

            batch_sim_loss = 0
            iou_batch = 0
        if batch_idx > num_train_samples:
            break

    training_loss /= batch_idx
    iou_all /= batch_idx
    return training_loss


def val_phase3(epoch, trackers, models, val_loader, sfseg_params, loss_fcn,
               device):
    for model in models:
        model.eval()

    val_loss = 0
    iou_all = 0
    num_trackers = len(trackers)
    M0 = sfseg_params["M0"]
    batch_sim_loss = 0
    iou_batch = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader)):
            phase3_bbox, gt_bbox_imgs = forward_batch(num_trackers, models,
                                                      sfseg_params, batch,
                                                      batch_idx, device)

            batch_loss = loss_fcn(phase3_bbox, gt_bbox_imgs[:, M0])
            crt_batch_loss = batch_loss.item()
            val_loss += crt_batch_loss
            batch_sim_loss += crt_batch_loss
            with torch.no_grad():
                iou = utils.iou(phase3_bbox[0, 0],
                                gt_bbox_imgs[0, M0, 0],
                                th=0.75)
                iou_batch += iou
                iou_all += iou

            # Optimizer
            if batch_idx % 10 == 9:
                batch_sim_loss /= 10
                iou_batch /= 10
                batch_sim_loss = 0
                iou_batch = 0

    val_loss /= len(val_loader)
    iou_all /= len(val_loader)
    return val_loss


def main():
    n_epochs = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # trackers = ["dimp", "atom", "segm", "siamban", "siamrpnpp"]
    trackers = ["dimp"]

    kernel_size = (3, 5, 5)
    M0 = kernel_size[0] // 2

    train_datasets = []
    train_datasets.append(
        wrap_train(got10kdataset.GOT10KDataset(split="train_few"),
                   trackers,
                   M0,
                   samples_per_video=1,
                   end_idx=500))
    train_datasets.append(
        wrap_train(trackingnetdataset.TrackingNetDataset(split="train_few"),
                   trackers,
                   M0=M0,
                   samples_per_video=1,
                   end_idx=500))
    train_composed_dataset = MultiDataset(train_datasets)

    val_datasets = []
    val_datasets.append(
        wrap_train(got10kdataset.GOT10KDataset(split="train_few"),
                   trackers,
                   M0,
                   samples_per_video=1,
                   start_idx=900))
    val_datasets.append(
        wrap_train(trackingnetdataset.TrackingNetDataset(split="train_few"),
                   trackers,
                   M0=M0,
                   samples_per_video=1,
                   start_idx=900))
    val_composed_dataset = MultiDataset(val_datasets)

    train_loader = DataLoader(train_composed_dataset,
                              batch_size=1,
                              shuffle=True,
                              num_workers=20)
    val_loader = DataLoader(val_composed_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=20)

    # models
    net_phase1 = UNetMedium(n_inp=4, n_outp=1, with_dropout=False)
    utils.load_model(net_phase1, "%s/phase2_net1.pth" % CHECKPOINTS_FOLDER)
    net_phase1.to(device)
    net_phase1 = nn.DataParallel(net_phase1)
    net_phase2 = nn.Conv2d(in_channels=len(trackers),
                           out_channels=1,
                           kernel_size=1,
                           bias=True)
    utils.load_model(net_phase2, "%s/phase2_net2.pth" % CHECKPOINTS_FOLDER)
    net_phase2.to(device)

    net_phase2_1 = UNetSmall(n_inp=1, n_outp=1, with_dropout=False)
    utils.load_model(net_phase2_1, "%s/phase2_net2_1.pth" % CHECKPOINTS_FOLDER)
    net_phase2_1.to(device)
    net_phase2_1 = nn.DataParallel(net_phase2_1)

    net_phase3 = UNetSmall(n_inp=1, n_outp=1, with_dropout=False)
    net_phase3.to(device)
    net_phase3 = nn.DataParallel(net_phase3)

    models = [net_phase1, net_phase2, net_phase2_1, net_phase3]

    all_params = utils.chain_generators(net_phase1.parameters(),
                                        net_phase2.parameters(),
                                        net_phase2_1.parameters(),
                                        net_phase3.parameters())

    optimizer = optim.SGD(all_params,
                          lr=1e-2,
                          weight_decay=1e-5,
                          nesterov=True,
                          momentum=0.9)

    scheduler = ReduceLROnPlateau(optimizer,
                                  patience=7,
                                  factor=0.1,
                                  threshold=0.005,
                                  min_lr=1e-5,
                                  verbose=True)

    loss_fcn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.)).to(device)

    sfseg_params = {}
    M0 = kernel_size[0] // 2
    sfseg_params["M0"] = M0
    sfseg_params["filter"] = pi.init_cnn_filter(kernel_size)
    sfseg_params["p"] = 0.1
    sfseg_params["alpha"] = 0.5

    for epoch in range(n_epochs):
        train_phase3(epoch, trackers, models, train_loader, sfseg_params,
                     optimizer, scheduler, loss_fcn, device)
        val_phase3(epoch, trackers, models, val_loader, sfseg_params, loss_fcn,
                   device)

        save_all_nets(models, epoch)


if __name__ == '__main__':
    main()
