import os
import random
import sys

import numpy as np
from tqdm import tqdm

# from our project
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from datasets import (got10kdataset, lasotdataset, nfsdataset, otbdataset,
                      trackingnetdataset, uavdataset)
from datasets.dataset import TestDatasetWrapper as wrap


def preprocess_sequence(seq):
    gt_dest = "/data/tracking-vot/experts/pytracking/pytracking/tracking_results/gt/gt_000/%s.txt" % (
        seq.name)
    with open(gt_dest, 'w') as fd:
        for listitem in seq.ground_truth_rect:
            fd.write('%s\n' % "\t".join(map(str, listitem)))


def preproc_test_datasets():
    print("TEST datasets")
    test_datasets = []
    test_datasets.append(wrap(otbdataset.OTBDataset(), trackers=None))
    test_datasets.append(wrap(nfsdataset.NFSDataset(), trackers=None))
    test_datasets.append(wrap(uavdataset.UAVDataset(), trackers=None))

    # test_datasets.append(
    #     wrap(got10kdataset.GOT10KDataset(split="test"), trackers))
    # test_datasets.append(
    #     wrap(trackingnetdataset.TrackingNetDataset(split="test"),
    #          trackers))
    preproc_path = "/data/sftrack-preprocessed/datasets/tracking/test/"

    for crt_dataset in test_datasets:
        print("[%s]..." % crt_dataset.name)
        for seq in tqdm(crt_dataset):
            preprocess_sequence(seq)


def main():
    preproc_test_datasets()


if __name__ == '__main__':
    main()
