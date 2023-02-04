# Copyright (c) EEEM071, University of Surrey

import glob
import os.path as osp
import re

from .base import BaseImageDataset


class CUB200(BaseImageDataset):

    dataset_dir = "CUB200"

    def __init__(self, root="datasets", verbose=True, **kwargs):
        super().__init__(root)
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, "train")
        self.query_dir = osp.join(self.dataset_dir, "query")
        self.gallery_dir = osp.join(self.dataset_dir, "gallery")

        self.check_before_run()

        train = self.process_dir(self.train_dir, relabel=True)
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> CUB200 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        (
            self.num_train_pids,
            self.num_train_imgs,
            self.num_train_cams,
        ) = self.get_imagedata_info(self.train)
        (
            self.num_query_pids,
            self.num_query_imgs,
            self.num_query_cams,
        ) = self.get_imagedata_info(self.query)
        (
            self.num_gallery_pids,
            self.num_gallery_imgs,
            self.num_gallery_cams,
        ) = self.get_imagedata_info(self.gallery)

    def check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError(f'"{self.dataset_dir}" is not available')
        if not osp.exists(self.train_dir):
            raise RuntimeError(f'"{self.train_dir}" is not available')
        if not osp.exists(self.query_dir):
            raise RuntimeError(f'"{self.query_dir}" is not available')
        if not osp.exists(self.gallery_dir):
            raise RuntimeError(f'"{self.gallery_dir}" is not available')

    def process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, "*/*.jpg"))

        dataset = []
        # FIXME: pseudo camera id for CUB200
        for i, img_path in enumerate(img_paths):
            pid = int(img_path.split("/")[-2].split(".")[0]) - 1
            dataset.append((img_path, pid, i))

        return dataset
