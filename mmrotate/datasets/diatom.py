# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os.path as osp
import xml.etree.ElementTree as ET
from typing import List, Optional, Union

import mmcv
import numpy as np
import torch
from mmengine.dataset import BaseDataset
from mmengine.fileio import get, get_local_path, list_from_file

from mmrotate.registry import DATASETS
from mmrotate.structures.bbox import hbox2qbox


@DATASETS.register_module()
class DIATOMDataset(BaseDataset):
    """HRSC dataset for detection.

    Note: There are two evaluation methods for HRSC datasets, which can be
    chosen through ``classwise``. When ``classwise=False``, it means there
    is only one class; When ``classwise=True``, it means there are 31
    classes of ships.

    Args:
        img_subdir (str): Subdir where images are stored.
            Defaults to 'AllImages'.
        ann_subdir (str): Subdir where annotations are.
            Defaults to 'Annotations'.
        classwise (bool): Whether to use all 31 classes or only one class.
            Defaults to False.
        file_client_args (dict): Arguments to instantiate the
            corresponding backend in mmdet <= 3.0.0rc6. Defaults to None.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """

    METAINFO = {
        'classes':
        ('diatom',),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
         (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
         (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
         (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255),
         (0, 82, 0), (120, 166, 157), (110, 76, 0), (174, 57, 255),
         (199, 100, 0), (72, 0, 118), (255, 179, 240), (0, 125, 92),
         (209, 0, 151), (188, 208, 182), (0, 220, 176), (255, 99, 164),
         (92, 0, 73)],
    }

    def __init__(self,
                 img_subdir: str = 'images',
                 ann_subdir: str = 'xmls',
                 classwise: bool = False,
                 file_client_args: dict = None,
                 backend_args: dict = None,
                 **kwargs) -> None:
        self.img_subdir = img_subdir
        self.ann_subdir = ann_subdir
        self.classwise = classwise
        self.backend_args = backend_args
        if file_client_args is not None:
            raise RuntimeError(
                'The `file_client_args` is deprecated, '
                'please use `backend_args` instead, please refer to'
                'https://github.com/open-mmlab/mmdetection/blob/dev-1.x/configs/_base_/datasets/coco_detection.py'  # noqa: E501
            )
        super().__init__(**kwargs)

    @property
    def sub_data_root(self) -> str:
        """Return the sub data root."""
        return self.data_prefix.get('sub_data_root', '')

    def load_data_list(self) -> List[dict]:
        """Load annotation from XML style ann_file.

        Returns:
            list[dict]: Annotation info from XML file.
        """
        assert self._metainfo.get('classes', None) is not None, \
            'classes in `DIATOMDataset` can not be None.'

        data_list = []
        img_files = glob.glob(
                osp.join(self.sub_data_root, self.img_subdir, f'*.png'))
        for img_file in img_files:
            img_id = osp.split(img_file)[1][:-4]
            file_name = osp.join(self.sub_data_root, self.img_subdir,
                                img_id + '.png')
            xml_path = osp.join(self.sub_data_root, self.ann_subdir,
                                img_id + '.xml')

            raw_img_info = {}
            raw_img_info['img_id'] = img_id.replace('png', '')
            raw_img_info['file_name'] = file_name
            raw_img_info['xml_path'] = xml_path

            parsed_data_info = self.parse_data_info(raw_img_info)
            data_list.append(parsed_data_info)
        return data_list

    @property
    def bbox_min_size(self) -> Optional[str]:
        """Return the minimum size of bounding boxes in the images."""
        if self.filter_cfg is not None:
            return self.filter_cfg.get('bbox_min_size', None)
        else:
            return None

    def parse_data_info(self, img_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            img_info (dict): Raw image information, usually it includes
                `img_id`, `file_name`, and `xml_path`.

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        data_info = {}
        img_path = img_info['file_name']
        data_info['img_path'] = img_path
        data_info['img_id'] = img_info['img_id']
        data_info['xml_path'] = img_info['xml_path']

        # deal with xml file
        with get_local_path(
                img_info['xml_path'],
                backend_args=self.backend_args) as local_path:
            raw_ann_info = ET.parse(local_path)
        root = raw_ann_info.getroot()

        width = int(root.find('size').find('width').text)
        height = int(root.find('size').find('height').text)
        if width is None or height is None:
            img_bytes = get(img_path, backend_args=self.backend_args)
            img = mmcv.imfrombytes(img_bytes, backend='cv2')
            width, height = img.shape[:2]
            del img, img_bytes

        data_info['height'] = height
        data_info['width'] = width

        instances = []
        for obj in raw_ann_info.find('objects').findall('object'):
            instance = {}
            if self.classwise:
                label = int(obj.find('id').text)
            else:
                label = 0

            hbbox = np.array([[
                float(obj.find('bbox').find('xmin').text),
                float(obj.find('bbox').find('ymin').text),
                float(obj.find('bbox').find('xmax').text),
                float(obj.find('bbox').find('ymax').text)
            ]], dtype=np.float32)

            polygon = hbox2qbox(torch.from_numpy(hbbox)).numpy().tolist()[0]

            ignore = False
            if self.bbox_min_size is not None:
                assert not self.test_mode
                w = hbbox[0][2] - hbbox[0][0]
                h = hbbox[0][3] - hbbox[0][1]
                if w < self.bbox_min_size or h < self.bbox_min_size:
                    ignore = True
            if ignore:
                instance['ignore_flag'] = 1
            else:
                instance['ignore_flag'] = 0
            instance['bbox'] = polygon
            instance['bbox_label'] = label
            instances.append(instance)
        data_info['instances'] = instances
        return data_info

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False) \
            if self.filter_cfg is not None else False
        min_size = self.filter_cfg.get('min_size', 0) \
            if self.filter_cfg is not None else 0

        valid_data_infos = []
        for i, data_info in enumerate(self.data_list):
            width = data_info['width']
            height = data_info['height']
            if filter_empty_gt and len(data_info['instances']) == 0:
                continue
            if min(width, height) >= min_size:
                valid_data_infos.append(data_info)

        return valid_data_infos

    def get_cat_ids(self, idx: int) -> List[int]:
        """Get COCO category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            List[int]: All categories in the image of specified index.
        """

        instances = self.get_data_info(idx)['instances']
        return [instance['bbox_label'] for instance in instances]