from scipy.io import loadmat
import os
import torch.utils.data as data
import cv2
import numpy as np
import torch
from core.box_utils import match


class WiderMatParse(object):
    def __init__(self, anno_mat, class_id):
        self.f = loadmat(anno_mat)
        self.event_list = self.f['event_list']
        self.file_list = self.f['file_list']
        self.face_bbx_list = self.f['face_bbx_list']
        self.class_id = class_id

    def next(self):
        for event_idx, event in enumerate(self.event_list):
            directory = event[0][0]

            for file, bbx in zip(self.file_list[event_idx][0], self.face_bbx_list[event_idx][0]):
                f = file[0][0]
                path_of_image = os.path.join(directory, f + '.jpg')

                bboxes = []
                labels = []
                bbx0 = bbx[0]
                for i in range(bbx0.shape[0]):
                    xmin, ymin, xoffset, yoffset = bbx0[i]
                    xmax = xmin + xoffset
                    ymax = ymin + yoffset
                    bboxes.append([int(xmin), int(ymin), int(xmax), int(ymax)])
                    labels.append(self.class_id)

                yield path_of_image, np.array(bboxes).astype(np.float32), np.array(labels)


class WiderDataset(data.Dataset):
    def __init__(self, directory, anno_mat, class_id, augmentation, anchors, threshold, variance):
        parse = WiderMatParse(anno_mat, class_id)
        self.anno = self._getanno(parse)
        self.directory = directory
        self.augmentation = augmentation
        self.anchors = anchors
        self.threshold = threshold
        self.variance = variance

    @staticmethod
    def _getanno(parse):
        result = []
        for anno in parse.next():
            result.append(anno)
        return result

    def __len__(self):
        return len(self.anno)

    def __getitem__(self, idx):
        path, bboxes, labels = self.anno[idx]
        # 这里需要先进行拷贝，防止数据被篡改
        bboxes = bboxes.copy()
        labels = labels.copy()
        path = os.path.join(self.directory, path)
        image = cv2.imread(path, cv2.IMREAD_COLOR)

        image, bboxes, labels = self.augmentation(image, bboxes, labels)

        # 上面都是在 numpy 下进行的操作，下面需要转换到 torch 中处理
        image = torch.from_numpy(image)
        labels = torch.from_numpy(labels)
        bboxes = torch.from_numpy(bboxes)
        labels, boxes = match(labels, bboxes, self.anchors, self.threshold, self.variance)

        return image, labels, boxes
