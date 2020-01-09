import numpy as np
from itertools import product
import torch
from math import ceil


class AnchorGenerator:
    def __init__(self, image_size, scales, strides):
        self.image_size = image_size
        self.scales = scales
        self.strides = strides
        self.feature_maps_size = self._get_feature_maps_size()

    def _get_feature_maps_size(self):
        h, w = self.image_size
        feature_maps = []
        for stride in self.strides:
            feature_maps.append((ceil(h / stride), ceil(w / stride)))
        return feature_maps

    def __call__(self):
        anchors = []
        height, width = self.image_size

        for scales, stride, (h, w) in zip(self.scales, self.strides, self.feature_maps_size):
            for y, x in product(range(h), range(w)):
                for scale, ratio, n in scales:
                    ratio = np.sqrt(ratio)
                    anchor_w = scale / width * ratio
                    anchor_h = scale / height / ratio

                    dense_cx, dense_cy = [], []
                    for i in range(n):
                        dense_cx.append((x + (2 * i + 1) / (2 * n)) * stride / width)
                        dense_cy.append((y + (2 * i + 1) / (2 * n)) * stride / height)
                    for anchor_cx, anchor_cy in product(dense_cx, dense_cy):
                        anchors.append([anchor_cx, anchor_cy, anchor_w, anchor_h])

        anchors = torch.tensor(anchors).view(-1, 4)
        return anchors
