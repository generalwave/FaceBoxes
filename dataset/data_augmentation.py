import numpy as np
import numpy.random as npr
import cv2


def intersect(box_a, box_b):
    # box 为 minmax 坐标格式
    # [A,2] -> [A,1,2] -> [A,B,2]
    # [B,2] -> [1,B,2] -> [A,B,2]
    num_a = box_a.shape[0]
    num_b = box_b.shape[0]
    min_xy = np.maximum(np.repeat(box_a[:, np.newaxis, :2], repeats=num_b, axis=1),
                        np.repeat(box_b[np.newaxis, :, :2], repeats=num_a, axis=0))
    max_xy = np.minimum(np.repeat(box_a[:, np.newaxis, 2:], repeats=num_b, axis=1),
                        np.repeat(box_b[np.newaxis, :, 2:], repeats=num_a, axis=0))
    inter = np.clip(max_xy - min_xy, a_min=0, a_max=None)
    return np.prod(inter, axis=2)


def area(boxes):
    # boxes 为 minmax 坐标格式
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def jaccard(box_a, box_b, mode="iou"):
    # box 为 minmax 坐标格式
    # 如果是 IOF，则 b 表示 background
    # 实际就是计算 IOU，只是这里是笛卡尔积方式的 IOU
    inter = intersect(box_a, box_b)
    area_a = np.repeat(np.expand_dims(area(box_a), axis=1), repeats=inter.shape[1], axis=1)
    area_b = np.repeat(np.expand_dims(area(box_b), axis=0), repeats=inter.shape[0], axis=0)
    if mode == "iou":
        union = area_a + area_b - inter
    else:
        union = np.maximum(area_b, 1)
    return inter / union


def _pad_to_square(image, bgr_mean):
    height, width, _ = image.shape
    long_side = max(width, height)
    image_t = np.empty((long_side, long_side, 3), dtype=image.dtype)
    image_t[:, :] = bgr_mean
    image_t[0:0 + height, 0:0 + width] = image
    return image_t


def random_cropping(image, gt_boxes, gt_labels, input_size, clip, smear_small_face, bgr_mean):
    height, width, _ = image.shape
    short_side = min(width, height)

    # 最大尝试次数，防止死循环
    times = 250
    for _ in range(times):
        if npr.uniform(low=0, high=1) < 0.2:
            scale = 1
        else:
            scale = npr.uniform(low=0.3, high=1.0)
        w = int(scale * short_side)
        h = w

        if width == w:
            x = 0
        else:
            x = npr.randint(width - w)
        if height == h:
            y = 0
        else:
            y = npr.randint(height - h)
        roi = np.array([x, y, x + w, y + h])

        overlaps = jaccard(roi[np.newaxis], gt_boxes, mode="iof")
        # 至少将 gt 中一个框住都可以
        if not np.any(overlaps >= 1):
            continue

        # 获取 gt 中心点在候选框内的 gt
        cxcy = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2
        mask = np.all(np.logical_and(roi[:2] <= cxcy, cxcy < roi[2:]), axis=1)
        # 防止对原数据的更改
        boxes_t = gt_boxes[mask].copy()
        labels_t = gt_labels[mask].copy()
        # 这个应该不可能了，因为有上面的 continue 部分保护
        if boxes_t.shape[0] == 0:
            continue

        # 将针对原图的坐标，转换为针对候选框的坐标，预测点不进行截断
        if clip:
            boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
            boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:])
        boxes_t[:, :2] -= roi[:2]
        boxes_t[:, 2:] -= roi[:2]

        # 删除特别小的训练数据集
        if clip:
            clip_boxes = boxes_t
        else:
            clip_boxes = boxes_t.copy()
            clip_boxes[:, :2] = np.maximum(clip_boxes[:, :2], 0)
            clip_boxes[:, 2:] = np.minimum(clip_boxes[:, 2:], roi[2:] - roi[:2])
        b_w_t = (clip_boxes[:, 2] - clip_boxes[:, 0]) / w * input_size
        b_h_t = (clip_boxes[:, 3] - clip_boxes[:, 1]) / h * input_size
        mask = np.minimum(b_w_t, b_h_t) > 16.0
        result_boxes_t = boxes_t[mask]
        result_labels_t = labels_t[mask]
        if result_boxes_t.shape[0] == 0:
            continue

        # 得到切割好的图片，这里没有 copy 是因为后面原图片不会继续进行处理了
        image_t = image[roi[1]:roi[3], roi[0]:roi[2]]

        # 将特别小的脸涂为均值
        if smear_small_face:
            mask = np.minimum(b_w_t, b_h_t) <= 16.0
            for i in range(len(mask)):
                if mask[i]:
                    xmin, ymin, xmax, ymax = clip_boxes[i]
                    image_t[int(ymin):int(ymax), int(xmin):int(xmax)] = bgr_mean

        return image_t, result_boxes_t, result_labels_t, False

    # 将特别小的脸涂为均值
    if smear_small_face:
        b_w_t = (gt_boxes[:, 2] - gt_boxes[:, 0]) / width * input_size
        b_h_t = (gt_boxes[:, 3] - gt_boxes[:, 1]) / height * input_size
        mask = np.minimum(b_w_t, b_h_t) <= 16.0
        for i in range(len(mask)):
            if mask[i]:
                xmin, ymin, xmax, ymax = gt_boxes[i]
                image[int(ymin):int(ymax), int(xmin):int(xmax)] = bgr_mean

    return image, gt_boxes, gt_labels, True


def _convert(image, alpha=1, beta=0):
    tmp = image.astype(float) * alpha + beta
    tmp[tmp < 0] = 0
    tmp[tmp > 255] = 255
    image[:] = tmp


def color_distort(image):
    image = image.copy()

    if npr.randint(2):
        # brightness distortion
        if npr.randint(2):
            _convert(image, beta=npr.uniform(-32, 32))
        # contrast distortion
        if npr.randint(2):
            _convert(image, alpha=npr.uniform(0.5, 1.5))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # saturation distortion
        if npr.randint(2):
            _convert(image[:, :, 1], alpha=npr.uniform(0.5, 1.5))
        # hue distortion
        if npr.randint(2):
            tmp = image[:, :, 0].astype(int) + npr.randint(-18, 18)
            tmp %= 180
            image[:, :, 0] = tmp
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    else:
        # brightness distortion
        if npr.randint(2):
            _convert(image, beta=npr.uniform(-32, 32))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # saturation distortion
        if npr.randint(2):
            _convert(image[:, :, 1], alpha=npr.uniform(0.5, 1.5))
        # hue distortion
        if npr.randint(2):
            tmp = image[:, :, 0].astype(int) + npr.randint(-18, 18)
            tmp %= 180
            image[:, :, 0] = tmp
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        # contrast distortion
        if npr.randint(2):
            _convert(image, alpha=npr.uniform(0.5, 1.5))

    return image


def _mirror(image, boxes):
    _, width, _ = image.shape
    if npr.randint(2):
        image = image[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2]
    return image, boxes


def _resize_subtract_mean(image, insize, bgr_mean):
    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    interp_method = interp_methods[npr.randint(5)]
    image = cv2.resize(image, (insize, insize), interpolation=interp_method)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb_mean = (bgr_mean[2], bgr_mean[1], bgr_mean[0])
    image = image.astype(np.float32)
    image -= rgb_mean
    return image.transpose(2, 0, 1)


class DataAugmentation:
    def __init__(self, input_size, bgr_mean, clip, smear_small_face):
        self.input_size = input_size
        self.bgr_mean = bgr_mean
        self.clip = clip
        self.smear_small_face = smear_small_face

    def __call__(self, image, bboxes, labels):
        # 进行论文所述的变换
        image_t, boxes_t, labels_t, need_padding = random_cropping(image, bboxes, labels, self.input_size, self.clip,
                                                                   self.smear_small_face, self.bgr_mean)
        image_t = color_distort(image_t)
        # 需要在最后进行 padding 操作，因为上面的颜色变换会改变均值
        if need_padding:
            image_t = _pad_to_square(image_t, self.bgr_mean)
        image_t, boxes_t = _mirror(image_t, boxes_t)

        height, width, _ = image_t.shape
        boxes_t[:, 0::2] /= width
        boxes_t[:, 1::2] /= height

        image_t = _resize_subtract_mean(image_t, self.input_size, self.bgr_mean)

        return image_t, boxes_t, labels_t


# 验证集进行的数据增强，这里只需要进行 padding 就好了
class ValDataAugmentation:
    def __init__(self, input_size, bgr_mean):
        self.input_size = input_size
        self.bgr_mean = bgr_mean

    def __call__(self, image, bboxes, labels):
        image_t = _pad_to_square(image, self.bgr_mean)

        height, width, _ = image_t.shape
        bboxes[:, 0::2] /= width
        bboxes[:, 1::2] /= height

        image_t = _resize_subtract_mean(image_t, self.input_size, self.bgr_mean)

        return image_t, bboxes, labels
