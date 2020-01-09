import torch


def to_minmax_coordinates(boxes):
    cxcy = boxes[:, :2]
    wh = boxes[:, 2:]
    return torch.cat((cxcy - wh / 2, cxcy + wh / 2), dim=1)


def to_center_coordinates(boxes):
    tl = boxes[:, :2]
    br = boxes[:, 2:]
    return torch.cat(((br + tl) / 2, br - tl), dim=1)


def intersect(box_a, box_b):
    # box 为 minmax 坐标格式
    # [A,2] -> [A,1,2] -> [A,B,2]
    # [B,2] -> [1,B,2] -> [A,B,2]
    num_a = box_a.size(0)
    num_b = box_b.size(0)
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(num_a, num_b, 2),
                       box_b[:, :2].unsqueeze(0).expand(num_a, num_b, 2))
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(num_a, num_b, 2),
                       box_b[:, 2:].unsqueeze(0).expand(num_a, num_b, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def area(boxes):
    # boxes 为 minmax 坐标格式
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def jaccard(box_a, box_b):
    # box 为 minmax 坐标格式
    # 实际就是计算 IOU，只是这里是笛卡尔积方式的 IOU
    inter = intersect(box_a, box_b)
    area_a = area(box_a).unsqueeze(1).expand_as(inter)
    area_b = area(box_b).unsqueeze(0).expand_as(inter)
    union = area_a + area_b - inter
    return inter / union


def encode(boxes, anchors, variances):
    # boxes 为 minmax 坐标格式
    # anchors 为 center 坐标格式
    # variances 分别为中心点方差、长宽方差
    g_cxcy = (boxes[:, 2:] + boxes[:, :2]) / 2 - anchors[:, :2]
    g_cxcy /= (anchors[:, 2:] * variances[0])

    g_wh = (boxes[:, 2:] - boxes[:, :2]) / anchors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]

    return torch.cat([g_cxcy, g_wh], dim=1)


def decode(loc, anchors, variances):
    # loc 为预测的结果
    # anchors 为 center 坐标格式
    # variances 分别为中心点方差、长宽方差
    p_cxcy = loc[:, :2] * anchors[:, 2:] * variances[0] + anchors[:, :2]
    p_wh = torch.exp(loc[:, 2:] * variances[1]) * anchors[:, 2:]

    boxes = torch.cat([p_cxcy, p_wh], dim=1)
    return to_minmax_coordinates(boxes)


def match(gt_labels, gt_boxes, anchors, threshold, variances):
    # ground_truth 为真实位置，gt 为计算位置
    # 得到每个 ground_truth 所有 anchors 的 iou
    overlaps = jaccard(gt_boxes, to_minmax_coordinates(anchors))
    # 得到与每个 ground_truth 的 iou 最大的 anchor，与该 anchor 对应的 ground_truth 就是该 anchor 待回归的 gt
    best_anchor_iou, best_anchor_idx = overlaps.max(dim=1)
    # 忽略掉 ground_truth 与 anchor 最大 iou 都小于阈值的 ground_truth
    best_anchor_filter = best_anchor_idx[best_anchor_iou >= 0.2]

    # 得到与每个 anchor 的 iou 最大的 ground_truth，与该 anchor 对应的 ground_truth 就是该 anchor 候选回归的 gt
    best_gt_iou, best_gt_idx = overlaps.max(dim=0)
    # 保证与每个 ground_truth 最大 iou 的 anchor 的 gt 为该 ground_truth
    best_gt_iou = best_gt_iou.index_fill(dim=0, index=best_anchor_filter, value=1)
    for i in range(best_anchor_idx.size(0)):
        best_gt_idx[best_anchor_idx[i]] = i

    # 得到每个 anchor 对应的 gt
    conf = gt_labels[best_gt_idx]
    # 背景的标签必须为 0
    conf[best_gt_iou < threshold] = 0
    loc = gt_boxes[best_gt_idx]
    loc = encode(loc, anchors, variances)

    return conf, loc
