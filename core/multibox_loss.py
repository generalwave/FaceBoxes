import torch
import torch.nn.functional as functional


class MultiBoxLoss:
    def __init__(self, num_classes, negpos_ratio, loc_weight):
        self.num_classes = num_classes
        self.negpos_ratio = negpos_ratio
        self.loc_weight = loc_weight

    def __call__(self, predicts, gt_labels, gt_boxes):
        p_labels, p_boxes = predicts
        batch_size = p_labels.size(0)

        # 获取正样本位置，保证背景的标签为 0
        positive = gt_labels > 0

        # 得到正负样本的数量
        num_pos = torch.sum(positive, dim=1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio * num_pos, min=0, max=positive.size(1) - 1)

        # 下面需要进行负样本的 OHEM，最关键为获取负例损失的排序
        loss_c = functional.cross_entropy(p_labels.view(-1, self.num_classes), gt_labels.view(-1), reduction="none")
        loss_c[positive.view(-1)] = 0
        loss_c = loss_c.view(batch_size, -1)
        # 先对负样本 loss 进行排序，得到 idx 的排序后，那么再对 idx 排序，得到的 idx 的 idx 就是原始的 rank
        _, loss_idx = torch.sort(loss_c, dim=1, descending=True)
        _, idx_rank = torch.sort(loss_idx, dim=1)
        negative = idx_rank < num_neg

        # 计算分类损失
        conf_filter = torch.gt(positive + negative, 0)
        conf_p = p_labels[conf_filter]
        conf_gt = gt_labels[conf_filter]
        loss_c = functional.cross_entropy(conf_p, conf_gt, reduction="sum")

        # 计算位置损失
        loc_p = p_boxes[positive].view(-1, 4)
        loc_gt = gt_boxes[positive].view(-1, 4)
        loss_l = functional.smooth_l1_loss(loc_p, loc_gt, reduction="sum")

        # 计算正样本的总个数
        n = torch.clamp(torch.sum(num_pos), min=1)

        return loss_c / n + self.loc_weight * loss_l / n
