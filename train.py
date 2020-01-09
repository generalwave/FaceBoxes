import torch
import torch.utils.data as data
from tqdm import tqdm
from time import time
from models.faceboxes import FaceBoxes
from core.multibox_loss import MultiBoxLoss
from core.anchor_generator import AnchorGenerator
from dataset.wider import WiderDataset
from dataset.data_augmentation import DataAugmentation, ValDataAugmentation
from config import CONFIG, CLASSES_NAME_ID
from core.learning_rate_adjust import LearningRateAdjust
import os


def image_loader():
    face_class_id = CLASSES_NAME_ID["face"]
    batch_size = CONFIG["batch_size"]
    thread_num = CONFIG["thread_num"]
    # AnchorGenerator 的参数
    image_size = CONFIG["image_size"]
    scales = CONFIG["scales"]
    strides = CONFIG["strides"]
    # DataAugmentation 参数
    input_size = CONFIG["input_size"]
    bgr_mean = CONFIG["bgr_mean"]
    clip = CONFIG["clip"]
    smear_small_face = CONFIG["smear_small_face"]
    # Dataset 参数
    train_directory = CONFIG["train_directory"]
    train_mat = CONFIG["train_mat"]
    val_directory = CONFIG["val_directory"]
    val_mat = CONFIG["val_mat"]
    threshold = CONFIG["threshold"]
    variance = CONFIG["variance"]

    anchor_generator = AnchorGenerator(image_size, scales, strides)
    anchors = anchor_generator()

    train_augmentation = DataAugmentation(input_size, bgr_mean, clip, smear_small_face)
    train_dataset = WiderDataset(train_directory, train_mat, face_class_id, train_augmentation,
                                 anchors, threshold, variance)
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                       num_workers=thread_num, drop_last=True)

    if val_mat is not None:
        val_augmentation = ValDataAugmentation(input_size, bgr_mean)
        val_dataset = WiderDataset(val_directory, val_mat, face_class_id, val_augmentation,
                                   anchors, threshold, variance)
        val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size,
                                         num_workers=thread_num, drop_last=True)
    else:
        val_dataloader = None

    return train_dataloader, val_dataloader


def train_one_epoch(model, device, loader, optimizer, criterion):
    model.train()

    total_num = 0
    total_loss = 0

    with tqdm(total=len(loader)) as pbar:
        for images, labels, boxes in loader:
            batch_size = images.size(0)
            images = images.to(device)
            labels = labels.to(device)
            boxes = boxes.to(device)

            predicts = model(images)

            optimizer.zero_grad()
            loss = criterion(predicts, labels, boxes)
            loss.backward()
            optimizer.step()

            total_num += batch_size
            total_loss += loss.item() * batch_size

            pbar.update(1)

    if total_num == 0:
        total_loss = 0
    else:
        total_loss = total_loss / total_num

    return total_loss


def val_one_epoch(model, device, loader,  criterion):
    model.eval()

    total_num = 0
    total_loss = 0

    with torch.no_grad():
        for images, labels, boxes in loader:
            batch_size = images.size(0)
            labels = labels.to(device)
            images = images.to(device)
            boxes = boxes.to(device)

            predicts = model(images)
            loss = criterion(predicts, labels, boxes)

            total_num += batch_size
            total_loss += loss.item() * batch_size

    if total_num == 0:
        total_loss = 0
    else:
        total_loss = total_loss / total_num

    return total_loss


def train_net(model, device, max_epoch, save_directory):
    lr = CONFIG["lr"]
    momentum = CONFIG["momentum"]
    weight_decay = CONFIG["weight_decay"]
    # MultiBoxLoss 的参数
    num_classes = CONFIG["num_classes"]
    negpos_ratio = CONFIG["negative_positive_ratio"]
    loc_weight = CONFIG["location_weight"]
    # LearningRateAdjust 参数
    epoches = CONFIG["epoches"]
    gamma = CONFIG["gamma"]

    criterion = MultiBoxLoss(num_classes, negpos_ratio, loc_weight)

    train_loader, val_loader = image_loader()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = LearningRateAdjust(epoches, lr, gamma, optimizer)

    for i in range(max_epoch):
        start = time()
        t_loss = train_one_epoch(model, device, train_loader, optimizer, criterion)
        if val_loader is not None:
            v_loss = val_one_epoch(model, device, val_loader, criterion)
        else:
            v_loss = t_loss
        end = time()

        msg = "t_loss:%f\tv_loss:%f\ttime:%f\tepoch:%d" % (t_loss, v_loss, end - start, i)
        print(msg)

        scheduler(i)

        params = model.state_dict()
        save_path = os.path.join(save_directory, "FaceBoxes_epoch_" + str(i) + ".pth")
        torch.save(params, save_path)


def main():
    num_classes = CONFIG["num_classes"]
    max_epoch = CONFIG["max_epoch"]
    save_directory = CONFIG["save_directory"]
    phase = "train"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FaceBoxes(num_classes, phase)
    model.to(device)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    train_net(model, device, max_epoch, save_directory)


if __name__ == "__main__":
    main()
