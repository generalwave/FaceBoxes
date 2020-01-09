import torch
import cv2
from models.faceboxes import FaceBoxes
from config import CONFIG, CLASSES_NAME_ID
from core.anchor_generator import AnchorGenerator
from core.box_utils import decode
import numpy as np


def nms(dets, thresh, mode="Union"):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if mode == "Union":
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == "Minimum":
            ovr = inter / np.minimum(areas[i], areas[order[1:]])
        else:
            raise IOError('sb a')

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def get_model():
    pretrain_model = "/Users/yangjiang/temp/gpu1/FaceBoxes_epoch_299_v2.pth"
    num_classes = CONFIG["num_classes"]
    params = torch.load(pretrain_model, map_location="cpu")
    model = FaceBoxes(num_classes, "test")
    model.load_state_dict(params)
    model.eval()
    return model


def get_anchors(image_size):
    scales = CONFIG["scales"]
    strides = CONFIG["strides"]
    anchor_generator = AnchorGenerator(image_size, scales, strides)
    anchors = anchor_generator()
    return anchors


def main():
    # 常用参数
    image_size = (1024, 1024)
    face_class_id = CLASSES_NAME_ID["face"]
    bgr_mean = CONFIG["bgr_mean"]
    variance = CONFIG["variance"]

    model = get_model()
    anchors = get_anchors(image_size)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    with torch.no_grad():
        while cap.isOpened():
            success, image = cap.read()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            image = cv2.flip(image, 1)
            image = image[:, 280:1000]
            image = cv2.resize(image, image_size, interpolation=cv2.INTER_LINEAR)
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rgb_mean = (bgr_mean[2], bgr_mean[1], bgr_mean[0])
            img = img.astype(np.float32)
            img -= rgb_mean
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).unsqueeze(0)

            conf, loc = model(img)
            conf = conf.squeeze(0)
            loc = loc.squeeze(0)

            # 解析最后的结果，得到人脸置信度及人脸框位置
            boxes = decode(loc, anchors, variance).numpy()
            boxes = boxes * np.array([image_size[1], image_size[0], image_size[1], image_size[0]])
            scores = conf[:, face_class_id].numpy()
            # 判决为人脸置信度门限
            mask = np.where(scores > 0.05)[0]
            boxes = boxes[mask]
            scores = scores[mask]
            # 只对 topk 进行 nms，降低消耗
            order = scores.argsort()[::-1][:5000]
            boxes = boxes[order]
            scores = scores[order]

            # nms 门限
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, 0.3)
            dets = dets[keep, :]
            # 最大检测人脸数
            dets = dets[:750, :]

            for i, b in enumerate(dets):
                # 显示门限
                if b[4] < 0.5:
                    continue
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))

                xmin = max(0, b[0])
                ymin = max(0, b[1])
                xmax = min(image_size[1], b[2])
                ymax = min(image_size[0], b[3])

                wname = "test%d" % i
                h = b[3] - b[1]
                w = b[2] - b[0]
                face = np.zeros((h, w, 3), image.dtype)
                face_xmin = max(0, -b[0])
                face_ymin = max(0, -b[1])
                face_xmax = face_xmin + xmax - xmin
                face_ymax = face_ymin + ymax - ymin
                face[face_ymin:face_ymax, face_xmin:face_xmax, :] = image[ymin:ymax, xmin:xmax, :]
                cv2.imshow(wname, face)

                cv2.rectangle(image, (xmin, ymin), (xmax - 1, ymax - 1), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(image, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
            cv2.imshow('test', image)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
