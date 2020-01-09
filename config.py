CLASSES_NAME_ID = {"__background__": 0, "face": 1}

CONFIG = {
    # 实际是因为每个 epoch 有400次迭代，共 120K 次
    "max_epoch": 300,
    # 需要改变学习率的 epoch 列表
    "epoches": [200, 250],
    # 学习率改变的比例
    "gamma": 0.1,
    "lr": 0.001,
    "batch_size": 32,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "thread_num": 16,

    # 数据集中各通道的均值
    "bgr_mean": (104, 117, 123),
    # 模型进行分类的数目
    "num_classes": len(CLASSES_NAME_ID),
    # 将 anchor 设置为正例的 iou 门限
    "threshold": 0.35,
    # 负样本和正样本的比例
    "negative_positive_ratio": 7,
    # 中心点方差和长宽方差
    "variance": [0.1, 0.2],
    # 定位误差的权重
    "location_weight": 2,

    # 训练输入图片大小
    "input_size": 1024,
    # 高和宽，生成 anchor 时的图片大小
    "image_size": [1024, 1024],
    # anchor 对应的尺度、比例、扩展密度
    "scales": [[(32, 1., 4), (64, 1., 2), (128, 1., 1)], [(256, 1., 1)], [(512, 1., 1)]],
    # 各层 anchor 对应的步进
    "strides": [32, 64, 128],

    # 是否需要将 boxes 划定在图内部
    "clip": False,
    # 是否需要将小脸涂为均值
    "smear_small_face": True,

    # 训练集目录
    "train_directory": "/data/jiang.yang/output/wider/WIDER_train/images",
    # 训练集 anno 文件位置
    "train_mat": "/data/jiang.yang/output/wider/wider_face_split/wider_face_train.mat",
    # 测试集目录
    "val_directory": "/data/jiang.yang/output/wider/WIDER_val/images",
    # 测试集 anno 文件位置
    "val_mat": "/data/jiang.yang/output/wider/wider_face_split/wider_face_val.mat",
    # 模型保存位置
    "save_directory": "/data/jiang.yang/output/FaceBoxes"
}
