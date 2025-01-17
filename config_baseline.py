import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"#运行设备
TRAIN_DIR = "./data_RSHaze_lite/train" #"./data_Sen_RGB/train"#训练数据目录路径data_HazeSyn
TEST_DIR = "./data_HazeSyn/test"#"./data_Sen_RGB/test"data_HazeSyn
VAL_DIR = "./data_RSHaze_lite/test"#"./data_Sen_RGB/val"#检验数据目录路径data_HazeSyn
BATCH_SIZE = 8#一次训练所抓取的数据样本数量为1
LEARNING_RATE = 1e-4    #学习率为1e-5
# gen
# Loss_supervised

LAMBDA_supervised_clean_MSE = 1#1
LAMBDA_supervised_clean_SSIM = 1e-2#1e-2


NUM_WORKERS = 4
NUM_EPOCHS = 150#完成10个世代的训练
LOAD_MODEL = True#是否使用已有的权重参数
SAVE_MODEL = True#保存权重参数


CHECKPOINT_GEN_C = "RSHaze_dehazeformer_.pth.tar"

transforms = A.Compose(
    [
        A.Resize(width=256, height=256,interpolation=Image.BICUBIC),#图像resize
        #A.RandomCrop(height=256,width=256),
        A.HorizontalFlip(p=0.5),#水平翻转——数据增强方式
        #A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=256-1),
        #前面的（0.5，0.5，0.5） 是 R G B 三个通道上的均值， 后面(0.5, 0.5, 0.5)是三个通道的标准差，最大值设为255
        ToTensorV2(),
     ],
    additional_targets={"image0": "image"},
)

transforms_test = A.Compose(
    [
        A.Resize(width=256, height=256,interpolation=Image.BICUBIC),#图像resize
        ToTensorV2(),
     ],
    additional_targets={"image0": "image"},
)