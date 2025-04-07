import numpy as np
from torchvision import datasets, transforms


class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None

def build_transform(is_train):
    input_size = 224
    resize_im = input_size > 32
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)

        transform = [
            transforms.RandomResizedCrop(input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())

    # return transforms.Compose(t)
    return t

def split_images_labels(imgs):
    # split trainset.imgs in ImageFolder
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)

# 以下为具体数据集获取即预处理

# CIFAR100
class iCIFAR100(iData):
    use_path = False

    train_trsf = build_transform(is_train=True)
    test_trsf = build_transform(is_train=False)
    common_trsf = [transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))]

    # 未打乱的class_order
    class_order = np.arange(100).tolist()

    def download_data(self):
        # 此时无需transforms
        train_dataset = datasets.cifar.CIFAR100('H:/DATA4DL/cifar', train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100('H:/DATA4DL/cifar', train=False, download=True)

        # 获取data与targets
        self.train_data, self.train_targets = train_dataset.data, np.array(train_dataset.targets)
        self.test_data, self.test_targets = test_dataset.data, np.array(test_dataset.targets)

class iCIFAR224(iData):
    use_path = False

    train_trsf = build_transform(is_train=True)
    test_trsf = build_transform(is_train=False)
    common_trsf = []

    # 未打乱的class_order
    class_order = np.arange(100).tolist()

    def download_data(self):
        # 此时无需transforms
        train_dataset = datasets.cifar.CIFAR100('E:/DATA4DL/cifar', train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100('E:/DATA4DL/cifar', train=False, download=True)

        # 获取data与targets
        self.train_data, self.train_targets = train_dataset.data, np.array(train_dataset.targets)
        self.test_data, self.test_targets = test_dataset.data, np.array(test_dataset.targets)

class CUB(iData):
    use_path = True

    train_trsf = build_transform(is_train=True)
    test_trsf = build_transform(is_train=False)
    common_trsf = []

    class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = 'E:/DATA4DL/CUB/train/'
        test_dir = 'E:/DATA4DL/CUB/test/'

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)