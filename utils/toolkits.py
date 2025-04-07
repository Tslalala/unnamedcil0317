import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import json
import argparse
from tqdm import tqdm
from sklearn.manifold import TSNE
import os

from . import data_category


def seed_set():
    torch.manual_seed(1)  # 影响CPU上随机操作
    torch.cuda.manual_seed(1)  # 影响GPU上随机操作
    torch.cuda.manual_seed_all(1)  # 影响多GPU上随机操作
    torch.backends.cudnn.deterministic = True  # CuDNN库以确定性模式运行,在 GPU 上相同的输入将会产生相同的输出
    torch.backends.cudnn.benchmark = False  # 禁用了CuDNN自动基准测试功能。启用时CuDNN会在多个卷积算法中选择最快的一个,可能会导致每次运行时选择不同的算法
    np.random.seed(1993)


# 加载数据集
def get_idata(dataset_name, args=None):
    dataset_name = dataset_name.lower()
    if dataset_name == "cifar100":
        return data_category.iCIFAR100()
    if dataset_name == "cifar224":
        return data_category.iCIFAR224()
    if dataset_name == "cub":
        return data_category.CUB()
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


# 创建split_dataset编队_increments
def get_increments_list(init_cls, increment, _class_order):
    assert init_cls <= len(_class_order), "No enough classes."
    _increments = [init_cls]
    while sum(_increments) + increment < len(_class_order):
        _increments.append(increment)
    offset = len(_class_order) - sum(_increments)
    if offset > 0:
        _increments.append(offset)
    # print(_increments)  # cifar100_inc5: [5,5,...,5]
    return _increments


def get_model(model_name, args):
    name = model_name.lower()
    if name == 'simplecil':
        from models0.simplecil import Learner
        print('--model simplecil')
        return Learner(args)
    elif name == 'constantcoordinatecil':
        from models0.constantcoordinatecil import Learner
        print('--model constantcoordinatecil')
        return Learner(args)
    elif name == 'crazyprotocil':
        from models0.crazyprotocil import Learner
        print('--model crazyprotocil')
        return Learner(args)
    elif name == 'gdprotocil':
        from models0.gdprotocil import Learner
        print('--model gdprotocil')
        return Learner(args)
    elif name == 'kmeanscil':
        from models0.kmeanscil import Learner
        print('--model kmeanscil')
        return Learner(args)
    elif name == 'gdprotoscil':
        from models0.gdprotoscil import Learner
        print('--model gdprotoscil')
        return Learner(args)
    elif name == 'apervpt':
        from models0.apervpt import Learner
        print('--model apervpt')
        return Learner(args)
    elif name == 'apervpt_simplecil':
        from models0.apervpt_simplecil import Learner
        print('--model apervpt_simplecil')
        return Learner(args)
    elif name == 'contrastcil':
        from models1.contrastcil import Learner
        print('--model contrastcil')
        return Learner(args)
    elif name == 'ncmlosscil':
        from models1.ncmlosscil import Learner
        print('--model ncmlosscil')
        return Learner(args)
    elif name == 'ncmacmcil':
        from models1.ncmacmcil import Learner
        print('--model ncmacmcil')
        return Learner(args)
    else:
        raise NotImplementedError("Unknown model {}.".format(model_name))


def load_json(setting_path):
    with open(setting_path) as data_file:
        param = json.load(data_file)
    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='Beginning Your Incremental Class Learning Experiment.')
    parser.add_argument('--config', type=str, default='./exps/cifar_in21k.json', help='Json file of settings.')
    return parser


def show_model_params(network):
    # Freeze the parameters for ViT.
    total_params = sum(p.numel() for p in network.parameters())
    print(f'{total_params:,} total parameters.', end='\t')
    total_trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.', end='\t')

    # if some parameters are trainable, print the key name and corresponding parameter number
    if total_params != total_trainable_params:
        for name, param in network.named_parameters():
            if param.requires_grad:
                print(name, param.numel(), end='\t')
    print()

def classify_with_proto(test_output, prototypes, top_num=2):
    """
    基于原型（L2距离）的分类函数
    Args:
        test_output: 测试样本特征 [768]
        prototypes: 所有类别原型 [35, 768]
        top_num: 返回前top_num个预测结果
    Returns:
        predict: 预测的类别索引列表（长度=top_num）
    """
    # 确保数据在CPU（若prototypes在GPU）
    test_output = test_output.cpu()
    prototypes = prototypes.cpu()

    # 向量化计算所有类别的L2距离（形状 [35]）
    distances = torch.sum((test_output - prototypes) ** 2, dim=1)  # [35]

    # 计算相似度（取距离倒数，避免除零）
    epsilon = 1e-8  # 防止距离为0导致无穷大
    similarities = 1 / (distances + epsilon)  # [35]

    # 直接取Top-K类别索引
    top_sim, top_indices = torch.topk(similarities, k=top_num, largest=True, sorted=True)
    predict = top_indices.tolist()

    return predict[:top_num]  # 确保返回长度一致


def top_k_accuracy(y_pred, y_true, k):
    correct_count = 0
    total_count = len(y_true)

    for i in range(total_count):
        # 检查真实标签是否在预测的前 k 个类别中
        if y_true[i] in y_pred[i][:k]:
            correct_count += 1

    # 计算 top-k 准确率
    accuracy = correct_count / total_count
    return accuracy


def get_protos_with_tqdm(data_loader, device, model):
    embedding_list = []
    label_list = []
    with torch.no_grad():
        for _, inputs, targets in tqdm(data_loader, desc="Inference", ncols=120):
            inputs = inputs.to(device)
            embedding = model(inputs)
            embedding_list.append(embedding.cpu())
            label_list.append(targets.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    # NCM, 对class’s features取mean
    class_list = np.unique(label_list)
    feature_proto_list = []
    for class_index in class_list:
        data_index = (label_list == class_index).nonzero().squeeze(-1)
        embeddings = embedding_list[data_index]
        proto = embeddings.mean(0)
        feature_proto_list.append(proto)

    return feature_proto_list


def get_protos(data_loader, device, model):
    embedding_list = []
    label_list = []
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            embedding = model.forward_features_(inputs)
            embedding_list.append(embedding.cpu())
            label_list.append(targets.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    # NCM, 对class’s features取mean
    class_list = np.unique(label_list)
    feature_proto_list = []
    for class_index in class_list:
        data_index = (label_list == class_index).nonzero().squeeze(-1)
        embeddings = embedding_list[data_index]
        proto = embeddings.mean(0)
        feature_proto_list.append(proto)

    return feature_proto_list


def test_accuracy(model, data_loader, prototypes, epoch=-1, num_epochs=0, device='cuda', words='Test', top_num=2):
    with tqdm(total=len(data_loader), desc=f"{words} Epoch [{epoch + 1}/{num_epochs}]",
              ncols=120) as pbar_test:
        y_pred, y_true = [], []
        with torch.no_grad():  # 不计算梯度
            for _, inputs, targets in data_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)

                # 批量计算与所有原型的L2距离（高效向量化）
                temperature = 1.0
                distances = torch.cdist(outputs, prototypes, p=2)
                logits = -distances / temperature

                # 取Top-K预测结果
                _, top_indices = torch.topk(logits, k=top_num, dim=1)
                batch_preds = top_indices.cpu().tolist()  # 取Top1预测

                y_pred.extend(batch_preds)
                y_true.extend(targets.cpu().tolist())

                # 更新测试进度条信息
                test_accuracy = 100 * top_k_accuracy(y_pred=y_pred, y_true=y_true, k=1)
                pbar_test.set_postfix(accuracy=f"{test_accuracy:.2f}%")
                pbar_test.update(1)

    return test_accuracy


def tsne_classes(feature_bank, target_bank):
    # 假设 feature_bank 和 target_bank 已经转换为 NumPy 数组
    feature_bank = feature_bank.numpy()
    target_bank = target_bank.numpy()

    # 执行 t-SNE 降维

    os.environ["LOKY_MAX_CPU_COUNT"] = "10"  # 假设你有4个CPU核心
    tsne = TSNE(n_components=2, random_state=0, n_jobs=1)
    output = tsne.fit_transform(feature_bank)

    # 获取唯一的类别标签
    unique_labels = np.unique(target_bank)

    # 示例：自定义 12 种高对比度颜色（HEX 格式）
    custom_colors = [
        '#1f77b4', '#ff7f0e', '#0000FF', '#FF00FF',
        '#9467bd', '#FFD700', '#00FFFF', '#7f7f7f',
        '#FF1493', '#17becf', '#4B0082', '#FF4500'
    ]

    # 绘制散点图
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(unique_labels):
        index = (target_bank == label)
        color = custom_colors[i % len(custom_colors)]  # 循环使用颜色
        plt.scatter(
            output[index, 0],
            output[index, 1],
            s=40,
            color=color,
            edgecolors='k',
            linewidths=0.3,
            alpha=0.8,
            label=f'Class {label}'
        )

    plt.legend(markerscale=2)
    plt.title('t-SNE Visualization')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.show()


def modified_loss_fn(current_features,
                     old_features_list,
                     labels,
                     prototypes,
                     current_loss_weight=1.0,
                     old_consistency_weight=0.12,
                     temperature=1.0):
    """
    改进的损失函数，整合当前任务损失和旧模型一致性约束

    Args:
        current_features (Tensor): 当前模型输出特征 [B, D]
        old_features_list (List[Tensor]): 历史模型输出特征列表 [n_old_models, B, D]
        labels (Tensor): 当前批次标签 [B]
        prototypes (Tensor): 所有类原型 [C, D]
        current_loss_weight (float): 当前分类损失权重
        old_consistency_weight (float): 旧模型一致性约束权重
        temperature (float): 相似度分布温度系数

    Returns:
        total_loss (Tensor): 整合后的总损失
    """
    # 原始分类损失（当前模型）
    distances = torch.cdist(current_features, prototypes, p=2)  # [B, C]
    logits = -distances / temperature
    loss_current = F.cross_entropy(logits, labels) * current_loss_weight

    # 旧模型一致性约束项（鼓励与旧模型输出差异）
    consistency_loss = 0
    for old_features in old_features_list:
        # 计算特征相似度（鼓励差异）
        # similarity = F.cosine_similarity(current_features, old_features, dim=1)  # [B]
        similarity = torch.sum((current_features - old_features) ** 2, dim=1)  # [B]
        consistency_loss += similarity.mean()  # 最大化差异即最小化相似度

    # 总损失 = 分类损失 - 相似度损失（因为要最小化总损失，等价于最大化差异）
    total_loss = loss_current - old_consistency_weight * consistency_loss

    return total_loss


def weighted_prompt_average(_prompt: dict, prompt_: dict, alpha: float) -> dict:
    """
    对两个OrderedDict结构的PromptTokens进行加权平均
    参数：
        _prompt: 旧Prompt字典（含各层参数）
        prompt_: 新Prompt字典（含各层参数）
        alpha: 新Prompt的权重（0.5表示各占50%）
    返回：
        new_prompt: 加权后的新Prompt字典
    """
    new_prompt = type(prompt_)()  # 保持原有结构（如OrderedDict）

    # 确保键完全一致
    assert _prompt.keys() == prompt_.keys(), "Prompt字典的层结构不匹配"

    for key in prompt_.keys():
        val_prompt = prompt_[key]
        val_old = _prompt[key]

        # 递归处理嵌套的OrderedDict（如多层级结构）
        if isinstance(val_prompt, dict):
            new_prompt[key] = weighted_prompt_average(val_old, val_prompt, alpha)
        else:
            # 对Tensor进行加权计算
            new_prompt[key] = (1 - alpha) * val_old + alpha * val_prompt

    return new_prompt