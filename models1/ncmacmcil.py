import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from convs.losses import NCMLoss
import timm
from collections import OrderedDict
from convs.vpt import build_promptmodel
import utils.toolkits as toolkits


class Learner:
    def __init__(self, args):
        self.cur_classes = None
        self.prototypes = None
        self.args = args
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self.classes_per_task = []
        self.feature_proto_list = None
        self.prompt_pool = []
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._old_network = self.call_model()

    def after_task(self):
        self._known_classes = self._total_classes

    def call_model(self):
        return build_promptmodel(modelname="vit_base_patch16_224_in21k",
                                 Prompt_Token_num=self.args["Prompt_Token_num"],
                                 VPT_type=self.args["VPT_type"], args=self.args,
                                 new_classes=0, frozen_heads=True).to(self._device)

    def incremental_train(self, data_manager):
        self._cur_task = self._cur_task + 1
        cur_classes_num = data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + cur_classes_num
        self.cur_classes = [i + self._known_classes for i in range(cur_classes_num)]
        self.classes_per_task.append(self.cur_classes)

        # 数据包构建
        batch_size = self.args["batch_size"]
        train_indices = np.arange(self._known_classes, self._total_classes)
        test_indices = np.arange(0, self._total_classes)

        train_dataset = data_manager.get_dataset(indices=train_indices, source="train", mode="train", )
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        test_dataset = data_manager.get_dataset(indices=test_indices, source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        train_dataset_for_protonet = data_manager.get_dataset(indices=train_indices, source="train", mode="test", )
        self.train_loader_for_protonet = DataLoader(train_dataset_for_protonet, batch_size=batch_size, shuffle=True, num_workers=0)

        self._train()

    def _train(self,):
        self._network = self.call_model()
        if self.prompt_pool != []:
            self._network.load_prompt(self.prompt_pool[0])
            self._network.to(self._device)
        self._network.eval()

        # 推理
        bias = 10 * torch.randn(768)
        feature_proto_list = [torch.randn(768) + bias for num in self.cur_classes]
        # feature_proto_list = toolkits.get_protos_with_tqdm(self.train_loader_for_protonet, self._device, self._network)
        self.previous_feature_proto_list = [] if self._known_classes == 0 else self.feature_proto_list
        self.feature_proto_list = feature_proto_list if self._known_classes == 0 else self.feature_proto_list + feature_proto_list
        self.prototypes = torch.stack(self.feature_proto_list).to(self._device)

        self.train_vpt()
        self._network.eval()

    def train_vpt(self,):
        # 参数初始化
        self._network.train()

        # show model parameters
        # toolkits.show_model_params(self._network)

        # train meta parameters
        lrs = self.args["lr_prompt"]
        if self._cur_task > len(lrs) - 1:
            lr_ = 5.0e-6
        else:
            lr_ = lrs[self._cur_task]
        optimizer = optim.SGD(self._network.parameters(), momentum=0, lr=lr_, weight_decay=5e-4)
        loss_fn = NCMLoss()

        # 训练VPT
        loss_average_min = np.inf
        train_accuracy_max = 0.0
        prompt_ = None
        for epoch in range(self.args["tuned_epoch"]):
            self._network.train()
            losses = 0.0

            # 创建 tqdm 进度条
            with (tqdm(total=len(self.train_loader), desc=f"Epoch [{epoch + 1}/{self.args["tuned_epoch"]}]", ncols=120) as pbar):
                for batch_counter, (_, inputs, targets) in enumerate(self.train_loader_for_protonet):
                    inputs, targets = inputs.to(self._device), targets.to(self._device)

                    # 前向传播
                    optimizer.zero_grad()
                    outputs = self._network(inputs)

                    # 准备tsne数据
                    target_long = targets.long().detach().squeeze()
                    if batch_counter == 0:
                        feature_bank, target_bank = outputs.detach().cpu(), target_long.detach().cpu()
                    else:
                        feature_bank = torch.cat((feature_bank, outputs.detach().cpu()), dim=0)
                        target_bank = torch.cat((target_bank, target_long.detach().cpu()), dim=0)

                    loss = loss_fn.forward(features=outputs, labels=targets, prototypes=self.prototypes)

                    # 反向传播
                    loss.backward()
                    optimizer.step()
                    losses += loss.item()

                    # 更新进度条中的信息
                    loss_average = losses / (pbar.n + 1)
                    pbar.set_postfix(loss=loss_average)  # 显示损失和准确率
                    pbar.update(1)  # 更新进度条

            # 推理得到新的prototypes
            feature_proto_list = toolkits.get_protos_with_tqdm(self.train_loader_for_protonet, self._device, self._network)
            self.feature_proto_list = self.previous_feature_proto_list + feature_proto_list
            self.prototypes = torch.stack(self.feature_proto_list).to(self._device)

            # 测试
            train_accuracy = toolkits.test_accuracy(model=self._network, data_loader=self.train_loader_for_protonet,
                                                    prototypes=self.prototypes, epoch=epoch,
                                                    num_epochs=self.args["tuned_epoch"], device=self._device, words='Train')

            # 绘tsne图
            toolkits.tsne_classes(feature_bank, target_bank)

            # 终止epoch loop
            if train_accuracy > 97.5:   # > 97.5
                prompt_ = self._network.obtain_prompt()
                break
            if train_accuracy > train_accuracy_max:
                train_accuracy_max = train_accuracy
                prompt_ = self._network.obtain_prompt()
                # print(f'train accuracy is {train_accuracy}, save prompt {prompt_}')

        # 做Prompt的Merging
        if self.prompt_pool != []:
            alpha = 1 / (self._cur_task + 1)
            prompt_ = toolkits.weighted_prompt_average(self.prompt_pool[-1], prompt_, alpha)

        # 再次推理得到新的prototypes
        self._old_network.load_prompt(prompt_)
        self._old_network.to(self._device)
        feature_proto_list = toolkits.get_protos_with_tqdm(self.train_loader_for_protonet, self._device, self._old_network)
        self.feature_proto_list = self.previous_feature_proto_list + feature_proto_list
        self.prototypes = torch.stack(self.feature_proto_list).to(self._device)
        toolkits.test_accuracy(model=self._network, data_loader=self.train_loader_for_protonet,
                               prototypes=self.prototypes, device=self._device, words='Merge')

        # 保存prompt_token
        self.prompt_pool.append(prompt_)

    # ----------------------------------------------------------------
    # Evaluation
    # ----------------------------------------------------------------
    def eval_accuracy(self, words='Test', top_num=1):
        model = self._network
        model.load_prompt(self.prompt_pool[-1])
        data_loader = self.test_loader
        num_classes = self.prototypes.size(0)  # 总类别数
        with tqdm(total=len(data_loader), desc=f"Task{words}", ncols=120) as pbar_test:
            y_pred, y_true = [], []
            with torch.no_grad():
                for _, inputs, targets in data_loader:
                    inputs, targets = inputs.to(self._device), targets.to(self._device)
                    outputs = model(inputs)

                    # 批量计算与所有原型的L2距离（高效向量化）
                    temperature = 1.0
                    distances = torch.cdist(outputs, self.prototypes, p=2)
                    logits = -distances / temperature

                    # 取Top-K预测结果
                    _, top_indices = torch.topk(logits, k=top_num, dim=1)
                    batch_preds = top_indices.cpu().tolist()  # 取Top1预测

                    y_pred.extend(batch_preds)
                    y_true.extend(targets.cpu().tolist())

                    # 更新进度条
                    test_accuracy = 100 * toolkits.top_k_accuracy(y_pred, y_true, k=1)
                    pbar_test.set_postfix(accuracy=f"{test_accuracy:.2f}%")
                    pbar_test.update(1)

        # ------------------------------------------------------------------
        # TSNE
        # ------------------------------------------------------------------
        model.load_prompt(self.prompt_pool[-1])  # 确保模型支持动态加载Prompt
        model.to(self._device)
        model.eval()
        for batch_counter, (_, inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(self._device), targets.to(self._device)

            # 前向传播
            outputs = model(inputs)

            # 准备tsne数据
            target_long = targets.long().detach().squeeze()
            if batch_counter == 0:
                feature_bank, target_bank = outputs.detach().cpu(), target_long.detach().cpu()
            else:
                feature_bank = torch.cat((feature_bank, outputs.detach().cpu()), dim=0)
                target_bank = torch.cat((target_bank, target_long.detach().cpu()), dim=0)

        # 绘tsne图
        toolkits.tsne_classes(feature_bank, target_bank)

        return test_accuracy