import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from sklearn import metrics


def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    loss_total = 0
    predict_all = np.array([], dtype=int)  # 储存验证集所有batch的预测结果
    labels_all = np.array([], dtype=int)  # 储存验证集所有batch的真实标签

    # iterate over the validation set
    with torch.no_grad():
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            videos, labels = batch[0], batch[1]
            print(labels.size())
            # move videos and labels to correct device and type
            videos = videos.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.long)

            outputs = net(videos)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predict = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predict)

    acc = metrics.accuracy_score(labels_all, predict_all)  # 计算验证集准确率

    return acc, loss_total / len(dataloader)

