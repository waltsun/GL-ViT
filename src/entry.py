import os
import sys
import random
import numpy as np
from scipy import stats

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image

import vision_transformer as vits
import utils
from tqdm import trange
import cv2
from torch.nn import functional as F
from dataloader import Dataloader
os.environ["CUDA_VISIBLE_DEVICES"] = "5"


# ---------------------dataset-------------------------
dataset = ["mini-ImageNet", "CUB"][0]
data_path = {
    "mini-ImageNet": '../../data/mini-ImageNet',
    "tiered-ImageNet": '../../data/tiered-ImageNet',
    "CUB": '../../data/CUB_fewshot_raw',
}[dataset]
cudnn.benchmark = True
shot_num = 1
train_config = {
    "way_num": 10,
    "shot_num": shot_num,
    "query_num": {
        1: 10,
        5: 5,
    }[shot_num],
}
test_config = {
    "way_num": 5,
    "shot_num": shot_num,
    "query_num": 30,
}
valid_config = test_config

dataloader_train = Dataloader(train_config["way_num"], train_config["shot_num"], train_config["query_num"], data_path, "train")
dataloader_val = Dataloader(valid_config["way_num"], valid_config["shot_num"], valid_config["query_num"], data_path, "val")
dataloader_test = Dataloader(test_config["way_num"], test_config["shot_num"], test_config["query_num"], data_path, "test")
# ---------------------dataset-------------------------


# ---------------------random seed---------------------
seed = 42
utils.fix_random_seeds(seed)
# ---------------------random seed---------------------


# ---------------------ViT model-----------------------
pretrained_weights={
    "mini-ImageNet": '../pretrain_ckpt/mini_imagenet_checkpoint.pth',
    "tiered-ImageNet": '../pretrain_ckpt/tiered_imagenet_checkpoint.pth',
    "CUB": '../pretrain_ckpt/dino_deitsmall16_pretrain.pth',
}[dataset]
arch='vit_small'
patch_size=16
checkpoint_key="teacher"

model = vits.vit_small(patch_size=patch_size, num_classes=0)
model.cuda()
utils.load_pretrained_weights(model,
                                pretrained_weights, 
                                checkpoint_key,  
                                arch,  
                                patch_size)
# ---------------------ViT model-----------------------


# -------------------functions-------------------------
@torch.no_grad()
def emd_inference_opencv(cost_matrix, weight1, weight2):
    # cost matrix is a tensor of shape [N,N]
    cost_matrix = cost_matrix.detach().cpu().numpy()

    weight1 = F.relu(weight1) + 1e-5
    weight2 = F.relu(weight2) + 1e-5

    weight1 = (weight1 * (weight1.shape[0] / weight1.sum().item())).view(-1, 1).detach().cpu().numpy()
    weight2 = (weight2 * (weight2.shape[0] / weight2.sum().item())).view(-1, 1).detach().cpu().numpy()

    _, _, flow = cv2.EMD(weight1, weight2, cv2.DIST_USER, cost_matrix)
    return flow

def get_similarity(A, B):
    n = A.shape[0]
    m = B.shape[0]

    A = A.unsqueeze(1).expand(-1, m, -1)
    B = B.unsqueeze(0).expand(n, -1, -1)
    return F.cosine_similarity(A, B, -1)

IF_PROJ = False
if IF_PROJ:
    from vision_transformer import Mlp
    proj = nn.Linear(384, 384).cuda()

IF_PATCH = True
NUM_PATCH = 4 if IF_PATCH else 14

FEATS = ["parallel", "cat", "global-local-minus", "local-only", "global-only", "global-local-plus", None][ 2 ]

def get_emd(support_features, query_features, query_episode_labels, support_attn, query_attn):

    n = support_features.shape[0]  # 5
    m = query_features.shape[0]  # 150

    support_features_global = support_features[:, 0]  # 5 * 384
    support_features_local = support_features[:, 1:]  # 5 * 196 * 384
    query_features_global = query_features[:, 0]  # 150 * 384
    query_features_local = query_features[:, 1:]  # 150 * 196 * 384

    if FEATS == "global-only":
        logits = get_similarity(query_features_global, support_features_global)
        return logits

    support_attn_global = support_attn[:, 0]  # 5
    support_attn_local = support_attn[:, 1:]  # 5 * 196
    query_attn_global = query_attn[:, 0]  # 150
    query_attn_local = query_attn[:, 1:]  # 150 * 196

    support_features_local = support_attn_local.unsqueeze(2) * support_features_local
    query_features_local = query_attn_local.unsqueeze(2) * query_features_local

    support_features_global = support_attn_global.unsqueeze(1) * support_features_global
    query_features_global = query_attn_global.unsqueeze(1) * query_features_global

    if IF_PATCH:
        support_features_local = F.adaptive_avg_pool2d(support_features_local.transpose(1, 2).view(n, -1, 14, 14), (NUM_PATCH, NUM_PATCH)).view(n, -1, NUM_PATCH*NUM_PATCH).transpose(1, 2)  # 5 * 25 * 384
        query_features_local = F.adaptive_avg_pool2d(query_features_local.transpose(1, 2).view(m, -1, 14, 14), (NUM_PATCH, NUM_PATCH)).view(m, -1, NUM_PATCH*NUM_PATCH).transpose(1, 2)  # 150 * 25 * 384
        support_attn_local = F.adaptive_avg_pool2d(support_attn_local.view(-1, 14, 14), (NUM_PATCH, NUM_PATCH)).view(-1, NUM_PATCH * NUM_PATCH)# * ((14 / NUM_PATCH) ** 2)  # 5 * 25
        query_attn_local = F.adaptive_avg_pool2d(query_attn_local.view(-1, 14, 14), (NUM_PATCH, NUM_PATCH)).view(-1, NUM_PATCH * NUM_PATCH)# * ((14 / NUM_PATCH) ** 2)  # 150 *  25

    if FEATS == "parallel":
        support_features = torch.cat([support_features_global.unsqueeze(1), support_features_local], dim = 1)  # 5 * 26 * 384
        query_features = torch.cat([query_features_global.unsqueeze(1), query_features_local], dim = 1)  # 150 * 26 * 384
        support_attn = torch.cat([support_attn_global.unsqueeze(1), support_attn_local], dim = 1)  # 5 * 26
        query_attn = torch.cat([query_attn_global.unsqueeze(1), query_attn_local], dim = 1)  # 150 * 26
    elif FEATS == "cat":
        support_features = torch.cat([support_features_local - support_features_global.unsqueeze(1).expand(-1, NUM_PATCH*NUM_PATCH, -1), support_features_local], dim = 2)  # 5 * 25 * 384*2
        query_features = torch.cat([query_features_local - query_features_global.unsqueeze(1).expand(-1, NUM_PATCH*NUM_PATCH, -1), query_features_local], dim = 2)  # 150 * 25 * 384*2
        support_attn = support_attn_local  # 5 * 25
        query_attn = query_attn_local  # 150 * 25
    elif FEATS == "global-local-minus":
        support_features = support_features_local - support_features_global.unsqueeze(1).expand(-1, NUM_PATCH*NUM_PATCH, -1)  # 5 * 25 * 384
        query_features = query_features_local - query_features_global.unsqueeze(1).expand(-1, NUM_PATCH*NUM_PATCH, -1)  # 150 * 25 * 384
        support_attn = support_attn_local  # 5 * 25
        query_attn = query_attn_local  # 150 * 25
    elif FEATS == "local-only":
        support_features = support_features_local
        query_features = query_features_local
        support_attn = support_attn_local
        query_attn = query_attn_local
    elif FEATS == "global-local-plus":
        support_features = support_features_local + support_features_global.unsqueeze(1).expand(-1, NUM_PATCH*NUM_PATCH, -1)  # 5 * 25 * 384
        query_features = query_features_local + query_features_global.unsqueeze(1).expand(-1, NUM_PATCH*NUM_PATCH, -1)  # 150 * 25 * 384
        support_attn = support_attn_local  # 5 * 25
        query_attn = query_attn_local  # 150 * 25
    else:
        raise NotImplementedError

    if IF_PROJ:
        support_features = proj(support_features)
        query_features = proj(query_features)

    logits = []
    for j in range(m):  # 150
        B = query_features[j]  # 26 * 384
        logit = []
        for i in range(n):  # 5
            A = support_features[i]  # 26 * 384
            weight1 = support_attn[i]  # 26
            weight2 = query_attn[j]  # 26
            similarity = get_similarity(A, B)  # 26 * 26
            flow = emd_inference_opencv(1 - similarity.detach(), weight1, weight2)
            score = (similarity * torch.tensor(flow).cuda()).sum()
            logit.append(score)
        logit = torch.stack(logit)  # 5
        logits.append(logit)
    logits = torch.stack(logits, dim = 0)  # 150 * 5
    return logits


def get_loss(support_features, query_features, query_episode_labels, support_attn, query_attn):
    support_attn = support_attn.detach()[:, :, 0]  # 5 * 6 * 197  (head = 6)
    query_attn = query_attn.detach()[:, :, 0]  # 150 * 6 * 197
    logits = get_emd(support_features, query_features, query_episode_labels, support_attn.mean(1), query_attn.mean(1))  # 150 * 5
    if shot_num > 1:
        logits = torch.log(F.softmax(logits, dim=-1).view(query_episode_labels.shape[0], -1, shot_num).mean(dim = -1))
        loss = F.nll_loss(logits, query_episode_labels)
    # print(logits.shape, query_episode_labels.shape)
    else:
        loss = F.cross_entropy(logits, query_episode_labels)
    _, pred = logits.max(dim = 1)
    acc_vec = pred.detach().eq(query_episode_labels)
    acc = acc_vec.float().mean()
    return loss, acc, acc_vec
# -------------------functions-------------------------


# -------------------optimizer-------------------------
from torch.optim import Adam, SGD
if IF_PROJ:
    optimizer = Adam([{"params": model.parameters()}, {"params": proj.parameters(), "lr": 1e-5}], lr = 1e-5)
else:
    optimizer = Adam(model.parameters(), lr = 1e-5)
# -------------------optimizer-------------------------


test_episode_num = 600
train_episode_num = 200

print("\nCUDA: ", os.environ["CUDA_VISIBLE_DEVICES"])
print("seed: ", seed)
print("test_episode_num: ", test_episode_num)
print("dataset: ", dataset)
print("shot num: ", shot_num)
print("patch: ", NUM_PATCH)
print("FEATS: ", FEATS)
print()

@torch.no_grad()
def test(dataloader):
    model.eval()
    avg_acc = []
    for i in trange(test_episode_num):
        support_images, support_labels, query_images, query_labels, query_episode_labels = dataloader.get_episode_batch()
        support_features, support_attn = model.get_last_selfattention(support_images.cuda())
        query_features, query_attn = model.get_last_selfattention(query_images.cuda())
        loss, acc, acc_vec = get_loss(support_features, query_features, query_episode_labels.cuda(), support_attn, query_attn)
        avg_acc.append(acc.item())
    avg, std = np.mean(avg_acc), stats.sem(avg_acc)
    model.train()
    tmpl, tmpr = stats.norm.interval(0.95, avg, std)
    itv = (tmpr-tmpl)/2
    return avg * 100, itv * 100

def train(dataloader):
    model.train()
    avg_acc = []
    val_acc = []
    val_itv = []
    test_acc = []
    test_itv = []
    for i in trange(train_episode_num):
        optimizer.zero_grad()
        support_images, support_labels, query_images, query_labels, query_episode_labels = dataloader.get_episode_batch()
        support_features, support_attn = model.get_last_selfattention(support_images.cuda())
        query_features, query_attn = model.get_last_selfattention(query_images.cuda())
        loss, acc, acc_vec = get_loss(support_features, query_features, query_episode_labels.cuda(), support_attn, query_attn)
        loss.backward()
        optimizer.step()
        avg_acc.append(acc.item())
        if (i + 1) % 25 == 0:
            avg_v, itv_v = test(dataloader_val)
            val_acc.append(avg_v), val_itv.append(itv_v)
            print("val: ", avg_v, itv_v)

            avg_t, itv_t = test(dataloader_test)
            test_acc.append(avg_t), test_itv.append(itv_t)
            print("test: ", avg_t, itv_t)

            # torch.save(model.state_dict(), "../ckpt/epoch-%03d-val-%.2f-test-%.2f.ckpt" % (i+1, avg_v, avg_t))
    print("val acc: ", val_acc)
    print("val itv: ", val_itv)
    print("test acc: ", test_acc)
    print("test itv: ", test_itv)
    print(np.max(test_acc))
    print(test_acc[np.argmax(val_acc)], test_itv[np.argmax(val_acc)])


train(dataloader_train)
