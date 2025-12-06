import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F

def save_data():
    ERA5_path = './torch_data/ERA5-grid'
    emip6_historical = './torch_data/emip6/historical/CanESM5'
    lsm_path = './torch_data/lsm.pt'

    ERA5 = []
    for p in sorted(os.listdir(ERA5_path)):
        ERA5.append(torch.from_numpy(np.load(os.path.join(ERA5_path, p))))
    ERA5 = torch.stack(ERA5) 

    emip6_his = []
    for p in sorted(os.listdir(emip6_historical)):
        com = []
        sub_path = os.path.join(emip6_historical, p)
        for subp in sorted(os.listdir(sub_path)):
            com.append(torch.from_numpy(np.load(os.path.join(sub_path, subp))))
        emip6_his.append(torch.stack(com))
    emip6_his = torch.stack(emip6_his) 

    lsm = torch.load(lsm_path) 


    xl, xr, yd, yu = [280, 520, 630, 950]
    ERA5, emip6_his, lsm = [x[..., xl:xr, yd:yu] for x in [ERA5, emip6_his, lsm]]

    emip6_his = emip6_his.permute(1,0,2,3) 
    ERA5 = ERA5[lsm.repeat(ERA5.shape[0],1,1)>0.1].reshape(ERA5.shape[0], 1, -1)    
    emip6_his = emip6_his[lsm.repeat(emip6_his.shape[0],1,1)>0.1].reshape(ERA5.shape[0], -1, ERA5.shape[-1])
    data = torch.concatenate([emip6_his, ERA5], dim=1)
    torch.save(data, './data/emip6-ERA5-[168, 6+1, 32521].pt')
    torch.save(lsm, './data/lsm-[240, 320].pt')

    

def mask(data, lsm):
    shape = data.shape
    data = data[lsm.repeat(data.shape[0], data.shape[1], 1, 1)>0.1]
    return data.reshape(shape[0], shape[1], -1)

def imask(data, lsm):
    idata = torch.zeros(data.shape[0], data.shape[1], lsm.shape[-2], lsm.shape[-1]).to(torch.float32)
    idata[lsm.repeat(data.shape[0], data.shape[1], 1, 1)>0.1] = data.reshape(-1)
    return idata


def get_data():
    data = torch.load('./data/emip6-ERA5-[168, 6+1, 32521].pt').to(torch.float32)
    lsm = torch.load('./data/lsm-[240, 320].pt').to(torch.float32)
    return data, lsm

def get_down_interpolate_data(mask_data, lsm, log=True):
    idata = imask(mask_data, lsm)

    def B_transform(x):
        return x.add(1).log10()
    if log:
        idata[:, -1, :, :] = B_transform(idata[:, -1, :, :])

    idata = F.interpolate(idata, scale_factor=1/10, mode='bilinear')
    lsm = F.interpolate(lsm.unsqueeze(0).unsqueeze(0), scale_factor=1/10, mode='bilinear')[0][0]
    mask_data = mask(idata, lsm)
    idata = imask(mask_data, lsm)
    return mask_data, lsm

def get_norm_data(data, lsm, T=12, fea_means=None, fea_stds=None):
    features = data[:, :-1, :] 
    fire = data[:, -1, :]

    features = features.permute(2,1,0)  
    fire = fire.permute(1,0).unsqueeze(1) 

    if fea_means is None or fea_stds is None:
        fea_means = torch.tensor([features[:, i, :].mean() for i in range(features.shape[1])]).unsqueeze(0).unsqueeze(-1)
        fea_stds = torch.tensor([features[:, i, :].std() for i in range(features.shape[1])]).unsqueeze(0).unsqueeze(-1)
    features = (features - fea_means) / fea_stds
    fire = fire

    return features, fire, lsm, fea_means, fea_stds


def rigion_mask(X, index=None):
    X = X.reshape(X.shape[0], X.shape[1], -1, 12)
    if index is None:
        index = torch.tensor([0 if torch.all(x==0) else 1 for x in X])
    X = X[index.bool(), :, :]
    X = X.reshape(X.shape[0], X.shape[1], -1, 12)
    return X, index

def irigion_mask(X, index):
    iX = torch.zeros(index.shape[0], X.shape[1], X.shape[2], X.shape[3])
    iX[index.bool(), :, :] = X
    return iX

def imodel_data(x, lsm, visual_iter):
    x = torch.fft.irfft(x / visual_iter.dataset.fire_factor)
    X = torch.zeros(visual_iter.dataset._index.shape[0], 1, x.shape[-2], x.shape[-1])
    X[visual_iter.dataset._index.bool(), :, :, :] = x
    x = X.reshape(visual_iter.dataset.index.sum(), -1, x.shape[-2], x.shape[-1]).permute(0,2,1,3)
    ix = irigion_mask(x, visual_iter.dataset.index)
    ix = imask(ix.flatten(-2,-1).permute(2,1,0), lsm)
    return ix