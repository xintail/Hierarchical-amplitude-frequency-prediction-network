import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt


from dataset import Amplitude_frequency_increment_Dataset, Post_processing_Dataset
from model import Amplitude_frequency_increment_Model, Hierarchical_Model, initialize_weights
from utils import get_data, get_down_interpolate_data, get_norm_data, imask, irigion_mask, imodel_data

def loss(x, x_hat):
    l1 = nn.MSELoss()(x.real, x_hat.real)
    l2 = nn.MSELoss()(x.imag, x_hat.imag)
    l3 = nn.MSELoss()(x.abs(), x_hat.abs())
    return l1 + l2 + l3

def post_loss(x, x_hat):
    return nn.MSELoss()(x, x_hat)

def train():
    data, lsm = get_data()
    mask_data, lsm = get_down_interpolate_data(data, lsm)
    features, fire, lsm, fea_means, fea_stds = get_norm_data(mask_data, lsm)

    batch_size_1 = 128
    batch_size_2 = 12
    device = "cuda:0"
    lr_1 = 8e-4
    lr_2 = 2e-3
    epochs_1 = 250
    epochs_2 = 1000

    net = Hierarchical_Model(64)
    net.to(device)
    net.apply(initialize_weights)
    

    train_iter = DataLoader(dataset=Amplitude_frequency_increment_Dataset(features, fire), batch_size=batch_size_1, shuffle=True, drop_last=True)
    test_iter = DataLoader(dataset=Amplitude_frequency_increment_Dataset(features, fire, is_train=False), batch_size=batch_size_1, shuffle=True, drop_last=True)


    Train_Loss = []
    test_Loss = []
    for param in net.net1.parameters():
        param.requires_grad = True
    for param in net.net2.parameters():
        param.requires_grad = False
    trainer = torch.optim.Adam(net.net1.parameters(), lr=lr_1)
    for epoch in range(epochs_1):
        net.train()
        ls = []
        for batch in train_iter:
            feature, fire = [x.to(torch.complex64).to(device) for x in [batch['feature'], batch['fire']]]
            fire_hat = net(feature)
            l = loss(fire, fire_hat)
            ls.append(l.cpu().detach())

            trainer.zero_grad()
            l.backward()
            trainer.step()
            print(f"\repoch: {epoch+1}, loss: {l}", end='')
        Train_Loss.append(torch.tensor(ls).mean())
    
        net.eval()
        ls = []
        for batch in test_iter:
            feature, fire = [x.to(torch.complex64).to(device) for x in [batch['feature'], batch['fire']]]
            fire_hat = net(feature)
            ls.append(loss(fire, fire_hat).cpu().detach())
        test_Loss.append(torch.tensor(ls).mean())
        print(f"   vloss: {torch.tensor(ls).mean()}", end='')    
        
    plt.plot(Train_Loss, label='Train')
    plt.plot(test_Loss, label='Test')
    plt.legend()
    plt.savefig("./visuals/Loss_1.png", dpi=300)
    plt.clf()
    plt.close()


    Train_Loss = []
    test_Loss = []
    data, lsm = get_data()
    mask_data, lsm = get_down_interpolate_data(data, lsm)
    features, fire, lsm, fea_means, fea_stds = get_norm_data(mask_data, lsm)
    for d_iter in [DataLoader(dataset=Amplitude_frequency_increment_Dataset(features, fire), batch_size=1, shuffle=False),
                   DataLoader(dataset=Amplitude_frequency_increment_Dataset(features, fire, is_train=False), batch_size=1, shuffle=False)]:
        net.eval()
        PRED = []
        Labels = []
        for batch in tqdm.tqdm(d_iter):
            feature, fire = [x.to(torch.complex64).to(device) for x in [batch['feature'], batch['fire']]]
            fire_hat = net(feature)
            PRED.append(fire_hat.cpu().detach())
            Labels.append(fire.cpu().detach())
        PRED = torch.concatenate(PRED, dim=0)
        Labels = torch.concatenate(Labels, dim=0)
        iPRED = imodel_data(PRED, lsm, d_iter).cpu().detach()     
        iLabels = imodel_data(Labels, lsm, d_iter).cpu().detach() 
        iPRED = torch.where(torch.isnan(iPRED), 0, iPRED)
        iLabels = torch.where(torch.isnan(iLabels), 0, iLabels)
        iPRED = torch.where(torch.isinf(iPRED), 5, iPRED)
        if d_iter.dataset.is_train:
            print("训练总损失为: ", nn.MSELoss()(iPRED, iLabels))
            train_iter = DataLoader(dataset=Post_processing_Dataset(iPRED, iLabels), batch_size=batch_size_2, drop_last=True, shuffle=True)
        else:
            print("测试总损失为: ", nn.MSELoss()(iPRED, iLabels))
            test_iter = DataLoader(dataset=Post_processing_Dataset(iPRED, iLabels), batch_size=batch_size_2, shuffle=True, drop_last=True)

    for param in net.net1.parameters():
        param.requires_grad = False
    for param in net.net2.parameters():
        param.requires_grad = True
    trainer = torch.optim.Adam(net.net2.parameters(), lr=lr_2, weight_decay=2e-3)
    for epoch in range(epochs_2):
        net.train()
        ls = []
        for batch in train_iter:
            feature, fire = [x.to(torch.float32).to(device) for x in [batch['feature'], batch['fire']]]
            fire_hat = net(feature)
            l = post_loss(fire, fire_hat)
            ls.append(l.cpu().detach())

            trainer.zero_grad()
            l.backward()
            trainer.step()
            print(f"\repoch: {epoch+1}, loss: {l}", end='')
        Train_Loss.append(torch.tensor(ls).mean())
    
        net.eval()
        ls = []
        for batch in test_iter:
            feature, fire = [x.to(torch.float32).to(device) for x in [batch['feature'], batch['fire']]]
            fire_hat = net(feature)
            ls.append(post_loss(fire, fire_hat).cpu().detach())
        test_Loss.append(torch.tensor(ls).mean())
        print(f"   vloss: {torch.tensor(ls).mean()}", end='')

        
    plt.plot(Train_Loss, label='Train')
    plt.plot(test_Loss, label='Test')
    plt.legend()
    plt.savefig("./visuals/Loss_2.png", dpi=300)
    plt.clf()
    plt.close()
    
    torch.save(net.state_dict(), 'checkpoints/lst.pt')

def test(checkpoint='checkpoints/lst.pt'):    
    device = "cuda:0"

    net = Hierarchical_Model(64)
    net.to(device)
    net.load_state_dict(torch.load(checkpoint))
    net.eval()
    

    data, lsm = get_data()
    mask_data, lsm = get_down_interpolate_data(data, lsm)
    features, fire, lsm, fea_means, fea_stds = get_norm_data(mask_data, lsm)
    for d_iter in [DataLoader(dataset=Amplitude_frequency_increment_Dataset(features, fire), batch_size=1, shuffle=False),
                   DataLoader(dataset=Amplitude_frequency_increment_Dataset(features, fire, is_train=False), batch_size=1, shuffle=False)]:
        PRED = []
        Labels = []
        for batch in tqdm.tqdm(d_iter):
            feature, fire = [x.to(torch.complex64).to(device) for x in [batch['feature'], batch['fire']]]
            fire_hat = net(feature)
            PRED.append(fire_hat.cpu().detach())
            Labels.append(fire.cpu().detach())
        PRED = torch.concatenate(PRED, dim=0)
        Labels = torch.concatenate(Labels, dim=0)
        iPRED = imodel_data(PRED, lsm, d_iter).cpu().detach()
        iLabels = imodel_data(Labels, lsm, d_iter).cpu().detach() 
        iPRED = torch.where(torch.isnan(iPRED), 0, iPRED)
        iLabels = torch.where(torch.isnan(iLabels), 0, iLabels)
        iPRED = torch.where(torch.isinf(iPRED), 5, iPRED)
        if d_iter.dataset.is_train:
            print("训练总损失为: ", nn.MSELoss()(iPRED, iLabels))
            train_iter = DataLoader(dataset=Post_processing_Dataset(iPRED, iLabels), batch_size=1, drop_last=True, shuffle=False)
        else:
            print("测试总损失为: ", nn.MSELoss()(iPRED, iLabels))
            test_iter = DataLoader(dataset=Post_processing_Dataset(iPRED, iLabels), batch_size=1, shuffle=True, drop_last=False)

    PRED = []
    Labels = []
    for batch in train_iter:
        feature, fire = [x.to(torch.float32).to(device) for x in [batch['feature'], batch['fire']]]
        fire_hat = net(feature)
        PRED.append(fire_hat)
        Labels.append(fire)
    train_PRED, train_Labels = [torch.concatenate(x).cpu().detach().squeeze(1) for x in [PRED, Labels]]
    print("训练总损失为: ", torch.nn.functional.mse_loss(train_PRED, train_Labels))   
    
    PRED = []
    Labels = []
    for batch in test_iter:
        feature, fire = [x.to(torch.float32).to(device) for x in [batch['feature'], batch['fire']]]
        fire_hat = net(feature)
        PRED.append(fire_hat)
        Labels.append(fire)
    test_PRED, test_Labels = [torch.concatenate(x).cpu().detach().squeeze(1) for x in [PRED, Labels]]
    print("测试总损失为: ", torch.nn.functional.mse_loss(test_PRED, test_Labels))

    row, col = 2, 12
    fig, axes = plt.subplots(row*2, col)
    for i in range(row):
        for j in range(col):
            axes[i][j].imshow(test_Labels[i*col+j], cmap='Reds')
    for i in range(row):
        for j in range(col):
            axes[i+row][j].imshow(test_PRED[i*col+j], vmin=0, vmax=Labels[i*col+j].max(), cmap='Reds')
    plt.savefig('test_visual.png', dpi=300)


if __name__ == "__main__":
    train()
    test()