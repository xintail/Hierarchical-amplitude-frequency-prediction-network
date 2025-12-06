import torch
from torch.utils.data import Dataset

from utils import rigion_mask


class Amplitude_frequency_increment_Dataset(Dataset):
    def __init__(self, features, fire, is_train=True, train_rate=0.8):
        self.is_train = is_train
        
        fire, self.index = rigion_mask(fire)
        features, _ = rigion_mask(features, index=self.index)

        fire_fft = torch.fft.rfft(fire)
        features_fft = torch.fft.rfft(features)
        if is_train:
            self.fire = fire_fft[:, :, :int(fire_fft.shape[2]*train_rate), :].permute(0,2,1,3).flatten(0,1).unsqueeze(1)
            self.features = features_fft[:, :, :int(features_fft.shape[2]*train_rate), :].permute(0,2,1,3).flatten(0,1).unsqueeze(1) 
        else:
            self.fire = fire_fft[:, :, int(fire_fft.shape[2]*train_rate):, :].permute(0,2,1,3).flatten(0,1).unsqueeze(1)              
            self.features = features_fft[:, :, int(features_fft.shape[2]*train_rate):, :].permute(0,2,1,3).flatten(0,1).unsqueeze(1) 

        
        self._index = torch.tensor([0 if torch.all(torch.abs(x)==0) else 1 for x in self.fire])
        self.features = self.features[self._index.bool(), :, :, :].reshape(-1, 1, 6, 7)
        self.fire = self.fire[self._index.bool(), :, :, :].reshape(-1, 1, 1, 7)

        self.fire_factor = (torch.log(torch.abs(self.fire)+1))/(torch.abs(self.fire)+1e-8)
        self.features_factor = (torch.log(torch.abs(self.features)+1))/(torch.abs(self.features)+1e-8)

        self.fire = self.fire * self.fire_factor
        self.features = self.features * self.features_factor 

    def __len__(self):
        return self.fire.shape[0]
    def __getitem__(self, index):
        return {"feature": self.features[index], "fire": self.fire[index]}
    

class Post_processing_Dataset(Dataset):
    def __init__(self, features, fire):
        self.features = features
        self.fire = fire
    def __len__(self):
        return self.fire.shape[0]
    def __getitem__(self, index):
        return {"feature": self.features[index], "fire": self.fire[index]}