from torch import nn
from torch.distributions import Laplace
from torch.utils.data import Dataset, DataLoader
import torch


class ToyNet1(nn.Module):
    def __init__(self):
        super().__init__()
        self.dens1 = nn.Linear(in_features=16, out_features=8)

    def forward(self, x):
        x = self.dens1(x)
        return x


class ToyNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.dens1 = nn.Linear(in_features=8, out_features=4)

    def forward(self, x):
        x = self.dens1(x)
        return x


class ToyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = ToyNet1()
        self.net2 = ToyNet2()

    def forward(self, x):
        x = self.net1(x)
        x = self.net2(x)
        x = Laplace(x, torch.tensor([1.0]))
        return x


class RandomDataset(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, index):
        sample = {'PA': torch.rand(1, 16).float(),
                  'Lateral': torch.rand(1, 16).float(),
                  'text': torch.rand(1, 16).float()}

        label = torch.randint(0, 1, (3,)).float()
        return sample, label

    def __len__(self):
        return 20


if __name__ == '__main__':
    device = torch.device('cuda')
    model = ToyNet()
    model = nn.DataParallel(model)
    model.to(device)
    rand_loader = DataLoader(dataset=RandomDataset(),
                             batch_size=8, shuffle=True)
    for batch, label in rand_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        for modality in batch.keys():
            output = model(batch[modality])
    print('done!')
