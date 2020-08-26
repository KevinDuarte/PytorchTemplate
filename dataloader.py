from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms



def get_mnist_dataset(root='./data/', train=True, download=True):
    transform = transforms.Compose([transforms.ToTensor()])  # used to transform PIL image to pytorch tensor

    return MNIST(root=root, train=train, download=download, transform=transform)


class TrainDataset(Dataset):
    def __init__(self):
        self.data = []  # this should hold a list of all samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]

        x, y = 0, 0  # placeholder for the sample input(s) and output(s)

        return x, y


class ValidationDataset(Dataset):
    def __init__(self):
        self.data = []  # this should hold a list of all samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]

        x, y = 0, 0  # placeholder for the sample input(s) and output(s)

        return x, y

