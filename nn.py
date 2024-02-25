"""
if using Harvey GPU, run the following command to install pytorch:
    "conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia"
otherwise:
    "conda install pytorch torchvision torchaudio cpuonly -c pytorch"
"""
from mpmath.identification import transforms
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, PILToTensor
from torch.utils.data import DataLoader
from torch import nn
from torch.nn.functional import relu, dropout, max_pool2d, softmax
from torch import optim
from PIL import Image
import torch

BATCH_SIZE = 100
NUM_EPOCHS = 1

train_data = datasets.MNIST(
    root='data',
    train=True,
    transform=ToTensor()
)

test_data = datasets.MNIST(
    root='data',
    train=False,
    transform=ToTensor()
)

train_dataloader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=1
)

test_dataloader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=1
)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(784, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.linear_relu_stack(x)

        return softmax(x, dim=1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNetwork().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()


def train():
    print('Beginning training.')
    model.train()
    for batch_idx, (images, labels) in enumerate(train_dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()


def test():
    print('Beginning testing.')
    model.eval()
    correct = 0

    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()


if __name__ == '__main__':
    train()
    test()

    print('Finished!')

    print('Importing image.')
    image = Image.open("num_6.png")
    input_transform = Compose([
        PILToTensor()
    ])

    img_tensor = input_transform(image).float()

    vector_output = model(img_tensor.unsqueeze(0).to(device))
    print(vector_output.shape)
    pred_1 = vector_output.argmax(dim=1, keepdim=True)
    print(pred_1.item())
