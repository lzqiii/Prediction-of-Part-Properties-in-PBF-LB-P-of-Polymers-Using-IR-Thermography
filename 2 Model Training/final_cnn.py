import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from resnet_cbam import resnet50_cbam, resnet34_cbam
from torchvision.models import resnet34,resnet50
# tensorboard --logdir=E:\Thesis\code\final\runs
# Enter this code in the cmd to visualize the training process, and note the change of directory
log_writer = SummaryWriter()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameter setting
in_channel = 1
num_classes = 1
learning_rate = 0.0001
eta_min = 0.00001
batch_size = 12
num_epochs = 200

data_transform = transforms.Compose([
    transforms.Resize(360),  # Scaling the image (Image), keeping the aspect ratio constant
    transforms.Grayscale(1),
    transforms.ToTensor(),  # Convert Image to Tensor, normalize to [0, 1]
    transforms.Normalize(mean=0.0601, std=0.1734)  # Standardized to [-1, 1], specifying the mean and standard deviation
])
train_dataset = datasets.ImageFolder(root="E:/Thesis/code/final/target_data/train/",
                                     transform=data_transform)
test_dataset = datasets.ImageFolder(root="E:/Thesis/code/final/target_data/test/",
                                    transform=data_transform)
validation_dataset = datasets.ImageFolder(root="E:/Thesis/code/final/target_data/val/",
                                          transform=data_transform)
# Data loader
train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=True)
test_loader = DataLoader(test_dataset,
                         batch_size=1,
                         shuffle=False)
validation_loader = DataLoader(validation_dataset,
                               batch_size=1,
                               shuffle=False)
"""
# Available for comparison using Resnet50
model = resnet50(pretrained=False, num_classes=num_classes)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.to(device)
"""
model = resnet34_cbam(pretrained=False, num_classes=num_classes).to(device)
# Set the loss function and optimizer
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(abs(num_epochs)), eta_min=0)
valid_loss_min = np.Inf


def convert_target(target):
    """Change the folder name from string to float. train"""
    i = 0
    y = torch.zeros_like(target)
    y = y.to(torch.float32)
    for item in target:
        y[i] = float(train_dataset.classes[int(item)])
        i = i + 1
    y = y + torch.normal(0, 0.0001, size=(y.size()))
    y = torch.unsqueeze(y, dim=1)
    # print(y)
    return y


def convert_target2(target):
    """Change the folder name from string to float. val"""
    i = 0
    y = torch.zeros(target)
    y = y.to(torch.float32)
    for item in target:
        y = float(validation_dataset.classes[int(item)])
    y = torch.tensor(y)
    y = torch.unsqueeze(y, dim=0)
    y = torch.unsqueeze(y, dim=1)
    # print(y)
    return y


def convert_target3(target):
    """Change the folder name from string to float. test"""
    i = 0
    y = torch.zeros(target)
    y = y.to(torch.float32)
    for item in target:
        y = float(test_dataset.classes[int(item)])
    y = torch.tensor(y)
    y = torch.unsqueeze(y, dim=0)
    y = torch.unsqueeze(y, dim=1)
    # print(y)
    return y


# The following part is training
def train(model, device, train_loader, optimizer, epoch):
    model.to(device)
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = convert_target(target).to(device)
        data_v = Variable(data)
        target_v = Variable(target)
        # print(target_v)
        optimizer.zero_grad()  # 梯度归零
        output = model(data_v)
        loss = criterion(output, target_v)
        # print(loss)
        loss.backward()
        optimizer.step()  # 更新梯度
        train_loss = loss + train_loss
        if (batch_idx + 1) % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
    log_writer.add_scalar('Loss/train', float(train_loss / len(train_loader)), epoch)


def validation(model, device, validation_loader, epoch):
    global valid_loss_min
    model.eval()
    validation_loss = 0
    with torch.no_grad():
        for data, target in validation_loader:
            data = data.to(device)
            target = convert_target2(target).to(device)
            output = model(data)
            validation_loss += criterion(output, target).item()  # 将一批的损失相加
    validation_loss /= len(validation_loader.dataset)
    log_writer.add_scalar('Loss/validation', float(validation_loss), epoch)
    print('EPOCH:{}    Validation set: Average loss: {:.4f}'.format(epoch, validation_loss))
    if validation_loss < valid_loss_min:
        valid_loss_min = validation_loss
        print("saving model ...")
        torch.save(model.state_dict(), 'res_34.pt')

"""
# You can turn this paragraph into a comment when you just want to verify
for epoch in range(1, num_epochs + 1):
    train(model, device, train_loader, optimizer, epoch)
    validation(model, device, validation_loader, epoch)
    scheduler.step()

log_writer.close()
torch.save(model.state_dict(), 'res_222.pt')
"""
# Functions for evaluating accuracy
def check_accuracy(loader, model):
    # if loader.dataset.train:
    # print("Checking acc on training data")
    # else:
    # print("Checking acc on testing data")
    num_correct = 0
    num_samples = 0
    validation_loss = 0
    model.load_state_dict(torch.load('res_34.pt'))
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            # print(x)
            y = convert_target3(y).to(device)
            # print(y)
            predictions = model(x)
            # 64*10
            print('pre', predictions)
            print('real', y)
            validation_loss = criterion(predictions, y) + validation_loss
            num_samples += 1
    # print(validation_loss)
    acc = validation_loss / num_samples
    acc = float(100 * (1 - acc))

    print(f'Got {num_samples} with accuracy {acc:.2f}')

    model.train()
    return acc


check_accuracy(test_loader, model)
