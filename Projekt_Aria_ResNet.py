import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from PIL import Image
from torch import nn
import matplotlib.pyplot as plt
from collections import namedtuple
import zipfile

# Dateien entpacken
extract_to = '/content/Image_data_class6.zip'

# Entpacke die ZIP-Datei
with zipfile.ZipFile('/content/Image_data_class6.zip', 'r') as zip_ref:
    zip_ref.extractall(extract_to)

def image_to_array(image_path):
    with Image.open(image_path) as img:
        return np.array(img)

def data_extract(folder_path):
    images = []
    labels = []
    i = -1
    for filename in os.listdir(folder_path):
        i += 1
        for pic in os.listdir(os.path.join(folder_path, filename)):
            if pic.endswith('.png') or pic.endswith('.jpg'):  # Check file format
                file_path = os.path.join(os.path.join(folder_path, filename), pic)
                img_array = image_to_array(file_path)
                images.append(img_array)
                labels.append(i)
    x = np.array(images)
    y = np.array(labels)
    return x, y

class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(Image.fromarray(image))
        else:
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        return image, label

def one_hot_collate(batch):
    images, labels = zip(*batch)
    labels_one_hot = torch.nn.functional.one_hot(torch.tensor(labels), num_classes=6)
    return torch.stack(images), labels_one_hot

folder_path_train = extract_to + "/train"
folder_path_val = extract_to + "/validation"

x_train, y_train = data_extract(folder_path_train)
x_val, y_val = data_extract(folder_path_val)

train_dataset = CustomDataset(x_train, y_train, transform=ToTensor())
val_dataset = CustomDataset(x_val, y_val, transform=ToTensor())

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2, collate_fn=one_hot_collate)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=2, collate_fn=one_hot_collate)

class BasicBlock(nn.Module):
    
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride = 1, downsample = False):
        super().__init__()
                
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, 
                               stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, 
                               stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace = True)
        
        if downsample:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1, 
                             stride = stride, bias = False)
            bn = nn.BatchNorm2d(out_channels)
            downsample = nn.Sequential(conv, bn)
        else:
            downsample = None
        
        self.downsample = downsample
        
    def forward(self, x):
        
        i = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        if self.downsample is not None:
            i = self.downsample(i)
                        
        x += i
        x = self.relu(x)
        
        return x


class ResNet(nn.Module):
    def __init__(self, config, output_dim):
        super().__init__()
                
        block, n_blocks, channels = config
        self.in_channels = channels[0]
            
        assert len(n_blocks) == len(channels) == 4
        
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        
        self.layer1 = self.get_resnet_layer(block, n_blocks[0], channels[0])
        self.layer2 = self.get_resnet_layer(block, n_blocks[1], channels[1], stride = 2)
        self.layer3 = self.get_resnet_layer(block, n_blocks[2], channels[2], stride = 2)
        self.layer4 = self.get_resnet_layer(block, n_blocks[3], channels[3], stride = 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(self.in_channels, output_dim)
        
    def get_resnet_layer(self, block, n_blocks, channels, stride = 1):
    
        layers = []
        
        if self.in_channels != block.expansion * channels:
            downsample = True
        else:
            downsample = False
        
        layers.append(block(self.in_channels, channels, stride, downsample))
        
        for i in range(1, n_blocks):
            layers.append(block(block.expansion * channels, channels))

        self.in_channels = block.expansion * channels
            
        return nn.Sequential(*layers)
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.fc(h)
        
        return x


def train(model, train_data, epochs, optimizer):
    model.train()
    losses = []
    batches = []

    for epoch in range(epochs):
        epoch_losses = []
        for i, (images, labels) in enumerate(train_data):
            y_pred = model(images)
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(y_pred, labels.float()) 
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            print(f'Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{len(train_data)}], Loss: {loss.item()}')
            batches.append(len(batches) + 1)
            losses.append(loss.item())
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        print(f'Epoch [{epoch+1}/{epochs}] Average Loss: {avg_epoch_loss}')
    
    # Plot the loss
    plt.plot(batches, losses)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Batch')
    plt.show()

def test(model, test_data):
    losses = []
    batches = []
    for i, (images, labels) in enumerate(test_data):
        y_pred = model(images)
        loss = torch.nn.functional.cross_entropy(y_pred, labels.float()) 
        losses.append(loss.item())
        batches.append(i + 1)
    avg_loss = sum(losses) / len(losses)
    print('Average Loss:', avg_loss)
    # Plot the loss
    plt.plot(batches, losses)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Test Loss per Batch')
    plt.show()





ResNetConfig = namedtuple('ResNetConfig', ['block', 'n_blocks', 'channels'])
resnet18_config = ResNetConfig(block=BasicBlock,
                               n_blocks=[2, 2, 2, 2],
                               channels=[64, 128, 256, 512])

model = ResNet(resnet18_config, output_dim=6)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs=2
train(model,train_loader, epochs, optimizer)
test(model, val_dataset)
