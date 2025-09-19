import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
import numpy as np
import copy


class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model, epoch):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)


class ResNet101(nn.Module):

    def __init__(self, num_classes):
        super(ResNet101, self).__init__()

        self.resnet101 = models.resnet101(pretrained=True)
        self.resnet101.fc = torch.nn.Linear(
            self.resnet101.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet101(x)


class ConvNet(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(ConvNet, self).__init__()

        # Input shape: (B, C, H, W)
        input_channels = input_shape[1]

        self.conv1 = nn.Conv2d(input_channels, 8, kernel_size=3, padding=0)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, stride=2)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=0)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=0)
        self.bn3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU()

        flatten_size = self._get_flatten_size(input_shape)

        self.fc1 = nn.Linear(flatten_size, 500)
        self.relu4 = nn.ReLU()

        self.fc2 = nn.Linear(500, 450)
        self.relu5 = nn.ReLU()

        self.fc3 = nn.Linear(450, num_classes)

    def _get_flatten_size(self, input_shape):
        with torch.no_grad():
            dummy_input = torch.zeros(input_shape)

            x = self.conv1(dummy_input)
            x = self.bn1(x)
            x = self.relu1(x)
            x = self.pool1(x)

            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu2(x)
            x = self.pool2(x)

            x = self.conv3(x)
            x = self.bn3(x)
            x = self.relu3(x)
            # return x.view(1, -1).shape[1]

            return x.view(x.size(0), -1).shape[1]

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu4(x)

        x = self.fc2(x)
        x = self.relu5(x)

        x = self.fc3(x)
        x = F.softmax(x, dim=1)  # Assuming classification task

        return x

if __name__ == "__main__":
    device="cuda"

    image_path = "./data/spectrograms/logmel/fold1/101415-3-0-2_1.png"
    espectrograma = np.asarray(Image.open(image_path))
    espectrograma = (espectrograma / 255).astype(np.float32)

    espec_tensor = torch.from_numpy(espectrograma.copy())
    espec_tensor = espec_tensor.unsqueeze(0)
    espec_tensor = espec_tensor.unsqueeze(0).to(device)

    cn = ConvNet(espec_tensor.shape, 10)
    cn.to(device)
    print(cn(espec_tensor))
