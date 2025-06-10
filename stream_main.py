import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


PATH = './waste_mgmt_self_defined.pth'



file_up = st.file_uploader("Upload an image", type="jpg")

from PIL import Image
image = Image.open(file_up)
st.image(image, caption='Uploaded Image.', use_column_width=True)

#Transform image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    ])

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 61 * 61, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
net.load_state_dict(torch.load(PATH, weights_only=True))

img = Image.open(file_up)
batch_t = torch.unsqueeze(transform(img), 0)
net.eval()
out = net(batch_t)
probs = torch.sigmoid(out)
return probs