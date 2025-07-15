import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
import boto3
import os

AWS_S3_BUCKET_NAME = 'waste-mgmt-zhenging'
AWS_REGION = 'us-east-1'
# AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
# AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

os.makedirs("tmp", exist_ok=True)

#comment out AWS access key when using ECS, instead add a IAM role with access to s3. Can use this locally
s3_client = boto3.client(
    service_name='s3',
    region_name=AWS_REGION
    # aws_access_key_id=AWS_ACCESS_KEY,
    # aws_secret_access_key=AWS_SECRET_KEY
)

s3_client.download_file(AWS_S3_BUCKET_NAME, 'waste_mgmt_self_defined.pth', 'tmp/waste_mgmt_self_defined.pth')



PATH = 'tmp/waste_mgmt_self_defined.pth'
# vgg_PATH = './waste_mgmt_vgg.pth'

classes = ("Organic", "Recyclable")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def predict(image_bytes):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])
    img = Image.open(io.BytesIO(image_bytes))
    batch_t = torch.unsqueeze(transform(img), 0)
    net = Net()
    net.eval()
    net.load_state_dict(torch.load(PATH, map_location=device, weights_only=True))
    out = net(batch_t)
    probs = torch.sigmoid(out)
    predicted_class = (probs >= 0.5).int().item()
    return classes[predicted_class]



