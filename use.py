
import matplotlib.pyplot as plt
import torch
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)
class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
batch_size = 128
nz = 100
ngf = 64
ndf = 64
nc = 3  
netG = Generator(nz, ngf, nc)
netG.load_state_dict(torch.load('generator.pth'))
netG.eval()  
b_size = 64
transform = transforms.Compose([transforms.Resize(64),
                                transforms.CenterCrop(64),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataset = datasets.Flowers102(root='data', split='train', transform=transform, download=True)

noise = torch.randn(b_size, nz, 1, 1)
with torch.no_grad():
    fake_images = netG(noise)
real_images, _ = next(iter(dataset))

def denormalize(tensor):
    tensor = tensor * 0.5 + 0.5 
    return tensor
num_images = min(len(real_images), len(fake_images), 5)  

fig, axs = plt.subplots(2, num_images, figsize=(15, 6))  

real_images = real_images / 255.0 if real_images.max() > 1 else real_images
fake_images = fake_images / 255.0 if fake_images.max() > 1 else fake_images
print(real_images)
import numpy as np
for i in range(num_images):
    ax = axs[0, i]
    real_imag = real_images.cpu().permute(1, 2, 0).numpy() 
    print(real_imag.shape)
    ax.imshow(real_imag)
    ax.axis('off')
    ax.set_title('Real')


for i in range(num_images):
    ax = axs[1, i]
    fake_img = fake_images[i].cpu().permute(1, 2, 0).numpy()  
    print(fake_img.shape)
    ax.imshow(fake_img)
    ax.axis('off')
    ax.set_title('Fake')

plt.tight_layout()
plt.show()