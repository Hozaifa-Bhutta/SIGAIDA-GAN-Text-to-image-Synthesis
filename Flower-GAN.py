import os
import random
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as tt
import torch
import torch.nn as nn
import cv2
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import os
import numpy as np



# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True) # Needed for reproducible results



DATA_DIR = "main_data/" # had to create a subfolder within main_data to for ImageFolder not to break

image_size = 64 # signficantly reduces size
batch_size = 64
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) # normalizes values between -1-1, makes it more convenient for discriminator training

# number of channels in the iamge
nc = 3
latent_size = 100 #input "noise" vector size

# size of feature map in generator
ngf = 64

# size of feature map in discriminator
ndf = 64


training_dataset = datasets.ImageFolder(DATA_DIR, transform = tt.Compose([
    tt.Resize(image_size), # resizes it to 'image_size'
    tt.CenterCrop(image_size),
    tt.ToTensor(),
    tt.Normalize(*stats) # normalizes values between -1-1, makes it more convenient for discriminator training
])) # check if you need crop + resize
train_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True) # batches training_dataset up and makes it iterable
# num_workers is the number of sub_processes on the CPU that loads the data
# example uses pin_memory which is only applciable for gpu usage

device = torch.device("mps") # apparently this works?!
print(device)


# displqy plot (not working for now)
example_batch = next(iter(train_dataloader))
plt.figure(figsize=(8,8)) # 8 by 8 grid
plt.title("Training Images")
plt.axis("off")
plt.imshow(np.transpose(make_grid(example_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
plt.show()


# custom weights initialization called on ``netG`` and ``netD``
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    """This class is the template for our generator module"""
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(

            #in = 128 x 1 x 1
            nn.ConvTranspose2d(latent_size, ngf * 8, kernel_size=4, stride=1, padding=0, bias=False), # upscales image  (can cause artifacts, look into this)
            nn.BatchNorm2d(ngf * 8), # normalizes data
            nn.ReLU(True), # activation function that turns 0s into 1s
            # out = 512 x 4 x 4
        
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias = False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # out = 512 x 8 x 8

            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride = 2, padding = 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # out = 128 x 16 x 16

            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride = 2, padding = 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # out = 64 x 32 x 32

            nn.ConvTranspose2d(64, nc, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
            # out = 3 (RGB channels) x 64 (size) x 64 (size)
        )
    def forward(self, x):
       
        output = self.model(x).to(device)
        return output

class Discriminator(nn.Module):
    """Module for discriminator"""
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # in 3 x 64 x 64
            nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 64 x 32 x 32

            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # out = 128 x 16 x 16

            nn.Conv2d(ndf*2, ndf * 4, kernel_size = 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # out = 256 x 8 x 8

            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # out = 512 x 4 x 4

            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride = 1, padding=0, bias=False),
            # out = 1 x 1 x 1

            nn.Sigmoid(), # activation

            nn.Flatten()
        )
    def forward(self, x):
        output = self.model(x).to(device)
        return output

generator = Generator().to(device=device)
generator.apply(weights_init)
print(generator)

discriminator = Discriminator().to(device=device)
discriminator.apply(weights_init)
print(discriminator)


sample_dir = 'generated'
os.makedirs(sample_dir, exist_ok=True)

def save_samples(index, latent_tensors, show=True):
    fake_images = generator(latent_tensors)
    fake_fname = 'generated-image-{}.png'.format(index)
    save_image(fake_images, os.path.join(sample_dir, fake_fname), nrow=8)
    print(f"Saving {fake_fname}")
    if show:
        fig, ax = plt.subplot(figsize =(8,8))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(fake_images.cpu().detach(), n_row = 8).permute(1,2,0))
        plt.show()

criterion = nn.BCELoss()
def train_step(real_images, opt_g, opt_d):
    discriminator.zero_grad() # zeroes out gradient from previous iter

    # pass real image through discrimiantor
    real_image_pred = discriminator(real_images)
    actual_answer = torch.ones(real_images.size(0), 1, device=device) # size: real_image.size(0) x 1 of ones 
    # loaded onto device
    real_loss = criterion(real_image_pred, actual_answer)
    real_score = torch.mean(real_image_pred).item() # average of all the prediction is the accuracy (since all of them should be 1)
    # Update discriminator weight
    real_loss.backward()

    # Generate fake image
    latent = torch.randn(batch_size, latent_size, 1, 1, device=device) # size: batch_size x latent_size x 1 x 1
    fake_images = generator(latent)

    # Pass fake image through discriminator
    fake_image_pred = discriminator(fake_images.detach())
    actual_answer = torch.zeros(fake_images.size(0), 1, device=device) # a vector of all 0s
    fake_loss = criterion(fake_image_pred, actual_answer)
    fake_loss.backward()
    fake_score = 1 - torch.mean(fake_image_pred).item()

    # Update discriminator weight
    opt_d.step()

    ############################################
    # TRAINING GENERATOR
    ############################################
    generator.zero_grad() # zeroes out gradient from previous iter

    # Try to fool the discriminator
    fake_image_pred = discriminator(fake_images) # the exact same thing as before but without the detach
    targets = torch.ones(batch_size, 1, device=device)
    gen_loss = criterion(fake_image_pred, targets)
    gen_score = torch.mean(fake_image_pred).item()
    
    # Update generator weights
    gen_loss.backward()
    opt_g.step()

    # loss and accuracy of discriminator on real images, fake images, and  generator
    return real_loss, fake_loss, gen_loss, real_score, fake_score, gen_score





fixed_latent = torch.randn(1, latent_size, 1, 1, device = device)
def avg(list):
    sum = 0
    for i in range(len(list)):
        sum += list[i]
    average = sum/len(list)
    return average
def fit(epochs, lr_g, lr_d, start_idx = 1):
    torch.cuda.empty_cache() #Releases all unoccupied cached memory currently held by the caching allocator so that those can be used in other GPU application and visible in nvidia-smi.
    size = len(train_dataloader)
   
    # Create optimizer
    opt_d = torch.optim.Adam(discriminator.parameters(), lr = lr_g, betas = (0.5, 0.999)) # Betas are used as for smoothing the path to the convergence also providing some momentum to cross a local minima or saddle point.
    opt_g = torch.optim.Adam(generator.parameters(), lr = lr_d, betas = (0.5, 0.999))
    for epoch in range(epochs):
         #Losses and scores
        loss_discriminator = 0
        discriminator_scores_on_real = 0
        discriminator_scores_on_fake = 0
        loss_generator = 0
        gen_scores = 0

        print(f"Epoch {epoch}\n----------------------")
        # Train discriminator
        for batch_num, (images,_ ) in enumerate(train_dataloader):
            print(f"{batch_num}/{size}")

            # Discriminator(real_images)
            real_images = images.to(device)
            real_loss, fake_loss, gen_loss, real_score, fake_score, gen_score = train_step(real_images, opt_g, opt_d)
            
            # Record losses and scores of discriminator
            loss_discriminator += (real_loss + fake_loss)/2 # loss on fake and real images / 2
            discriminator_scores_on_real += real_score
            discriminator_scores_on_fake += fake_score

            # Record losses and scores of generator
            loss_generator += gen_loss
            gen_scores += gen_score


        # Log losses & scores (last batch)
        print(f"loss_d_average: {loss_discriminator/128:.4f}, discriminator_scores_on_real: {discriminator_scores_on_real/128:.4f}, discriminator_scores_on_fake: {discriminator_scores_on_fake/128:.4f}")
        print(f"loss_g average: {loss_generator/128:.4f}, generator_score: {gen_scores/128:.4f}")

        # Save generated images
        save_samples(epoch+start_idx, fixed_latent, show=False)

    return losses_g, losses_d




hitory = fit(60, 0.0002, 0.0002)
