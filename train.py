import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from Models.generator import Generator
from Models.discriminator import Discriminator
from utils.dataset import AnimeDataset
from utils.helpers import create_folders,save_generated_images,plot_losses
from utils.checkpoint import save_checkpoint,load_checkpoint
from utils.gradient_penalty import gradient_penalty

#hyperparameters
batch_size=64
lr=0.0002
noise_dim=100
epochs =52
n_critic = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("working device: ",device)

#dataset
dataset = AnimeDataset(r"C:\Users\pc\Downloads\images",image_size=64)
loader = DataLoader(dataset,batch_size=batch_size,shuffle=True)

#loading models
D = Discriminator().to(device)
G = Generator(noise_dim).to(device)

#loss function and optimization
opt_G = optim.Adam(G.parameters(),lr=lr,betas=(0.5,0.999))
opt_D = optim.Adam(D.parameters(),lr=lr,betas=(0.5,0.999))


create_folders()

generator_losses = []
discriminator_losses = []

start_epoch = load_checkpoint(
    G,
    D,
    opt_G,
    opt_D
)

fixed_noise = torch.randn(64, noise_dim, 1, 1, device=device)
for epoch in range(start_epoch,epochs):
    print(f"\n Epoch {epoch+1}/{epochs} started")
    loop = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")

    for i, real_image in enumerate(loop):
        real_img=real_image.to(device)
        batch_size=real_image.size(0)

        real_label = torch.ones(batch_size,device=device)
        fake_label = torch.zeros(batch_size,device=device)


        #training discriminator
        for _ in range(n_critic):

            # Generate new fake each critic step
            z = torch.randn(batch_size, noise_dim, 1, 1, device=device)
            fake_img = G(z)

            # Real + fake scores
            real_scores = D(real_img)
            fake_scores = D(fake_img.detach())

            # Gradient penalty
            gp = gradient_penalty(D, real_img, fake_img.detach(), device)

            # WGAN-GP loss
            loss_D = -(torch.mean(real_scores) - torch.mean(fake_scores)) + 10 * gp

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()


        #training Generator

        fake_scores = D(fake_img)
        loss_G = -torch.mean(fake_scores)

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

        loop.set_postfix(
            D_loss=loss_D.item(),
            G_loss=loss_G.item()
        )

        generator_losses.append(loss_G.item())
        discriminator_losses.append(loss_D.item())

    print(f"Epoch [{epoch+1}/{epochs}]  Loss D: {loss_D.item():.4f}  Loss G: {loss_G.item():.4f}")
    save_generated_images(G,fixed_noise,epoch)
    save_checkpoint(G,D,opt_G,opt_D,epoch)


plot_losses(generator_losses,discriminator_losses)