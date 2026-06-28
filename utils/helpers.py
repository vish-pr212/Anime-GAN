import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torch
import os
import matplotlib.pyplot as plt

def show_images(batch, num_images=16):
    grid = vutils.make_grid(batch[:num_images], nrow=4, normalize=True)
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.show()

def create_folders():

    os.makedirs("outputs/images", exist_ok=True)
    os.makedirs("outputs/checkpoints", exist_ok=True)


def save_generated_images(generator,fixed_noice,epoch):

    generator.eval()

    with torch.no_grad():

        fake_images = generator(fixed_noice)

        grid = vutils.make_grid(
            fake_images,
            normalize=True,
            nrow=8
        )

        vutils.save_image(
            grid,
            f"outputs/images/epoch_{epoch+1:03d}.png"
        )

    generator.train()


def plot_losses(generator_losses,discriminator_losses):

    plt.figure(figsize=(10,5))

    plt.plot(generator_losses,
             label="Generator Loss")

    plt.plot(discriminator_losses,
             label="Discriminator Loss")

    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("GAN Training")

    plt.legend()

    plt.grid(True)

    plt.show()