import torch
import os

def save_checkpoint(generator,
                    discriminator,
                    optimizer_g,
                    optimizer_d,
                    epoch):

    checkpoint = {

        "epoch": epoch,

        "generator_state_dict":
            generator.state_dict(),

        "discriminator_state_dict":
            discriminator.state_dict(),

        "optimizer_g_state_dict":
            optimizer_g.state_dict(),

        "optimizer_d_state_dict":
            optimizer_d.state_dict()

    }

    torch.save(
        checkpoint,
        "outputs/checkpoints/latest_checkpoint.pth"
    )

    print(f"Checkpoint saved (Epoch {epoch+1})")


def load_checkpoint(generator,
                    discriminator,
                    optimizer_g,
                    optimizer_d):

    checkpoint_path = "outputs/checkpoints/latest_checkpoint.pth"

    if not os.path.exists(checkpoint_path):

        print("No checkpoint found.")

        return 0

    checkpoint = torch.load(checkpoint_path)

    generator.load_state_dict(
        checkpoint["generator_state_dict"]
    )

    discriminator.load_state_dict(
        checkpoint["discriminator_state_dict"]
    )

    optimizer_g.load_state_dict(
        checkpoint["optimizer_g_state_dict"]
    )

    optimizer_d.load_state_dict(
        checkpoint["optimizer_d_state_dict"]
    )

    start_epoch = checkpoint["epoch"] + 1

    print(f"Checkpoint loaded. Resume from Epoch {start_epoch}")

    return start_epoch