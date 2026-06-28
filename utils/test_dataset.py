from torch.utils.data import DataLoader
from dataset import AnimeDataset
from helpers import show_images

dataset = AnimeDataset(r"C:\Users\pc\Downloads\images",image_size=64)
dataloader = DataLoader(dataset,batch_size=16,shuffle=True)

batch = next(iter(dataloader))
print("Batch shape:", batch.shape)

show_images(batch)