import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transform

class AnimeDataset(Dataset):
    def __init__(self,root_dir,image_size=64):
        self.root_dir = root_dir
        self.image_paths = [
            os.path.join(root_dir,img)
            for img in os.listdir(root_dir)
            if img.endswith((".jpg", ".png", ".jpeg"))
        ]
        
        self.transform = transform.Compose([
            transform.Resize((image_size,image_size)),
            transform.ToTensor(),
            transform.Normalize([0.5],[0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image