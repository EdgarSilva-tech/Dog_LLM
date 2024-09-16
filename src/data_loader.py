from torchvision import datasets, transforms
from torch.utils.data import DataLoader

train_dir=r"C:\Users\edgar\OneDrive\Ambiente de Trabalho\AI_Projects\Dog_LLM\data\train"
val_dir=r"C:\Users\edgar\OneDrive\Ambiente de Trabalho\AI_Projects\Dog_LLM\data\validation"

train_transform = transforms.Compose([
    # Resize the images to 64x64
    transforms.Resize(size=(64, 64)),
    # Flip the images randomly on the horizontal
    transforms.TrivialAugmentWide(num_magnitude_bins=31), # p = probability of flip, 0.5 = 50% chance
    # Turn the image into a torch.Tensor
    transforms.ToTensor() # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0 
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor()
])

def load_data(train_data: str = train_dir, val_data: str = val_dir, train_transform: transforms.Compose = train_transform, val_transform: transforms.Compose = val_transform):
    train_data = datasets.ImageFolder(root=train_dir, transform=train_transform, target_transform=None)
    val_data = datasets.ImageFolder(root=val_dir, transform=val_transform)

    class_names = train_data.classes

    train_dataloader = DataLoader(dataset=train_data, batch_size=32, num_workers=1, shuffle=True)

    val_dataloader = DataLoader(dataset=val_data, batch_size=32, num_workers=1, shuffle=False)

    return train_dataloader, val_dataloader, class_names

