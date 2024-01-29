import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

# sample image for testing
# image = Image.open(f'{root_dir}/data/images/train/a/abbey/00000001.jpg')
# print(f'Data type of my image: {type(image)}')
# print(f"Shape of image: {np.array(image).shape}")
# print(f'Channel mode:', image.mode)
# print(f"Value range of image: {image.getextrema()}")
# plt.imshow(image)
# plt.title('PIL Image')
# plt.axis('off')

class MiniPlaces(Dataset):
    def __init__(self, root_dir, split, transform=None, label_dict=None):
        assert split in ['train', 'val', 'test']
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.filenames = []
        self.labels = []
        self.label_dict = label_dict if label_dict is not None else {}

        with open(root_dir + '/' + split + '.txt', 'r') as file:
          for line in file:
              [filename, label] = line.strip().split()
              self.filenames.append(filename)
              self.labels.append(int(label))
              if (split == 'train'):
                self.label_dict[int(label)] = filename.split('/')[2]

    def __len__(self):
        dataset_len = len(self.filenames)
        return dataset_len

    def __getitem__(self, idx):
        image = Image.open(f'{self.root_dir}/images/{self.filenames[idx]}')
        if (self.transform):
          image = self.transform(image)
        label = self.labels[idx]
        return image, label

image_net_mean = torch.Tensor([0.485, 0.456, 0.406])
image_net_std = torch.Tensor([0.229, 0.224, 0.225])

data_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=image_net_mean, std=image_net_std),
    transforms.Lambda(lambda x: torch.flatten(x))
])

def tensor_to_image(image):
    img_tensor      = image.clone().detach()          # we make a copy of the tensor
    img_reshaped    = img_tensor.view(3, 32, 32)      # reshape image vector to (3, 32, 32)
    img_denormed    = transforms.Normalize(mean=-image_net_mean/image_net_std, std=1/image_net_std)(img_reshaped)
    img_transposed  = img_denormed.permute(1, 2, 0)   # transpose to (32, 32, 3)
    numpy_arr = img_transposed.numpy()
    return numpy_arr



# TEST LOADER:

seed_everything(0)

data_root = os.path.join(root_dir, 'data')
miniplaces_train = MiniPlaces(data_root,
                              split='train',
                              transform=data_transform)

print('len of trainining dataset:', len(miniplaces_train))
print('label_dict:', miniplaces_train.label_dict)
print(len(miniplaces_train))

random_idxs = np.random.choice(len(miniplaces_train), 3)
print('Filenames:',
      [miniplaces_train.filenames[i] for i in random_idxs])
print('Class IDs:', [miniplaces_train.labels[i] for i in random_idxs])
print('Class names:', [
    miniplaces_train.label_dict[miniplaces_train.labels[i]]
    for i in random_idxs
])

miniplaces_val = MiniPlaces(data_root, split='val', transform=data_transform)
print('val label_dict:', miniplaces_val.label_dict)


# LOADING

batch_size = 64
num_workers = 2

train_loader = DataLoader(miniplaces_train,
                          batch_size=batch_size,
                          num_workers=num_workers,
                          shuffle=True)
val_loader = DataLoader(miniplaces_val,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        shuffle=False)