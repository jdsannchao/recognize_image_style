
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader

from torchvision.models import resnet18
from torchvision.models import resnet50

import pandas as pd
import numpy as np

import os
from torchvision.io import read_image
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, target_transform=None):
        self.img_labels = df
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = np.array((self.img_labels.iloc[idx,4:-2]).astype(int))
        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)
            
        return image, torch.tensor(label, dtype=torch.int8)


def train_trans(augment=True):
    augs = transforms.Compose([transforms.ToPILImage(), transforms.Lambda(lambda image: image.convert('RGB')),
                           transforms.Resize((224, 224)),transforms.RandomHorizontalFlip(p=0.9), 
                           transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
                           transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    return augs


def test_trans():
    augs = transforms.Compose([transforms.ToPILImage(), 
                transforms.Lambda(lambda image: image.convert('RGB')),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    return augs

def cross_entropy_one_hot(input, target):
    _, labels = target.max(dim=1)
    return nn.CrossEntropyLoss()(input, labels)



def train_loop(dataloader, model, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X,y =X.to(device), y.to(device)
        # print(y.shape)
        # Compute prediction and loss
        pred = model(X)
        # print(pred.shape)
        # y=y.unsqueeze(1)
        loss = cross_entropy_one_hot(pred, y)
        # loss = nn.BCELoss(pred, y)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 50 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
        scheduler.step()



def acc(dataloader, model):
    """
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X,y =X.to(device), y.to(device)
            pred = model(X)
            test_loss +=  cross_entropy_one_hot(pred, y).item()
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")



## Work Directory inputs
df = pd.read_csv('flickr_df_mar2014-Copy.csv')
df=df[df['retrival']==True]
df['id']=df['id'].astype(str)+'.jpg'  
df_train=df[df['_split']=='train']
df_test=df[df['_split']=='test']

img_dir='./flikr_img/'


## model paremeters
BS=128 # Batch Size

learning_rate =  0.1 * BS/256    # initial_lr, linear scaling

epochs = 20  #10,20,30  

loss='CE' 

## GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

## model

model = resnet18(pretrained=True)
model.fc = nn.Sequential(
    nn.Linear(512, 1024),
    nn.ReLU(True),
    nn.Linear(1024, 20)
) 
model.to(device)


## Dataloader

training_data = CustomImageDataset(df_train, img_dir, train_trans())
testing_data = CustomImageDataset(df_test, img_dir, test_trans())

train_dataloader = DataLoader(training_data, batch_size=BS, shuffle=True)
test_dataloader = DataLoader(testing_data, batch_size=BS, shuffle=True)


loss_fn = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs)

## training
model.train()
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    
print("Done!")

print("Train")
acc(train_dataloader, model)
print("Test")
acc(test_dataloader, model)