import torch
from torchvision import transforms
import torch.nn as nn
import os
from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
from torchvision import models
from torch.optim import lr_scheduler
from torch import optim
import json
from torch.utils.data.sampler import SubsetRandomSampler
import scipy.io
import torchvision
import copy
import time
import sys
torch.manual_seed(0)
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

# set paths to labels and image data

data_dir_path = 'data/images/'
labels_path = 'data/imagelabels.mat'
class_label_path = 'data/new_class_label_map'

# standard normalization for Imagenet models
# Augment training data.

data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(45),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
    }


class MyDataset(Dataset):
    def __init__(self, image_labels, data_dir, transform=None):
        """

        :param image_labels_path: path to our labels
        :param root_dir: the directory which houses our images
        :param transform: apply any transform on our sample
        """

        self.image_labels = image_labels
        self.root_dir = data_dir
        self.transform = transform

    def __len__(self):
        label_dict = scipy.io.loadmat(self.image_labels)
        return len(label_dict['labels'][0])

    def __getitem__(self, idx):
        image_path_list = sorted([os.path.join(self.root_dir, filename) for filename in os.listdir(self.root_dir)])
        image = Image.open(image_path_list[idx])

        label_dict = scipy.io.loadmat(self.image_labels)
        label_list = label_dict['labels'][0]

        label_list[:] = [i - 1 for i in label_list]
        label = label_list[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


image_datasets = {x: MyDataset(image_labels=labels_path,
                                  data_dir=data_dir_path,
                               transform=data_transforms[x]) for x in ['train', 'valid', 'test']}


def get_samplers(dataset, shuffle=True, first_split=0.8, second_split=0.9, random_seed=1):
    """

    :param dataset: dataset
    :param shuffle: shuffles the indices so random elements from our data find themselves in our splits.
    :param first_split: this is for train
    :param second_split: for validation
    :param random_seed: for test
    :return: train, validation and test loaders
    """
    dataset_size = len(dataset)  # get any, we just need the length, say its n
    indices = list(range(dataset_size))  # 0 though n-1.
    split1 = int(np.floor(first_split * dataset_size))
    split2 = int(np.floor(second_split * dataset_size))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx = indices[:split1]
    val_idx = indices[split1:split2]
    test_idx = indices[split2:]

    samplers = {'train': SubsetRandomSampler(train_idx), 'valid': SubsetRandomSampler(val_idx),
                'test': SubsetRandomSampler(test_idx)}

    return samplers


samplers = get_samplers(image_datasets['train'])

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, sampler=samplers[x]) for x in
                  ['train', 'valid', 'test']}


dataset_size = {x: len(samplers[x]) for x in ['train', 'valid', 'test']}


def get_class_label_map(path):
    """the class number to the flower name dictionary"""

    with open(path) as f:
        data = json.load(f)

    return data


def imshow(image, ax=None, title=None):
    """to visuliaze any image passed to it"""

    if ax is None:
        fig, ax = plt.subplots()
    if title:
        plt.title(title)

    if isinstance(image, np.ndarray):
        image = image.transpose((1, 2, 0))
    elif isinstance(image, torch.Tensor):
        image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax


def get_loaders(dataloaders, num_images, nrow, key='train'):
    """purpose of this function is to show how train transforms augment our training data"""

    imgs, lbls = next(iter(dataloaders[key]))
    imgs_lst = [imgs[i] for i in range(num_images)]
    out = torchvision.utils.make_grid(imgs_lst, nrow=nrow)
    lbls_lst = [lbls[i].item() for i in range(num_images)]

    data = get_class_label_map(class_label_path)

    data = {int(k): v for k, v in data.items()}

    flowers = []

    for item in lbls_lst:
        if item in data:
            flowers.append(data[item])

    return out, flowers


class Neural(nn.Module):

    def __init__(self, input_size, hidden_layers, output_size, drop_p=0.5):
        super(Neural, self).__init__()  # inherits the __init__ method of the parent class Module.
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend(nn.ModuleList([nn.Linear(h1, h2) for h1, h2 in layer_sizes]))

        self.output = nn.Linear(hidden_layers[-1], output_size)
        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, x):
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            x = self.dropout(x)

        x = self.output(x)

        return F.log_softmax(x, dim=1)  # log(e^xi/sum(e^xj)) # j = 1 through n; n = # of classes.


z = Neural(2100, [540], 102)

# TRAIN FUNCTION ONE


def train_model(model, criterion, optimizer, scheduler, num_epochs=3, device='cuda'):

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('{}/{}'.format(epoch + 1, num_epochs))
        print('-' * 50)

        # each epoch has a train and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for images, labels in dataloaders[phase]:

                # convert ByteTensor to LongTensor
                labels = labels.type(torch.LongTensor)

                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()

                # when evaluates to true tracks history, else compute only forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    logps = model.forward(images)
                    _, preds = logps.max(dim=1)
                    # torch.exp(_) would gives back probs.
                    loss = criterion(logps, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_size[phase]
            epoch_acc = running_corrects.double() / dataset_size[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model


model = models.vgg19(pretrained=True)

classifier = Neural(25088, [4096], 102)

for param in model.parameters():
    param.requires_grad = False

model.classifier = classifier

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

num_epochs = 10


def load_model(filepath):
    checkpoint = torch.load(filepath)

    if checkpoint['arch'] == 'vgg19':
        model = models.vgg19(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False

    else:
        print('Please load the correct pre_trained architecture')
        sys.exit()

    classifier = Neural(checkpoint['input'], checkpoint['hidden_layers'], checkpoint['output'])

    model.classifier = classifier

    model.load_state_dict(checkpoint['state_dict'])

    return model


# use this new_model to predcit accuracy and loss


def calc_acc(model, data='test', device='cpu'):

    test_loss = 0
    test_acc = 0

    for inputs, labels in dataloaders[data]:
        model.to(device)
        model.eval()
        inputs, labels = inputs.to(device), labels.to(device)
        labels = labels.type(torch.LongTensor)

        with torch.no_grad():
            logps = model.forward(inputs)  # batch_size * number of classes
            loss = criterion(logps, labels)  # computes loss for the current batch.

            logprobs, preds = logps.max(dim=1)
            matches = (preds == labels)

            test_loss += loss.item()
            test_acc += matches.type(torch.FloatTensor).mean()

        # no epoch loss since we have no epochs to test

    return test_loss, test_acc


def predict(loader, data, model, idx):
    dataloader = loader[data]
    imgs, lbls = next(iter(dataloader))
    img, lbl = imgs[idx], lbls[idx].item()

    model.eval()

    probs = torch.exp(model.forward(img.unsqueeze(0)))
    top_5_probs, top_5_lbls = probs.topk(5)

    top_5_probs = top_5_probs.detach().numpy().tolist()[0]
    top_5_lbls = top_5_lbls.detach().numpy().tolist()[0]

    class_label_map = get_class_label_map(class_label_path)
    class_label_map = {int(k): v for k, v in class_label_map.items()}

    true_title_name = class_label_map[lbl]

    pred_flower_names = [class_label_map[key] for key in top_5_lbls]

    return img, top_5_probs, true_title_name, pred_flower_names


def plt_solution(loader, data, model, idx):
    plt.figure(figsize=(6, 10))
    ax = plt.subplot(2, 1, 1)

    img, top_5_probs, true_title_name, pred_flower_names = predict(loader, data, model, idx)

    imshow(img, ax, title=true_title_name)

    # # Plot bar chart
    plt.subplot(2, 1, 2)
    sns.barplot(x=top_5_probs, y=pred_flower_names, color=sns.color_palette()[0])
    plt.show()


def main():
    imgs, names = get_loaders(dataloaders, 8, 4)
    imshow(imgs, title=names)
    plt.show()

    model_ft = train_model(model = model, criterion=criterion, optimizer=optimizer, scheduler=scheduler,
                num_epochs=num_epochs, device='gpu')

    checkpoint = {'arch': 'vgg19', 'state_dict': model.state_dict(),
    'input': 25088, 'output': 102,'hidden_layers': [each.out_features for each in classifier.hidden_layers]}
    torch.save(checkpoint, 'classifier.pth')

    new_model = load_model('classifier.path')

    test_loss, test_accuracy = calc_acc(model=new_model, data='test', device='cpu')
    print(test_loss)
    print(test_accuracy)

    # this inturn calls predict.
    plt_solution(loader=dataloaders, data='test', model=new_model, idx=5)


if __name__ == "__main__":
    main()









