import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch import optim
from torchvision import datasets
import CNN
import visdom
import numpy as np
from torchvision import transforms
import copy
import matplotlib.pyplot as plt
import time


# set transforms for traina and validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])
    ]),
    'valid': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])
    ]),
}

# get trainset and testset
trainset = datasets.MNIST('MNIST_data', download=True, train=True,
                                       transform=data_transforms['train'])
validset = datasets.MNIST('MNIST_data', download=True, train=False,
                                 transform=data_transforms['valid'])

# datasets, dataloaders and sizes of each set
datasets_dict = {'train': trainset, 'valid': validset}
dataloaders_dict = {x: torch.utils.data.DataLoader(datasets_dict[x], batch_size=100, shuffle=True) for x in
                    ['train', 'valid']}
dataset_size = {x: len(dataloaders_dict[x].dataset) for x in ['train', 'valid']}


def train(model, dataloaders, optimizer, criterion, num_epochs, is_inception, device='cuda'):
    """trains model and retunrs model state_dict and loss and accuracy for train and validation"""

    start = time.time()

    # train loss and accuracy
    train_loss_history = []
    train_acc_history = []

    # valdiation loss and accuracy
    val_loss_history = []
    val_acc_history = []

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):

        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'=='*20)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0.0

            for inputs, labels in dataloaders[phase]:

                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):

                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model.forward(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 =criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2

                    else:
                        outputs = model.forward(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = outputs.max(dim=1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)

            epoch_loss = running_loss/dataset_size[phase]
            epoch_acc = running_corrects.double()/dataset_size[phase]

            print("{} Loss: {} Acc: {}".format(phase, epoch_loss, epoch_acc))

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc.item())
            elif phase == 'valid':
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc.item())

    time_elapsed = time.time() - start
    print('Time taken to train {:.0f}m {:0f}s'.format(time_elapsed//60, time_elapsed % 60))
    print('best val acc:'.format(best_acc))
    model.load_state_dict(best_model_wts)

    return best_model_wts, train_loss_history, val_loss_history, train_acc_history, val_acc_history


def main():
    # instantiate model and set model training bells and whistles
    model = CNN.MyConv()
    num_epochs = 7
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # call train function
    model_ft, train_loss_history, val_loss_history, train_acc_history, val_acc_history = \
        train(model=model, dataloaders=dataloaders_dict, optimizer=optimizer, criterion=criterion,
              num_epochs=num_epochs, is_inception=False, device='cpu')

    # The visualizations

    # instantiate visdom object
    vis = visdom.Visdom()
    # set windows for loss and accuracy
    loss_win = vis.line(X=torch.zeros((1)), Y=torch.zeros((1)))
    acc_win = vis.line(X=torch.zeros((1)), Y=torch.zeros((1)))

    vis.line(Y=np.column_stack((train_loss_history, val_loss_history)),
             X=np.column_stack((range(len(train_loss_history)),
                                range(len(val_loss_history)))),
             win=loss_win, opts=dict(xlabel='epochs', ylabel='loss', title='train and valid losses',
                                     legend=['train', 'valid'], showlegend=True))
    vis.line(Y=np.column_stack((train_acc_history, val_acc_history)), X=np.column_stack((range(len(train_acc_history)),
                                                                                         range(len(val_acc_history)))),
             win=acc_win, opts=dict(xlabel='epochs', ylabel='loss', title='train and valid accuracy',
                                    legend=['train', 'valid'], showlegend=True))

    torch.save(model.state_dict(), 'cnn_net.pth')
    new_model = CNN.MyConv()
    new_model.load_state_dict(torch.load('cnn_net.pth'))

    # set class to label map
    class_label_map = {0: "zero", 1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven",
                       8: "Eight", 9: "Nine"}

    def predict(loader_dict, model, key, phase='valid'):
        image, labels = next(iter(loader_dict[phase]))
        img, lbl = image[key], labels[key]

        lbl = lbl.item()

        plt.imshow(transforms.ToPILImage()(img), interpolation="bicubic")
        plt.title(class_label_map[lbl])

        model.eval()
        model.cpu()

        out = model.forward(img.unsqueeze_(0))
        ps = torch.exp(out)

        ps, pred = ps.max(dim=1)
        pred = pred.item()
        predicted_label = class_label_map[pred]

        return predicted_label

    pred_label = predict(loader_dict=dataloaders_dict, model=new_model, key=3)  # phase = 'valid' by default
    print(f'Model Prediction: {pred_label}')
    plt.show()


if __name__ == "__main__":
    main()
