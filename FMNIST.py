import torch
import visdom
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import Dataset
from torch.optim import lr_scheduler
from torch import optim
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
import CNN
import MNIST
torch.set_printoptions(precision=10)

# get train and test dataset
trainset = datasets.FashionMNIST('FMNIST_data', download=True, train=True,
                                 transform=MNIST.data_transforms['train'])
validset = datasets.FashionMNIST('FMNIST_data', download=True, train=False, transform=MNIST.data_transforms['valid'])

# datasets, dataloaders and sizes of each set
datasets_dict = {'train': trainset, 'valid': validset}
dataloaders_dict = {x: torch.utils.data.DataLoader(datasets_dict[x], batch_size=100, shuffle=True) for x in
                    ['train', 'valid']}
dataset_size = {x: len(dataloaders_dict[x].dataset) for x in ['train', 'valid']}


def main():
    # instantiate model and set model training bells and whistles
    num_epochs = 7
    net = CNN.MyConv()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # call train function
    model_ft, train_loss_history, val_loss_history, train_acc_history, val_acc_history = \
        MNIST.train(model=net, dataloaders=dataloaders_dict, optimizer=optimizer, criterion=criterion,
                    num_epochs=num_epochs, is_inception=False, device='cpu')

    # instantiate visdom object
    vis = visdom.Visdom(env='Fashion')

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

    torch.save(net.state_dict(), 'cnn_net.pth')
    new_model = CNN.MyConv()
    new_model.load_state_dict(torch.load('cnn_net.pth'))

    class_label_map = {0: "T-shirt", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat", 5: "Sandal", 6: "Shirt",
                       7: "Sneaker", 8: "Bag", 9: "Ankle Boot"}

    def predict(loader_dict, model, key, phase='valid'):
        image, labels = next(iter(loader_dict[phase]))
        img, lbl = image[key], labels[key]

        lbl = lbl.item()

        plt.imshow(transforms.ToPILImage()(img), interpolation="bicubic")
        plt.title(class_label_map[lbl])

        model.eval()
        model.cpu()

        out = net.forward(img.unsqueeze_(0))
        ps = torch.exp(out)

        ps, pred = ps.max(dim=1)
        pred = pred.item()
        predicted_label = class_label_map[pred]

        return predicted_label

    pred_label = predict(loader_dict=dataloaders_dict, model=net, key=41)
    plt.show()
    print(f'Model Prediction: {pred_label}')


if __name__ == "__main__":
    main()
























