import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from net import MultiLayerNet
from dataset import XiaoDataset


# 定义初始化方式     
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)


# 打印参数信息
def print_weight(m):
    if isinstance(m, nn.Linear):
        print("weight", m.weight)
        print("bias:", m.bias)


def train(model, train_loader, criterion, optimizer, epoch):
    for i, samples in enumerate(train_loader):
        x = samples['input']
        y = samples['output']
        #print(i, x.size(), y.size())
    
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)

        # Compute and print loss
        loss = criterion(y_pred, y)

        if i == len(train_loader) - 1:
            print('epoch: {} | iter: {}/{} | loss: {}'.format(epoch, i, len(train_loader) - 1, loss.item()))

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validate(model, val_loader, criterion):
    preds = None
    true_values = None
    for i, samples in enumerate(val_loader):
        x = samples['input']
        y = samples['output']
        #print(i, x.size(), y.size())

        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)

        # Compute and print loss
        loss = criterion(y_pred, y)

        if i == 0:
            preds = y_pred
            true_values = y
        else:
            preds = torch.cat((preds, y_pred), 0)
            true_values = torch.cat((true_values, y), 0)
            if i == len(val_loader) - 1:
                print('validating, iter: {}/{} | loss: {}'.format(i, len(val_loader) - 1, loss.item()))
    
    errors = preds - true_values

    
def main():
    # N is batch size; L is number of layers; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, L, D_in, H, D_out = 1024, 20, 3, 10, 7

    # Construct our model by instantiating the class defined above
    model = MultiLayerNet(L, D_in, H, D_out)

    # initialize
    model.apply(weight_init)
    print(model)
    model.apply(print_weight)

    train_set = XiaoDataset(txt_file='data_train.txt')
    val_set = XiaoDataset(txt_file='data_val.txt')
    train_loader = DataLoader(train_set, batch_size=N, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=N, shuffle=True, num_workers=4)

    # Construct our loss function and an Optimizer. The call to model.parameters()
    # in the SGD constructor will contain the learnable parameters of the two
    # nn.Linear modules which are members of the model.

    criterion = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    for epoch in range(0, 10):
        model.train()
        train(model, train_loader, criterion, optimizer, epoch)
        model.eval()
        validate(model, val_loader, criterion)
        

if __name__ == '__main__':
    main()
