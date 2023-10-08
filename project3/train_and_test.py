from load_data import Cifar10, unpickle, load_batch_data
from models import CifarNet
import torch.nn as nn
import torch
import torch.utils.data as data
import time
import numpy as np
from sklearn.metrics import accuracy_score


def train_and_val(train_loader, val_loader, lr, model, epochs, criterion, device):
    best_loss_test = float("inf")

    for epoch in range(epochs):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # learning rate, 0.001
        batch = 0
        "start training"
        for data, label in train_loader:
            data = data.to(device=device, dtype=torch.float32)
            batch += 1
            model.train()  # train mode
            pred = model(data)  # forward propagation
            optimizer.zero_grad()  # initialize optimizer
            loss = criterion(pred, label.to(device=device, dtype=torch.long))
            loss.backward()  # backward propagation
            optimizer.step()  # update model parameters
            if batch % 10 == 0:
                print("\rTrain Epoch: {:d} | Train loss: {:.4f} | Batch : {}/{}".format(epoch + 1, loss, batch, len(train_loader)))

            "start testing"
            if batch % 30 == 0:
                loss_sum = 0
                pred_list, true_list = [], []
                for data, label in val_loader:
                    model.eval()  # evaluating mode
                    with torch.no_grad():  # no gradient
                        data = data.to(device=device, dtype=torch.float32)
                        pred = model(data)
                        loss_test = criterion(pred, label.to(device=device, dtype=torch.long))
                        loss_sum += loss_test
                        pred = pred.cpu().detach().numpy()
                        pred = np.argmax(pred, axis=1)
                        pred_list += pred.tolist()
                        true_list += label.numpy().tolist()
                loss_avg = loss_sum / len(val_loader)
                acc = accuracy_score(true_list, pred_list)
                if loss_avg <= best_loss_test:
                    best_loss_test = loss_avg
                    torch.save(model.state_dict(), './saved_model/cifar_net_4_1_1.pth')
                print("\rTest Epoch: {:d} | Test loss: {:.4f} | Test Accuracy: {:.4%} | Best evaluation loss: {:.6f}".format(epoch + 1, loss_avg, acc, best_loss_test))
                time.sleep(0.1)


if __name__ == "__main__":
    # freeze resnet parameter or not
    # feature_extract = True
    # model = models.resnet18(pretrained=True)
    model = CifarNet()

    # set_parameter_requires_grad(model, feature_extract)
    all_data = []
    all_label = []
    for i in range(5):
        train_file = './cifar-10-batches-py/data_batch_' + str(i + 1)
        dict_data = unpickle(file=train_file)
        batch_data = load_batch_data(dict_data[b'data'])
        all_data.append(batch_data)
        all_label.append(dict_data[b'labels'])
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    train_set = Cifar10(data_array=all_data, label_array=all_label)
    train_loader = data.DataLoader(dataset=train_set, batch_size=64, shuffle=True)

    test_file = './cifar-10-batches-py/test_batch'
    dict_data = unpickle(file=test_file)
    batch_data = load_batch_data(dict_data[b'data'])
    test_set = Cifar10(data_array=batch_data, label_array=np.array(dict_data[b'labels']))
    val_loader = data.DataLoader(dataset=test_set, batch_size=64, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # train_and_val(train_loader, val_loader, 1e-3, model, 30, criterion, device)

    model_param = torch.load('./saved_model/cifar_net_origin.pth')
    model.load_state_dict(model_param)
    pred_list, true_list = [], []
    for data, label in val_loader:
        model.eval()  # evaluating mode
        with torch.no_grad():  # no gradient
            data = data.to(device=device, dtype=torch.float32)
            pred = model(data)
            loss_test = criterion(pred, label.to(device=device, dtype=torch.long))
            pred = pred.cpu().detach().numpy()
            pred = np.argmax(pred, axis=1)
            pred_list += pred.tolist()
            true_list += label.numpy().tolist()
    acc = accuracy_score(true_list, pred_list)
    print("\rTest Accuracy: {:.4%}".format(acc))









