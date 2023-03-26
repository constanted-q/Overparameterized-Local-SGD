from resnet import ResNet18,ResNet50
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader
from GoogleNet import GoogLeNet
from train import gra_norm
import numpy as np




transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = datasets.CIFAR10(root='./data/cifar10', train=True,
                                download=False, transform=transform)

test_dataset = datasets.CIFAR10(root='./data/cifar10', train=False,
                               download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=100, shuffle=True)
evalloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=1000, shuffle=True)
testloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=100, shuffle=False)

if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")
global_model = ResNet18()
#global_model = GoogLeNet()
global_model = global_model.to(device)
print(sum(p.numel() for p in global_model.parameters() if p.requires_grad))
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(global_model.parameters(), lr=0.01)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    global_model.train()
    train_loss = 0
    correct = 0
    total = 0
    i = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = global_model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if i % 20 == 0:
            #print(f'I = : [{i:5d}] Loss: {train_loss / (i+1):.4f}')
            inputs, labels = next(iter(evalloader))
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            # with torch.no_grad():
            outputs = global_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            gradient_norm = gra_norm(global_model)
            print(f'I = : [{i:5d}] GradNorm: {gradient_norm:.3f}')
            print(f'I = : [{i:5d}] Loss: {loss:.4f}')
        i+=1
    #print(f'Epoch: [{epoch + 1:5d}] Loss: {train_loss/len(trainloader):.3f}')
    #print(f'Train Accuracy: {100 * correct / total} %')


def test(epoch):
    global best_acc
    global_model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = global_model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()


    # Save checkpoint.
    acc = 100.*correct/total
    print(f'Test Accuracy: {acc} %')


for epoch in range(200):
    train(epoch)
    test(epoch)