import time

import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm

# 超参数
batch_size = 64
learning_rate = 0.005
# momentum = 0.5
EPOCH = 10

# 数据归一化
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5], std=[0.5])])

train_data = datasets.MNIST(root="./data/mnist",
                            transform=transform,
                            train=True,
                            download=True)

test_data = datasets.MNIST(root="./data/mnist",
                           transform=transform,
                           train=False,
                           download=True)

train_loader = DataLoader(train_data, batch_size=batch_size,
                          shuffle=True, num_workers=8)
test_loader = DataLoader(test_data, batch_size=batch_size,
                         shuffle=True, num_workers=8)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=5),  # 卷积
            torch.nn.ReLU(),  # 激活
            torch.nn.MaxPool2d(kernel_size=2),  # 池化
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(64 * 4 * 4, 512),
            torch.nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv1(x)  # 一层卷积层,一层池化层,一层激活层(图是先卷积后激活再池化，差别不大)
        x = self.conv2(x)  # 再来一次
        x = x.view(-1, 64 * 4 * 4)
        x = self.fc(x)
        return x  # 最后输出的是维度为10的，也就是（对应数学符号的0~9）


model = Net()

criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


def train(epoch):
    tqdm.write('iterator %d:' % (epoch + 1))
    time.sleep(0.01)
    for batch_idx, data in enumerate(tqdm(train_loader, 0)):
        inputs, target = data

        # forward + backward + update
        outputs = model(inputs)

        loss = criterion(outputs, target)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()


def test():
    correct = 0
    total = 0
    tqdm.write('Calculating the accuracy of iteration %d:' % (epoch + 1))
    time.sleep(0.01)
    with torch.no_grad():
        for data in tqdm(test_loader):
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)  # 选取相似度最大的数值作为预测数值
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = correct / total
    tqdm.write('Accuracy of iteration %d: %.2f %% \n' % (epoch + 1, 100 * acc))
    time.sleep(0.01)
    return acc


if __name__ == '__main__':
    acc_list_test = []
    for epoch in range(EPOCH):
        train(epoch)
        acc_test = test()
        acc_list_test.append(acc_test)

    plt.plot(acc_list_test)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy On TestSet')
    plt.show()
