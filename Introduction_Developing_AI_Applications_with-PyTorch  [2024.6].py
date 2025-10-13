# https://github.com/pytorch/pytorch

# define by run 방식

# autograd(자동미분), optim, nn 모듈

# 세션(접속을 확리하고 나서 끊을 때까지 일련의 통신)과 인스턴스(소프트웨어로 구현된 가상적인 컴퓨터 또는 머신)
# 90분 규칙 : 세션이 끊기고 90분 뒤면 인스턴스 소멸
# 120분 규칙 : 새로운 인스턴스 기동 후 12시간 경과 후 인스턴스 소멸

# CPU, GPU, TPU
# CPU보다 GPU 사용이 빠름. Google Collab 에서 '수정 - 노트설정' 통해서 활용 가능

# 학습 파라미터 (가중치, 바이어스)
# 입력된 데이터에 가중치 곱하여 총합 취합 후 바이어스 더해서 활성화 함수에 의해 처리 후 출력

# 하이퍼 파라미터

# 순전파, forward
# 예측치
# 역전파, backward

!pip list

import torch

a = torch.tensor([1,2,3])
print(a, type(a))

b = torch.tensor([[1,2],[3,4]])
print(b)

c = torch.tensor([[1,2],[3,4]], dtype=torch.float64)
print(c)

d = torch.arange(0,10)
print(d)

e = torch.zeros(2,3)
print(e)

f = torch.rand(2,3)
print(f)

print(f.size())

g = torch.linspace(-5,5,10)
print(g)

a = torch.tensor([[1,2],[3,4.]])
b = a.numpy()
print(a, b)

c = torch.from_numpy(b)
print(c)

a = torch.tensor([[1,2,3],[4,5,6]])
print(a[0,1])

print(a[1:2, :2])

print(a[:, [0,2]])

print(a[a>3])

a[0,2] = 11
print(a)

a[:, 1] = 22
print(a)

a[a>10] = 33
print(a)

a = torch.tensor([1,2,3])
b = torch.tensor([4,5,6])

c = torch.tensor([[6,5,4], [3,2,1]])

print(a+3)

print(a+b)

print(c+2)

print(c+a)

print(c+c)

a = torch.tensor([0,1,2,3,4,5,6,7])

print(a)

b = a.view(2,4)

print(b)

c = a.view(2,-1)

print(c)

d = torch.tensor([[[0,1],[2,3]],[[4,5],[6,7]]])

print(d)

e = d.view(-1)

print(e)

f = torch.arange(0,8).view(1,2,1,4)

print(f)

g = f.squeeze()

print(g)

h = torch.arange(0,8).view(2,-1)

print(h)

i = h.unsqueeze(2)

print(i)

a = torch.tensor([[1,2,3],[4,5,6.]])

m = torch.mean(a)

print(m.item())

m = a.mean()

print(m.item())

print(a.mean(0))

print(torch.sum(a).item())

print(torch.max(a).item())

print(torch.min(a).item())

# + - * / // %

a = torch.tensor([[1,2,3],[4,5,6]])
b = torch.tensor([1,2,3])

print(a+b)

print(a-b)

print(a*b)

print(a/b)

print(a//b)

print(a%b)

# 활성화 함수

import torch
from torch import nn
import matplotlib.pylab as plt

# 시그모이드 함수

m = nn.Sigmoid()

x = torch.linspace(-1,1,10)
y = m(x)

plt.plot(x,y)
plt.show()

# 하이퍼블릭 탄젠트 함수

m = nn.Tanh()

x = torch.linspace(-1,1,10)
y = m(x)

plt.plot(x,y)
plt.show()

# 램프 함수

m = nn.ReLU()

x = torch.linspace(-1,1,10)
y = m(x)

plt.plot(x,y)
plt.show()

# 항등 함수

x = torch.linspace(-1,1,10)
y = x

plt.plot(x,y)
plt.show()

# 소프트맥스 함수

m = nn.Softmax(dim=1)

x = torch.tensor([[1.0, 2.0, 3.0],[3.0, 2.0, 1.0])

y = m(x)

print(y)

# 손실 함수

# 평균 제곱 오차

y = torch.tensor([3.0, 3.0, 3.0, 3.0, 3.0])
t = torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0])

loss_func = nn.MSELoss()
loss = loss_func(y,t)
print(loss.item())

# 교차 엔트로피 오차

x = torch.tensor([[1.0, 2.0, 3.0],[3.0, 2.0, 1.0])

t = torch.tensor([2,0])

loss_func = nn.CrossEntropyLoss()
loss = loss_func(y,t)
print(loss.item())

# 최적화 알고리즘

# 발 밑의 경사, 지금까지의 경로, 경과 시간 요소 필수

from torch import optim

# 경사하락법

# optimizer = optim.SGD(.., )

# 모멘텀, 관성학

# optimizer = optim.SGD(.., momentum=0.9)

# AdaGrad, 갱신량 자동 조정

# optimizer = optim.Adagrad(... )

# RMSProp, 경사하강법 + 학습률 조정

# optimizer = optim.RMSprop(... )

# Adam, 다양한 최적화 알고리즘의 장점을 합산

# optimizer = optim.AdamI... )



# 손글씨 문자 이미지

import matplotlib.pyplot as plt
from sklearn import datasets

digits_data = datasets.load_digits()

n_img = 10
plt.figure(figsize=(10,4))
for i in range(n_img):
    ax = plt.subplot(2,5,i+1)
    ax.imshow(digits_data.data[i].reshape(8,8), cmap = "Greys_r")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

print("데이터의 형태:", digits_data.data.shape)
print("라벨:", digits_data.target[:n_img])

import torch
from sklearn.model_selection import train_test_split

digit_images = digits_data.data
labels = digits_data.target
x_train, x_test, t_train, t_test = train_test_split(digit_images, labels)

x_train = torch.tensor(x_train, dtype=torch.float32)
t_train = torch.tensor(t_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
t_test = torch.tensor(t_test, dtype=torch.float32)

from torch import nn

net = nn.Sequential(nn.Linear(64,31), nn.ReLU(), nn.Linear(32,16), nn.ReLU(), nn.Linear(16,10))

print(net)

import torch.optim as optim

loss_fnc = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

record_loss_train = []
record_loss_test = []

for i in range(1000):
    optimizer.zero_grad()

    y_train = net(x_train)
    y_test = net(x_test)

    loss_train = loss_fnc(y_train, t_train)
    loss_test = loss_fnc(y_test, t_test)
    record_loss_train.append(loss_train.item())
    record_loss_test.append(loss_test.item())

    loss_train.backward()

    optimizer.step()

    if i % 100 == 0:
        print("Epoch:", i, "Loss_Train:", loss_train.item(), "Loss_Test:", loss_test.item())

plt.plot(range(len(record_loss_train)), record_loss_train, label ='Train')
plt.plot(range(len(record_loss_test)), record_loss_train, label ='Test')
plt.legend()

plt.xlabel("Epochs")
plt.ylabel("Error")
plt.show()

y_test = net(x_test)
count = (y_test.argmax(1) == t.test).sum().item()
print("정답률:", str(count/len(y_test)*100) + '%')

img_id = 0
x_pred = digit_images[img_id]
image = x_pred.reshape(8,8)
plt.imshow(image, cmap="Greys_r")
plt.show()

x_pred = torch.tensor(x_pred, dtype=torch.float32)
y_pred = net(x_pred)



x = torch.ones(2,3,requires_grad=True)
print(x)

y = x+2
print(y, y.grad_fn)

z = y*3
print(z)

out = z.mean()
print(out)

a = torch.tensor([1.0], requires_grad=True)
b = a*2
b.backward()
print(a.grad)

def calc(a):
    b = a*2 + 1
    c = b*b
    d = c/(c+2)
    e = d.mean()
    return e

x = [1.0, 2.0, 3.0]
x = torch.tensor(x, requires_grad=True)
y = calc(x)
y.backward()
print(x.grad)

delta = 0.001

x = [1.0, 2.0, 3.0]
x = torch.tensor(x)
y = calc(x)

x_1 = [1.0+delta, 2.0, 3.0]
x_1 = torch.tensor(x_1)
y_1 = calc(x_1)

x_2 = [1.0, 2.0+delta, 3.0]
x_2 = torch.tensor(x_2)
y_2 = calc(x_2)

x_3 = [1.0, 2.0, 3.0+delta]
x_3 = torch.tensor(x_3)
y_3 = calc(x_3)

grad_1 = (y_1-y)/delta
grad_2 = (y_2-y)/delta
grad_3 = (y_3-y)/delta

grads = torch.stack((grad_1, grad_2, grad_3))
print(grads)

# 훈련데이터를 1회 다 써서 학습하는 것은 1 에포크(epoch)단위로 표현
# 여러개를 그룹으로 합쳐서 한 번의 학습에 사용될때 이 그룹을 배치(batch)라고 지칭

# dataloader는 gpu 사용을 추천
# https://pytorch.org/vision/stable/datasets.html

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

# from torchvision.datasets import MNIST
# from torchvision import transform

img_size = 28

mnist_train = MNIST("./data", train = True, download=True, transform=transforms.ToTensor())
mnist_test = MNIST("./data", train = False, download=True, transform=transforms.ToTensor())

print('훈련데이터수:', len(mnist_train), '테스트 데이터 수:', len(mnist_test))

from torch.utils.data import DataLoader

batch_size = 256
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 == nn.Linear(img_size*img_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        seld.fc3 = nn.Linear(512, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, img_size*img_size)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(fc3)
        return x

    net = Net()
    net.cuba() # GPU 대응
    print(net)

loss_fnc = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=0.01)

record_loss_train = []
record_loss_test = []

for i in range(10):
    net.train()
    loss_train = 0
    for j in (x,t) in enumerate(train_loader):
#        x, t = x.cuda(), t.cuda()
        y = net(x)
        loss = loss_fnc(y, t)
        loss_train += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss_train /= j+1
    record_loss_train.append(loss_train)

    net.eval()
    loss_test = 0
    for j, (x,t) in enumerate(test_loader):
#        x, t = x.cuda(), t.cuda()
        y = net(x)
        loss = loss_fnc(y, t)
        loss_test += loss.item()
    loss_test /= j+1
    record_loss_test.append(loss_test)

    if i%1 == 0:
        print(i, loss_train, loss_test)

import matplotlib.pyplot as plt

plt.plot(range(len(record_loss_train)), record_loss_train, label = 'Train')
plt.plot(range(len(record_loss_test)), record_loss_test, label = 'Test')
plt.legend()

plt.xlabel('x')
plt.ylabel('y')
plt.show()

correct = 0
total = 0
net.eval()
for i, (x,t) in enumerate(test_loader):
#    x, t = x.cuda(), t.cuda()
    y = net(x)
    correct += (y.argmax(1) == t).sum().item()
    total += len(x)
print(str(correct/total *100) + '%')



# CNN

# 합성곱층, 국소성 성질을 활용하여 필터(커널)를 통해 특징의 검출
# 컬러이미지는 RGB 즉, 3가지 채널을 갖고 있다 표현
# self.conv1 = nn.Conv2d(3, 6, 5)

# 풀링층, 각 영역의 대표값을 꺼내어 나열하며 새로운 이미지 생성하는 행위
# MAX 풀링, 평균 풀링
# self.pool = nn.MaxPool2d(2,2)

# 패딩, 입력 이미지를 둘러싸듯이 픽셀을 배치하는 테크닉
# 제로패딩, 주변이 0 인 경우
# 스트라이드, 합성곱에서 필터가 이동하는 간격



# 범화성능, 미지의 데이터에 대한 대응력
# 회전, 확대, 축소, 상하좌우 이동, 반전, 일부 소거 등을 통해 이미지를 부풀려 데이터 확장 시킴

from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

cifar10_data = CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
print("데이터의 수:", len(cifar10_data))

n_image = 25
cifar10_loader = DataLoader(cifar10_data, batch_size=n_image, shuffle=True)
dataiter = iter(cifar10_loader)
images, labels = next(dataiter)

plt.figure(figsize=(10,10))

for i in range(n_image):
    ax = plt.subplot(5, 5, 1+i)
    ax.imshow(images[i].permute(1,2,0))
    label = cifar10_classes[labels[i]]
    ax.set_title(label)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

transform = transforms.Compose([transforms.RandomAffine((-45, 45), scale=(0.5, 1.5)), transforms.ToTensor()]) # 회전, 크기 조절
cifar10_data = CIFAR10(root="./data", train=False, download=True, transform=transform)

cifar10_loader = DataLoader(cifar10_data, batch_size=n_image, shuffle=True)
dataiter = iter(cifar10_loader)
images, labels = next(dataiter)

plt.figure(figsize=(10,10))

for i in range(n_image):
    ax = plt.subplot(5, 5, 1+i)
    ax.imshow(images[i].permute(1,2,0))
    label = cifar10_classes[labels[i]]
    ax.set_title(label)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

transform = transforms.Compose([transforms.RandomAffine((0, 0), scale=(0.5, 0.5)), transforms.ToTensor()]) # 상하좌우 변경
cifar10_data = CIFAR10(root="./data", train=False, download=True, transform=transform)

cifar10_loader = DataLoader(cifar10_data, batch_size=n_image, shuffle=True)
dataiter = iter(cifar10_loader)
images, labels = next(dataiter)

plt.figure(figsize=(10,10))

for i in range(n_image):
    ax = plt.subplot(5, 5, 1+i)
    ax.imshow(images[i].permute(1,2,0))
    label = cifar10_classes[labels[i]]
    ax.set_title(label)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip((p=0.5), transforms.ToTensor()]) # 상하좌우 반전
cifar10_data = CIFAR10(root="./data", train=False, download=True, transform=transform)

cifar10_loader = DataLoader(cifar10_data, batch_size=n_image, shuffle=True)
dataiter = iter(cifar10_loader)
images, labels = next(dataiter)

plt.figure(figsize=(10,10))

for i in range(n_image):
    ax = plt.subplot(5, 5, 1+i)
    ax.imshow(images[i].permute(1,2,0))
    label = cifar10_classes[labels[i]]
    ax.set_title(label)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

transform = transforms.Compose([transforms.ToTensor(), transforms.RandomErasing(p=0.5)]) # 일부 소거
cifar10_data = CIFAR10(root="./data", train=False, download=True, transform=transform)

cifar10_loader = DataLoader(cifar10_data, batch_size=n_image, shuffle=True)
dataiter = iter(cifar10_loader)
images, labels = next(dataiter)

plt.figure(figsize=(10,10))

for i in range(n_image):
    ax = plt.subplot(5, 5, 1+i)
    ax.imshow(images[i].permute(1,2,0))
    label = cifar10_classes[labels[i]]
    ax.set_title(label)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()



# 드롭 아웃, 출력층 이외의 뉴런을 일정 확률로 무작위 소거하는 테크닉

# self.dropout = nn.Dropout(p=0.5)



from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

affine = transforms.RandomAffine((-30, 30), scale=(0.8, 1.2)) # 회전, 크기 조절
flip = transforms.RandomHorizontalFlip(p=0.5)

normalize = transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))

to_tensor = transforms.ToTensor()

transform_train = transforms.Compose([affine, flip, to_tensor, normalize])
transform_test = transforms.Compose([to_tensor, normalize])
cifar10_train = CIFAR10("./data", train=True, download=True, transform = transform_train)
cifar10_test = CIFAR10("./data", train=False, download=True, transform = transform_test)

batch_size = 64
train_loader = DataLoader(cifar10_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(cifar10_test, batch_size=batch_size, shuffle=False)

import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 5)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 16*5*5)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

net = Net()
# net.cuda()
print(net)

from torch import optim

loss_fnc = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters())

record_loss_train = []
record_loss_test = []

for i in range(20):
    net.train() # 훈련모드
    loss_train = 0
    for j, (x,t) in enumerate(train_loader):
#        x, t = x.cuda(), t.tuda()
        y = net(x)
        loss = loss_fnc(y, t)
        loss_train += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss_train /= j+1
    record_loss_train.append(loss_train)

    net.eval() # 평가모드
    loss_test = 0
    for j, (x,t) in enumerate(test_loader):
#        x, t = x.cuda(), t.tuda()
        y = net(x)
        loss = loss_fnc(y, t)
        loss_test += loss.item()
    loss_train /= j+1
    record_loss_train.append(loss_train)

    if i%1 == 0:
        print( i, loss_train, loss_test )

import matplotlib.pyplot as plt

plt.plot(range(len(record_loss_train)), record_loss_train, label="Train")
plt.plot(range(len(record_loss_test)), record_loss_test, label="Test")
plt.legend()

plt.xlabel("x")
plt.xlabel("y")
plt.show()

correct = 0
total = 0
net.eval()
for i, (x,t) in enumerate(test_loader):
#    x, t = x.cuda(), t.cuda()
    y = net()
    correct += (y.argmax(1) == t)>sum().item()
    total += len(x)
print(str(correct/total*100) + "%")

cifar10_loader = DataLoader(cifar10_test, batch_size=1, shuffle=True)
dataiter = iter(cifar10_loader)
images, labels = next(dataiter)

plt.imshow(images[0].permute(1,2,0))
plt.tick_params(labelbotton=False, labelleft=False, bottom=False, left=False)
plt.show()

net.eval()
# x, t = images.cuda(), label.cuda()
y = net()
print(cifar10_classes[labels[0]], cifar10_classes[y.argmax().item()]



# RNN(Recurrent Neural Network, 순환신경망)
# 입력층 - 중간층 - 출력층, 중간층의 루프
# 경사 폭발 혹은 경사 소설 문제 발생 ~ LSTM 개요의 기억셀을 통해 대처 가능

# self.rnn = nn.RNN( # RNN층
#   input_size=1 # 입력수
#   hide_size=64 # 뉴런수
#   batch_first = True,) # 입력형태를 (배치 크기, 시각 수, 입력 수)로 한다
# ...
#  y_rnn, h = self.rnn(x, None) # 모든 시각의 출력, 중간층의 최종 시각의 값

import torch
import math
import matplotlib.pyplot as plt

# 노이즈가 있는 사인 곡선을 시계열을 활용하여 학습

sin_x = torch.linspace(-2*math.pi, 2*math.pi, 100)
sin_y = torch.sin(sin_x) + 0.1*torch.rand(len(sin_x))
plt.plot(sin_x,sin_y)
plt.show()

from torch.utils.data import TensorDataset, DataLoader

# 데이터 전처리

n_time = 10
n_sample = len(sin_x)-n_time

input_data = torch.zeros((n_sample, n)time, 1))
correct_data = torch.zeros((n_sample, 1))

for i in ragne(n_sample):
    input_data[i] = sin_y[i:i+n_time].view(-1,1)
    correct_data[i] = sin_y[i+n_time:i+n_time+1]

dataset = TensorDataset(input_data, correct_data)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=64, batch_first=True,)
        self.fc = nn.Linear(64,1)

    def forward(self,x):
        y_rnn, h = self.rnn(x, None)
        y = self.fc(y_rnn[:, -1, :])

        return y

net =  Net()
print(net)

from torch import optim

loss_fnc = nn.MSELoss()

optimizer = optim.SGD(net.parameters(), lr=0.01)

record_loss_train = []

epochs = 100

for i in range(epochs):
    net.train()
    loss_train = 0
    for j, (x,t) in enumerate(train_loader):
        y = net(x)
        loss = loss_fnc(y, t)
        loss_train += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss_train /= j+1
    record_loss_train.append(loss_train)

    # 경과 표시
    if i%10==0 or i==epochs-1:
        net.eval() # 평가모드
        print( i, loss_train)
        predicted = list(input_data[0].view(-1))
        for i in range(n_sample):
            x = torch.tensor(predicted[-n_time:]) # 가장 최근의 시계열

        x = x.view(1, n_time, 1)
        y = net(x)
        predicted.append(y[0].item())

    plt.plot(range(len(sin_y)), sin_y, label = "C")
    plt.plot(range(len(predicted)), predicted, label = "P")
    plt.legend()
    ple.show()

plt.plot(range(len(record_loss_train)), record_loss_train, label = "T")
plt.legend()

plt.xlabe("X")
plt.ylabel("Y")
plt.show()

# LSTM (Long Short Term Memory)
# 입력층 - 중간층 - 출력층, RNN과 유사하게 중간층에서 재귀
# 다만 내부에 게이트라는 구조 도입으로 과거 정보 기억 여부 판단
# 출력게이트  - 망각게이트 - 입력게이트 - 기억셀

# self.rnn = nn.LSTM( # LSTM층
#   input_size=n_in # 입력수
#   hide_size=n_mid # 뉴런수
#   batch_first = True,) # 입력형태를 (배치 크기, 시각 수, 입력 수)로 한다
# ...
# y_rnn, (h, c) = self.rnn(x, None) # 모든 시각의 출력, 중간층의 최종 시각 값, 기억 셀

# GRU (Gated Recurrent Unit)
# LSTM을 개량한 구조로 단순하고 파라미터 수가 적으며 계산량이 억제됨
# (입력게이트 + 망각게이트)= 갱신게이트 - 리셋게이트

# self.rnn = nn.GRU( # GRU층
#   input_size=n_in # 입력수
#   hide_size=n_mid # 뉴런수
#   batch_first = True,) # 입력형태를 (배치 크기, 시각 수, 입력 수)로 한다
# ...
# y_rnn, h = self.rnn(x, None) # 모든 시각의 출력, 중간층의 최후 시각 값

# 기억셀을 고려할 필요 없음



from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# https://github.com/zalandoresearch/fashion-mnist

fmnist_data = FashionMNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
fmnist_classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Sneaker", "Bag", "Ankle boot"]
print(len(fmnist_data))

n_image = 25
fmnist_loader = DataLoader(fmnist_data, batch_size=n_image, shuffle=True)
dataiter = iter(fmnist_loader)
images, labels = next(dataiter)

img_size = 28
plt.figure(figsize=(10,10))
for i in range(n_image):
    ax = plt.subplot(5,5,i+1)
    ax.imshow(images[i].view(img_size, img_size), cmap="Greys_r")
    label = fmnist_classes[labels[i]]
    ax.set_title(label)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

img_size = 28

n_time = 14
n_in = img_size
n_mid = 256
n_out = img_size
n_sample_in_img = img_size-n_time

dataloader = DataLoader(fmnist_data, batch_size=len(fmnist_data), shuffle=False)
dataiter = iter(dataloader)
train_imgs, labels = next(dataiter)
train_imgs = train_imgs.view(-1, img_size, img_size)

n_sample = len(train_imgs)*n_sample_in_img

input_data = torch.zeros((n_sample, n_time, n_in))
correct_data = torch.zeros((n_sample, n_out))
for i in range(len(train_imgs)):
    for j in rnage(n_sample_in_img):
        sample_id = i*n_sample_in_img+j
        input_data[sample_id] = train_imgs[i, j:j+n_time]
        correct_data[sample_id] = train_imgs[i, j+n_time]

dataset = TensorDataset(input_data, correct_data)
train_loader = DataLoader(dataset, batch_size=128, shuffle=True)

n_disp = 10

disp_data = FashionMNIST(root="./data", train=False, download=True, transform = transforms.ToTensor())
disp_loader = DataLoader(disp_data, batch_size=n_disp, shuffle=False)
ditaiter = iter(disp_loader)
disp_imgs, labels = next(dataiter)
disp_imgs = disp_imgs.view(-1, img_size, img_size)

import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(input_size=n_in, hidden_size=n_mid,batch_first = True, )
        self.fc = nn.Linear(n_mid, n_out)

    def forward(self, x):
        y_rnn, (h, c) = self.rnn(x, None)
        y = self.fc(y_rnn[:, -1, :])
        return y

net = Net()
# net.cuda()
print(net)

import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM( input_size=n_in, hidden_size=n_mid, batch_first=True, )
        self.fc = nn.Linear(n_mid, n_out)

    def forward(self, x):
        y_rnn (h, c) = self.rnn(x, None)
        y = self.fc(y_rnn[:, -1, :])
        return y

net = Net()
# net.cuda()
print(net)

def generate_image():
    print("Original")
    plt.figuare(figsize=(20,2))
    for i in range(n_disp):
        ax = plt.subplot(1, n_disp, i+1)
        aax.imshow(disp_imgs[i], cmap="Greys_r", vmin=0.0, v_max=1.0)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

    print("Generated")
    net.eval()
    gen_imgs = disp_imgs.clone()
    plt.figuare(figsize=(20,20)
    for i in range(n_disp):
        for j in range(n_sample_in_img):
            x = gen_imgs[i, j:j+n_time].view(1, n_time, img_size)
#            x = x.cuda()
            gen_imgs[i, j+n_time] = net(x)[0]
        ax = plt.subplot(1, n_disp, i+1)
        ax.imshow(gen_imgs[i].detach(), cmap="Greys_r", vmin=0.0, vmax=1.0)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

from torch import optim

loss_fnc = nn.MSELoss()

optimizer = optim.Adam(net.parameters())

record_loss_train = []

epochs = 30
for i in range(epochs):
    net.train)_
    loss_train = 0
    for j, (x, t) in enumerate(train_loader):
#        x, t = x.cuda(), t.cuda()
        y = net(x)
        loss = loss_fnc(y,t)
        loss_train += loss.item()
        optimizer.zero_grad()
        loss.backwward()
        optimizer.step()
    loss_train /= j+1
    record_loss_train.apppend(loss_train)

    if i%5==0 or i==epochs-1:
        print(i, loss_train)
        generate_images()

plt.plot(range(len(record_loss_train)), record_loss_train, label="T")
plt.legend()

plt.xlabel("Ec")
plt.ylabel("Er")
plt.show()



# 모델 저장 후 로컬 환경으로 다운로드

# import torch

# for key in net.state_dict():
#     print(key, ":", net.state_dict()[key].size())

# torch.save(net.state_dict(), "prac_cnn.pth")  ## prac_cnn.pth

# 앱의 구축

# 'ngrok, https://ngrok.com/' 회원 가입 후 'Your Authtoken' 복사하여 Authtoken 설정에서 붙여넣고 저장


# !pip install streamlit==1.8.1 --quiet
# !pip install pyngrok==4.1.1 --quiet

## 굳이 Jupyter Notebook 에서 하고 싶다면, 다만 pyngrok은 안됨
## pip install pygwalker
## pip install streamlit


# import streamlit as st
# from pyngrok import ngrok


# 훈련한 파라미터, name_cnn.pth 업로드

# Commented out IPython magic to ensure Python compatibility.
# %%writefile model.py
# #이미지 인식을 훈련한 모델을 읽어 들이고 예측하는 코드
# 
# class Net(nn.Module):
#     ...
# # 이미지 인식의 모델
# net.load_state_dict(torch.load"prac_cnn.pth", map_location=torch.device("cpu")))
# #훈련한 파라미터 읽어 들이기와 설정. 이떄, cpu 지정으로 gpu 훈련 모델을 cpu에서도 사용 가능
# def predict(img):
# # 예측 모델
# net.eval()
# y = net()
# # 예측
# return ...
# # 결과 반환

# Commented out IPython magic to ensure Python compatibility.
# %%writefile app.py
# #이미지 인식을 훈련한 모델을 읽어 들이고 예측하는 코드
# 
# # https://docs.streamlit.io/library/api-reference
# ## st.sidebar
# ## st.title("")  ##타이틀 표시
# ## st.write("")  ##다양한 타입의 인수 화면 표시
# ## st.caption("") ##작은 글씨체로 텍스트를 표시
# ## st.radio("")  ##이미지를 업로드, 카메라로 촬영 2가지의 선택지를 주는 버튼
# ## st.file_uploader("")  ##파일을 업로드 가능한 영역이 배치
# ## st.camera_input("")  ##웹카메라 실행
# ## st.image("")  ##화면에 이미지 표시
# ## st.pyplot("")  ##matplotlib 그래프 표시

# !ngrok authtoken YourAuthtoken 중 'YourAuthtoken' 부분을 처음에 복사한 '나의 YourAuthtoken'로 설정
## 이때 깃허브에 나의 YourAuthtoken를 업로드 하지 않도록 주의

# !streamlit run app.py &>/dev/null& ## '&>/dev/null&' 출력 표시하지 않고 백그라운드 실행

# ngrok.kill() ##프로세스 종료
# url = ngrok.connect(port="8501") ##8501은 접속 의미

# print(url)
## 에서 나온 결과 복사하여 http를 https로 변경하여 페이지 표시하고 페이지 뜨는지 및 결과 확인

# 각 라이브러리의 버전 확인하여 'requirements.txt'에 저장

# with open("requirements.txt", "w") as w:
#     w.write("streamlit== \n")
#     w.write("torch== \n")
#     w.write("torchvision== \n")
#     w.write("Pillow\n")
#     w.wrtie("matplotlib\n")

# app.py, model.py, requirements.txt 저장하기

# 깃허브 저장소에 업로드한 파일(app.py, model.py, requirements.txt)을 토대로 '스트림라인, https://streamlit.io/' 에서 앱 제작