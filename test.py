# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import oss2
import re
from io import BytesIO
from tqdm import tqdm # 进度条

from io import BytesIO
from PIL import Image

import torch  # 基础库
import torch.nn as nn  # 神经网络库
import torch.nn.functional as F  # 迭代权重和偏差用
import torch.optim as optim  # 根据梯度更新网络参数时的优化器
from torch.utils.data.dataset import Dataset  # 自定义数据集
from torchvision import datasets, transforms  # 包含一些常用的数据集、模型、转换函数等等
from torch.nn.parallel import DistributedDataParallel as DDP  # 数据并行化，分布式的时候使用

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--inputs", type=str, default='inputspath')
parser.add_argument("--checkpointDir", type=str, default='Checkpoint dir')

args = parser.parse_args()  # 解析参数

ak = "LTAI4G5FSAqJg5Wyham9uTXN"#ak
akSec = "zj0ejgnEyLUutaPe8Zg4Ra1hVqZUZS"#secret

temp = re.search(r"oss://(.*?)\.(.*?)/(.*)", args.inputs)
bucket = temp.group(1)
host = temp.group(2)
auth = oss2.Auth(ak, akSec)
bucket = oss2.Bucket(
    auth,
    host,
    bucket)

class OSSMnistDataset(Dataset):# 从oss加载数据集的方法
  """
  Args:
    oss_prefix (string): a oss directory to mnist dataset. such as oss://bucket/mnist/
    host: oss endpoint
    ak: your oss ak
    akSec: your oss akSec
    train: wether load training data or test data
    transform: pytorch transforms for transforms and tensor conversion
  """
  training_file = 'training.pt'
  test_file = 'test.pt'

  def __init__(self, oss_path, ak, akSec, train=True, transform=None):
    self.train = train  # training set or test set
    if self.train:
      data_file = self.training_file
    else:
      data_file = self.test_file

    self.transform = transform

    o = re.search(r"oss://(.*?)\.(.*?)/(.*)", oss_path)
    # bucket = o.group(1)
    # host = o.group(2)
    path = o.group(3)

    # auth = oss2.Auth(ak, akSec)
    # bucket = oss2.Bucket(
    #     auth,
    #     host,
    #     bucket)
    buffer = BytesIO(bucket.get_object(os.path.join(path, data_file)).read())
    self.data, self.targets = torch.load(buffer)

  def __getitem__(self, index):
    """
        Args:
        index (int): Index
        Returns:
        tuple: (image, target) where target is index of the target class.
    """
    img, target = self.data[index], int(self.targets[index])

    # doing this so that it is consistent with all other datasets
    # to return a PIL Image
    img = Image.fromarray(img.numpy(), mode='L')

    if self.transform is not None:
      img = self.transform(img)

    return img, target

  def __len__(self):
    return len(self.data)

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()  # 两层二维卷积，两层全连接层
    self.conv1 = nn.Conv2d(1, 20, 5, 1)
    self.conv2 = nn.Conv2d(20, 50, 5, 1)
    self.fc1 = nn.Linear(4*4*50, 500)
    self.fc2 = nn.Linear(500, 10)

  def forward(self, x):
    x = F.relu(self.conv1(x)) # 激活函数
    x = F.max_pool2d(x, 2, 2) # 最大池化层
    x = F.relu(self.conv2(x)) # 激活函数
    x = F.max_pool2d(x, 2, 2) # 最大池化层
    x = x.view(-1, 4*4*50) # tensor维度转化
    x = F.relu(self.fc1(x)) # 激活函数
    x = self.fc2(x) 
    return F.log_softmax(x, dim=1) # 归一化后log

def train(model, device, train_loader, optimizer, epoch):
  model.train()
  for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad() # 清空过往梯度
    output = model(data)
    loss = F.nll_loss(output, target) # 损失函数
    loss.backward() # 反向传播
    optimizer.step() # 梯度更新
    if batch_idx % 10 == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad(): # 关闭tensor自动求导，省资源
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      output = model(data)
      test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
      pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
      correct += pred.eq(target.view_as(pred)).sum().item()

  test_loss /= len(test_loader.dataset)

  print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

def main():
  device = torch.device("cuda") # 将构建的张量或者模型分配到相应的设备上

  kwargs = {'num_workers': 1, 'pin_memory': True}

  train_dataset = OSSMnistDataset( # 读取训练数据
    args.inputs,
    ak,
    akSec,
    train=True,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))

  train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) # 分布式采样器

  train_loader = torch.utils.data.DataLoader( # 对数据进行batch的划分
    train_dataset, sampler=train_sampler,
    batch_size=64, **kwargs)

  test_dataset = OSSMnistDataset(
    args.inputs,
    ak,
    akSec,
    train=False,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
  test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=64, **kwargs)

  model = Net().to(device) # 将模型加载到设备上
  model = DDP(model) # 分布式 
  optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5) # 优化器

  for epoch in range(1, 10): # 训练及测试
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)


  # bucket_name
  # auth = oss2.Auth(ak, akSec)
  # bucket_name = "test-experiment-0202-2"
  # bucket = oss2.Bucket(auth, endpoint, bucket_name)
  buffer_end = BytesIO()
  torch.save(model.state_dict(), buffer_end)
  bucket.put_object('net.pt', buffer_end.getvalue())

  # temp = bucket_error


  
  # torch.save(model.state_dict() ,'net_kwargs.pt')
  # print('net_kwargs.pt saved')
  # torch.save(model ,'net.pt')
  # print('net.pt saved')
  # with open('./result.txt','w') as file:
  #   file.write('net_kwargs.pt & net.pt saved')

  

if __name__ == '__main__':
  torch.distributed.init_process_group("nccl") # 初始化分布式进程组
  main()