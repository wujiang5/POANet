#%%
import math
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
#%%
def eular2reverse(eular):
    x = math.radians(- eular[0])
    y = math.radians(- eular[1])
    z = math.radians(- eular[2])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(x), -np.sin(x)],
                   [0, np.sin(x), np.cos(x)]])
    Ry = np.array([[np.cos(y), 0, np.sin(y)],
                   [0, 1, 0],
                   [-np.sin(y), 0, np.cos(y)]])
    Rz = np.array([[np.cos(z), -np.sin(z), 0],
                   [np.sin(z), np.cos(z), 0],
                   [0, 0, 1]])
    R = Rx.dot(Ry.dot(Rz))
    return R


def eular2matrix(eular):
    x = math.radians(eular[0])
    y = math.radians(eular[1])
    z = math.radians(eular[2])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(x), -np.sin(x)],
                   [0, np.sin(x), np.cos(x)]])
    Ry = np.array([[np.cos(y), 0, np.sin(y)],
                   [0, 1, 0],
                   [-np.sin(y), 0, np.cos(y)]])
    Rz = np.array([[np.cos(z), -np.sin(z), 0],
                   [np.sin(z), np.cos(z), 0],
                   [0, 0, 1]])
    R = Rz.dot(Ry.dot(Rx))
    return R

class PoseDataset(Dataset):
    def __init__(self, inputPath, mode):
        self.inputPath = inputPath
        with open(inputPath + 'pose/' + mode + '.txt', 'r') as file:
            self.inputFile = [line.rstrip('\n') for line in file.readlines()]
        self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.3, 0.3, 0.3])])

        name = []
        pose = np.empty((0, 3))
        with open(inputPath + 'pose/' + mode + '.txt', 'r') as file:
            for line in file:
                elements = line.strip().split()
                name.append(elements[0])
                data = np.array([[   float(elements[1]), float(elements[2]), float(elements[3]) ],])
                pose = np.concatenate((pose, data), axis=0)

        self.name = name
        self.pose = pose
        self.pose_matrix = torch.empty(len(self.inputFile), 3, 3)
        self.pose_reverse = torch.empty(len(self.inputFile), 3, 3)
        for index in range(len(self.inputFile)):
            self.pose_matrix[index] = torch.FloatTensor(eular2matrix(self.pose[index, 0:3]))
            self.pose_reverse[index]= torch.FloatTensor(eular2reverse(self.pose[index, 0:3]))
        self.pose_regre = torch.FloatTensor(self.pose[:, 0:3])
    def __getitem__(self, index):
        image = Image.open(self.inputPath + 'image/' + self.name[index])
        image = image.convert('RGB')
        image = self.transform(image)
        return image, self.pose_matrix[index], self.pose_reverse[index], self.pose_regre[index], self.name[index]
    def __len__(self):
        return len(self.inputFile)