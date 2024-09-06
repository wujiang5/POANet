#%%
import os
import time
import torch
import numpy as np
import torchvision.models
from torch.utils.data import DataLoader

from lib.model import POANet
from lib.dataset import PoseDataset
from lib.loss import GeodesicLoss, compute_euler_angles_from_rotation_matrices
from lib.transform import quaternion_to_matrix, matrix_to_quaternion
DEVICE = torch.device("cuda")

NUM_EPOCH = 120
NUM_BATCH = 100
lr = 0.001
gamma = 0.1
milestones = [40, 80]

DATA = 'biwi'  #biwi, biwi(without mask), 300w-lp
NAME = 'weight/' + DATA
if not os.path.exists(NAME): os.makedirs(NAME)
FILE_TRAIN =  NAME + '/' + DATA + 'trainLog.txt'
FILE_TEST  =  NAME + '/' + DATA + 'testLog.txt'
#%%
model = POANet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2]).to(DEVICE)
train_dataset = PoseDataset('./dataset/' + DATA + '/', 'Train')
test_dataset  = PoseDataset('./dataset/' + DATA + '/', 'Test')
train_loader = DataLoader(dataset = train_dataset, batch_size = NUM_BATCH, shuffle=True, num_workers=8)
test_loader  = DataLoader(dataset = test_dataset,  batch_size = NUM_BATCH, shuffle=False)
crit = GeodesicLoss().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
#%%
for epoch in range(1, NUM_EPOCH + 1):
    START = time.time()
    model.train()

    TRAIN = .0
    MAE_TRAIN_PIT = .0
    MAE_TRAIN_YAW = .0
    MAE_TRAIN_ROLL= .0
    print('-------- {} Epoch Start--------'.format(epoch))
    for batch, (image, pose_matrix, pose_reverse, pose_regre, _) in enumerate(train_loader):
        optimizer.zero_grad()
        image = image.to(DEVICE)
        pose_matrix = pose_matrix.to(DEVICE)
        pose_reverse = pose_reverse.to(DEVICE)
        pose_regre = pose_regre.to(DEVICE)
        pred_matrix, pred_reverse = model(image)
        loss = crit(pose_matrix, pred_matrix) + crit(pose_reverse, pred_reverse)
        TRAIN += loss
        loss.backward()
        optimizer.step()

        q1 = matrix_to_quaternion(pred_matrix)
        q2 = matrix_to_quaternion(torch.transpose(pred_reverse, 1, 2))
        q = torch.lerp(q1, q2, 0.5)
        matrix = quaternion_to_matrix(q)

        pred_euler = compute_euler_angles_from_rotation_matrices(matrix) * 180 / np.pi
        MAE_TRAIN_PIT += torch.sum(torch.abs(pred_euler[:,0] - pose_regre[:, 0]))
        MAE_TRAIN_YAW += torch.sum(torch.abs(pred_euler[:,1] - pose_regre[:, 1]))
        MAE_TRAIN_ROLL+= torch.sum(torch.abs(pred_euler[:,2] - pose_regre[:, 2]))

        if (batch + 1) % 200 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss %.2f' % (
                epoch, NUM_EPOCH, batch + 1, len(train_dataset) // NUM_BATCH, loss.item()))
    scheduler.step()
    model.eval()
    TEST = .0
    MAE_TEST_PIT = .0
    MAE_TEST_YAW = .0
    MAE_TEST_ROLL= .0
    with torch.no_grad():
        for image, pose_matrix, pose_reverse, pose_regre, _ in test_loader:
            image = image.to(DEVICE)
            pose_matrix = pose_matrix.to(DEVICE)
            pose_reverse = pose_reverse.to(DEVICE)
            pose_regre = pose_regre.to(DEVICE)

            pred_matrix, pred_reverse = model(image)
            loss = crit(pose_matrix, pred_matrix) + crit(pose_reverse, pred_reverse)

            q1 = matrix_to_quaternion(pred_matrix)
            q2 = matrix_to_quaternion(torch.transpose(pred_reverse, 1, 2))
            q = torch.lerp(q1, q2, 0.5)
            matrix = quaternion_to_matrix(q)

            TEST+=loss
            pred_euler = compute_euler_angles_from_rotation_matrices(matrix) * 180 / np.pi
            MAE_TEST_PIT += torch.sum(torch.abs(pred_euler[:,0] - pose_regre[:, 0]))
            MAE_TEST_YAW += torch.sum(torch.abs(pred_euler[:,1] - pose_regre[:, 1]))
            MAE_TEST_ROLL+= torch.sum(torch.abs(pred_euler[:,2] - pose_regre[:, 2]))
    TRAIN = TRAIN / (len(train_dataset)// NUM_BATCH)
    TEST  = TEST  / (len(test_dataset) // NUM_BATCH)

    STRING_TRAIN= 'Epoch = %d, Train Loss %.2f, pit %.2f, yaw %.2f, roll %.2f, avg %.2f\n' % (epoch, TRAIN.item(),
        MAE_TRAIN_PIT/len(train_dataset), MAE_TRAIN_YAW/len(train_dataset), MAE_TRAIN_ROLL/len(train_dataset),
        (MAE_TRAIN_PIT + MAE_TRAIN_YAW + MAE_TRAIN_ROLL)/(3*len(train_dataset)))
    STRING_TEST = 'Epoch = %d, Test Loss %.2f, pit %.2f, yaw %.2f, roll %.2f, avg %.2f\n' % (epoch, TEST.item(),
        MAE_TEST_PIT / len(test_dataset), MAE_TEST_YAW / len(test_dataset), MAE_TEST_ROLL / len(test_dataset),
        (MAE_TEST_PIT + MAE_TEST_YAW + MAE_TEST_ROLL)/(3*len(test_dataset)))

    print(STRING_TRAIN)
    print(STRING_TEST)
    print("Epoch Time: {:02f} Sec".format(time.time() - START))
    with open(FILE_TRAIN, 'a') as file: file.write(STRING_TRAIN)
    with open(FILE_TEST, 'a') as file: file.write(STRING_TEST)

    current_mae = (MAE_TEST_PIT + MAE_TEST_YAW + MAE_TEST_ROLL) / (3 * len(test_dataset))
    if epoch == 1:
        best_mae = current_mae
        torch.save(model, NAME + '/bestmodel.pth')
    elif current_mae < best_mae:
        best_mae = current_mae
        torch.save(model, NAME + '/bestmodel.pth')