#%%
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from lib.model import POANet
from lib.loss import GeodesicLoss, compute_euler_angles_from_rotation_matrices
from lib.transform import quaternion_to_matrix, matrix_to_quaternion
from lib.dataset import PoseDataset
DEVICE = torch.device("cuda")

NUM_BATCH = 100
DATA = 'biwi'  #biwi, biwi(without mask), 300w-lp
NAME = 'weight/' + DATA
#%%
model = torch.load(NAME + '/bestmodel.pth')
model = model.to(DEVICE)
test_dataset  = PoseDataset('./dataset/' + DATA + '/', 'Test')
test_loader  = DataLoader(dataset = test_dataset,  batch_size = NUM_BATCH, shuffle=False)
crit = GeodesicLoss().to(DEVICE)
#%%
model.eval()
TEST = .0
MAE_TEST_PIT = .0
MAE_TEST_YAW = .0
MAE_TEST_ROLL= .0

angle = np.empty((0, 3))
with torch.no_grad():
    for image, pose_matrix, pose_reverse, pose_regre, _ in test_loader:
        start = time.time()

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
        angle = np.concatenate((angle,  pose_regre[:,0:3].cpu().numpy()), axis=0)


filename = NAME + '/predition.txt'
with open(filename, 'w') as file:
    for row in angle:
        line = ' '.join(map(str, row))
        file.write(line + '\n')
TEST  = TEST  / (len(test_dataset) // NUM_BATCH)
STRING_TEST = 'Test Loss %.2f, pit %.2f, yaw %.2f, roll %.2f, avg %.2f\n' % (TEST.item(),
    MAE_TEST_PIT / len(test_dataset), MAE_TEST_YAW / len(test_dataset), MAE_TEST_ROLL / len(test_dataset),
    (MAE_TEST_PIT + MAE_TEST_YAW + MAE_TEST_ROLL)/(3*len(test_dataset)))
print(STRING_TEST)