clc,clear,close;
load('randflag.mat');
pathImage = '../../../archive/faces_0/';
pathMask  = '../../../archive/head_pose_masks/';
pathPose  = '../../dataset/biwi/pose/';

if ~exist(pathPose, 'dir')
    mkdir(pathPose)
end

poseTrain = [];
poseTest  = [];
filenameTrain = fopen([pathPose, 'filenameTrain', '.txt'], 'w');
filenameTest  = fopen([pathPose, 'filenameTest', '.txt'], 'w');

for i = 1:24
    cam_intrinsic = zeros(9,1);
    fid = fopen([pathImage, num2str(i,'%02d'),'/depth.cal'], 'r');
    cam_intrinsic(1:3) = fscanf(fid, '%f %f %f\n',3);
    cam_intrinsic(4:6) = fscanf(fid, '%f %f %f\n',3);
    cam_intrinsic(7:9) = fscanf(fid, '%f %f %f\n',3);
    fclose(fid);
    
    eular = [];
    for j = 0:999
        fileDepth = [pathImage, num2str(i,'%02d'),'/frame_', num2str(j,'%05d'), '_depth.bin'];
        filePose  = [pathImage, num2str(i,'%02d'),'/frame_', num2str(j,'%05d'), '_pose.txt'];
        fileMask  = [pathMask,  num2str(i,'%02d'),'/frame_', num2str(j,'%05d'), '_depth_mask.png'];
        if ~exist(fileMask,'file')||~exist(filePose,'file')||~exist(fileDepth,'file')
            continue
        end

        R = readmatrix(filePose);
        R = R(1:3,:)';
        pitch = atan2(R(3,2), R(3,3)) * 180 / pi;
        yaw  = - atan2(-R(3,1), sqrt(R(3,2)^2 + R(3,3)^2)) * 180 / pi;
        roll = - atan2(R(2,1), R(1,1)) * 180 / pi;
        eular = [eular;pitch,yaw,roll];
        if randflag(i) == 1
            fprintf(filenameTrain, '%s\n', [num2str(i,'%02d'), '_', num2str(j,'%04d'), '.jpg']);
        else
            fprintf(filenameTest, '%s\n', [num2str(i,'%02d'), '_', num2str(j,'%04d'), '.jpg']);
        end
    end
    if randflag(i) == 1
        poseTrain = [poseTrain; eular];
    else 
        poseTest  = [poseTest; eular];
    end
end
fclose(filenameTrain);
fclose(filenameTest);
save([pathPose, 'poseTrain.mat'], 'poseTrain');
save([pathPose, 'poseTest.mat'],  'poseTest');
