%%
clc,clear,close;
pathImage = 'D:/HeadPoseEstimation/2016_3DDFA/300W_LP/';
pathPose  = '../../dataset/300w-lp/pose/';
if ~exist(pathPose, 'dir')
    mkdir(pathPose)
end
folder = { 'AFLW2000', ...
    'AFW', 'AFW_Flip',  ...
    'HELEN', 'HELEN_Flip', ...
    'IBUG', 'IBUG_Flip', ...
    'LFPW', 'FLPW_Flip'};
poseTrain = [];
poseTest  = [];
filenameTrain = fopen([pathPose, 'filenameTrain', '.txt'], 'w');
filenameTest  = fopen([pathPose, 'filenameTest', '.txt'], 'w');

for i = 1:length(folder)
    listImage = dir(fullfile([pathImage, folder{1,i}], '*.mat'));
    eular = [];
    for j = 1:length(listImage)
        Para = load([pathImage,folder{1,i},'\', listImage(j).name]);
        pit  = Para.Pose_Para(1);
        yaw  = Para.Pose_Para(2);
        roll = Para.Pose_Para(3);
        eular = [eular; rad2deg(pit),rad2deg(yaw),rad2deg(roll)];

        name = listImage(j).name;
        if i == 1
            fprintf(filenameTest, '%s\n', [folder{1,i}, '/', name(1:end-4), '.jpg']);
        else
            fprintf(filenameTrain, '%s\n', [folder{1,i}, '/', name(1:end-4), '.jpg']);
        end
    end
    if i == 1
        poseTest = [poseTest; eular];
    else 
        poseTrain  = [poseTrain; eular];
    end
end
fclose(filenameTrain);
fclose(filenameTest);
save([pathPose, 'poseTrain.mat'], 'poseTrain');
save([pathPose, 'poseTest.mat'],  'poseTest');