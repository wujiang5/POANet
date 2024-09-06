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
filenameTrain = fopen([pathPose, 'Train', '.txt'], 'w');
filenameTest  = fopen([pathPose, 'Test', '.txt'], 'w');

for i = 1:length(folder)
    listImage = dir(fullfile([pathImage, folder{1,i}], '*.mat'));
    eular = [];
    for j = 1:length(listImage)
        Para = load([pathImage,folder{1,i},'\', listImage(j).name]);
        pitch  = rad2deg(Para.Pose_Para(1));
        yaw  = rad2deg(Para.Pose_Para(2));
        roll = rad2deg(Para.Pose_Para(3));
        name = listImage(j).name;
        if i == 1
            fprintf(filenameTest, '%s %f %f %f\n', [folder{1,i}, '/', name(1:end-4), '.jpg'], pitch, yaw, roll);
        else
            fprintf(filenameTrain, '%s %f %f %f\n', [folder{1,i}, '/', name(1:end-4), '.jpg'], pitch, yaw, roll);
        end
    end
end
fclose(filenameTrain);
fclose(filenameTest);