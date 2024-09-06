%%
clc,clear,close;
pathImage = 'D:/HeadPoseEstimation/2016_3DDFA/300W_LP/';
pathNormal= '../../dataset/300w-lp/image/'; 

folder = { 'AFLW2000', ...
    'AFW', 'AFW_Flip',  ...
    'HELEN', 'HELEN_Flip', ...
    'IBUG', 'IBUG_Flip', ...
    'LFPW', 'FLPW_Flip'};

Shp = load('Model_Shape.mat');
Exp = load('Model_Exp.mat');
mu  = Shp.mu_shape + Exp.mu_exp;
tri = Shp.tri;
nver = length(mu);
ntri = length(tri);

height = 224;
width = 224;
channel = 3;
for i = 1:length(folder)
    listImage = dir(fullfile([pathImage, folder{1,i}], '*.mat'));
    if ~exist([pathNormal, folder{1,i}], 'dir')
        mkdir([pathNormal, folder{1,i}]);
    end
    for j = 1:length(listImage)
        Para = load([pathImage,folder{1,i},'\', listImage(j).name]);
        vertex = mu + Shp.w * Para.Shape_Para + Exp.w_exp * Para.Exp_Para;
        vertex = reshape(vertex, 3, length(vertex)/3);
        norm = NormDirection(vertex, Shp.tri);

        pit  = Para.Pose_Para(1);
        yaw  = Para.Pose_Para(2);
        roll = Para.Pose_Para(3);
        scale= Para.Pose_Para(7) * 224/450;
        t3d = [112;112;1] + randn(3,1) * 5;
        R = RotationMatrix(pit, yaw, roll);
        ProjectVertex = scale * R * vertex + repmat(t3d, 1, size(vertex, 2));
        image = zBufferC(tri, ProjectVertex, norm);
        
        name = listImage(j).name;
        imwrite(image, [pathNormal, folder{1,i},'/', name(1:end-3), 'jpg']);
    end
end
