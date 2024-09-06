clc,clear,close;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Option1. Download from BIWI official website 
% (The URL seems to be temporarily unavailable)
% https://data.vision.ee.ethz.ch/cvl/gfanelli/head_pose/head_forest.html
% 
% Download "kinect_head_pose_db.tar" and unzip it to "hpdb" folder
% Download "head_pose_masks.tgz" and unzip it to "head_pose_masks" folder
% set pathImage = '/yourpath/hpdb/'
% set pathMask  = '/yourpath/head_pose_masks/'
% 
% Option2. Download from kaggle website
% 
% https://www.kaggle.com/datasets/kmader/biwi-kinect-head-pose-database?resource=download
% Download "archive.zip" and unzip it to "archive" folder
% set pathImage = '/yourpath/archive/faces_0/'
% set pathMask  = '/yourpath/archive/head_pose_masks/'
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

pathImage = '../../../archive/faces_0/';
pathMask  = '../../../archive/head_pose_masks/';
pathNormal= '../../dataset/biwi/image/';
if ~exist(pathNormal, 'dir')
    mkdir(pathNormal)
end
height = 480;
width = 640;
for i = 1:24
    cam_intrinsic = zeros(9,1);
    fid = fopen([pathImage, num2str(i,'%02d'),'/depth.cal'], 'r');
    cam_intrinsic(1:3) = fscanf(fid, '%f %f %f\n',3);
    cam_intrinsic(4:6) = fscanf(fid, '%f %f %f\n',3);
    cam_intrinsic(7:9) = fscanf(fid, '%f %f %f\n',3);
    fclose(fid);
    for j = 0:999
        fileDepth = [pathImage, num2str(i,'%02d'),'/frame_', num2str(j,'%05d'), '_depth.bin'];
        fileMask  = [pathMask,  num2str(i,'%02d'),'/frame_', num2str(j,'%05d'), '_depth_mask.png'];
        fileNormal= [pathNormal,num2str(i,'%02d'), '_', num2str(j,'%04d'), '.jpg'];
        if ~exist(fileMask,'file')||~exist(fileDepth,'file')
            continue
        end
        [depth.x, depth.y, depth.z] = mxReadDepthFile(fileDepth, cam_intrinsic);
        mask = imread(fileMask);
        flag = find(mask);
        
        FV.faces = delaunay(depth.x(flag), depth.y(flag));
        FV.vertices = [depth.x(flag), depth.y(flag), depth.z(flag)];
        normal = normDirection(FV.faces, FV.vertices);
        normal = (normal + 1)/2;
        image = zeros(height, width, 3);
        image(flag + 0 * height*width) = normal(:,1);
        image(flag + 1 * height*width) = normal(:,2);
        image(flag + 2 * height*width) = normal(:,3);


        
        cx = 0;
        cy = 0;
        cc = 0;
        for ii = 1:480
            for jj = 1:640
                if mask(ii,jj)==255
                    cy = cy + ii;
                    cx = cx + jj;
                    cc = cc + 1;
                end
            end
        end
        cy = floor(cy/cc);
        cx = floor(cx/cc);
		radius = (cam_intrinsic(1) + cam_intrinsic(5))/(2*mean(depth.z(flag)))/0.65 * 100;
        rect = [cx - radius, cy - radius, 2*radius, 2*radius];
        rect(1:2) = rect(1:2) +  randn(1,2) * 5;
        image = imresize(imcrop(image , rect), [224,224]);
        imwrite(image,  fileNormal);
    end
end