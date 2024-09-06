function [img_output] = zBufferC(triangle, vertex, matrixPose, vertexColor, img)
    if nargin<5
        img = zeros(224,224,3) + 255.0;
    end
    if nargin<4
        vertexNormal = normDirection(triangle, vertex);
        vertexColor = (vertexNormal + 1.0)/2;
    end
    if nargin<3
        transform = mvpTransform(224,224);
        vertex = homoTransform(vertex, transform);
    else
        transform = mvpTransform(224,224)*matrixPose;
        vertex = homoTransform(vertex, transform);
    end
    if size(triangle,1) ~=3,triangle = double(triangle');else,triangle = double(triangle);end
    if size(vertex,1) ~=3,vertex = double(vertex');else,vertex = double(vertex);end  
    if size(vertexColor,1) ~=3,vertexColor = double(vertexColor');else,vertexColor = double(vertexColor);end
    if max(img(:))<=1, img = double(img*255.0); else, img = double(img); end
    if max(vertexColor(:))<=1, vertexColor = vertexColor*255; end

    [height, width, channel] = size(img);
    vertex(2,:) = height + 1 - vertex(2,:);

    vertex = vertex - 1;
    triangle = triangle - 1;
    nver = length(vertex);
    ntri = length(triangle);
    [img_output,~] = zBuffer_1_1C(triangle, vertex, vertexColor, img, nver, ntri, height, width, channel);
    img_output = uint8(img_output);
end