function [img_output] = zBufferC(triangle, vertex, normal)
    img = zeros(224,224,3);
    vertexColor = (normal + 1.0)/2;

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
    [img_output] = zBuffer(triangle, vertex, vertexColor, img, nver, ntri, height, width, channel);
    img_output = uint8(img_output);
end