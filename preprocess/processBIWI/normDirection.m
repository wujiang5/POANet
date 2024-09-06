function vertexNorm = normDirection(triangle, vertex)
    if size(vertex,1) ~=3
        vertex = vertex';
    end
    if size(triangle,1) ~=3
        triangle = triangle';
    end
    
    % triangleNorm
    pt0 = vertex(:, triangle(1, :));
    pt1 = vertex(:, triangle(2, :));
    pt2 = vertex(:, triangle(3, :));
    triangleNorm = cross(pt1 - pt0, pt2 - pt0);
    
    % vertexNorm
    vertexNorm = Tnorm_VnormC(double(triangleNorm), double(triangle), double(size(triangle,2)), double(size(vertex,2)));
    
    % normalize
    magnitude = sum(vertexNorm .* vertexNorm);
    flag = find(magnitude==0);
    magnitude(flag)=1;
    vertexNorm(1, flag) = ones(length(flag),1);
    vertexNorm = vertexNorm ./ sqrt(repmat(magnitude, 3, 1));
    vertexNorm = vertexNorm';

    