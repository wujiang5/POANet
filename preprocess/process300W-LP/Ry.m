function [MatrixY] = Ry(y)
    y = deg2rad(y);
    MatrixY = [cos(y),0,sin(y);
            0,1,0;
            -sin(y),0,cos(y)];
end