function [MatrixX] = Rx(x)
    x = deg2rad(x);
    MatrixX = [1,0,0;
        0,cos(x),-sin(x);
        0,sin(x),cos(x);];
end