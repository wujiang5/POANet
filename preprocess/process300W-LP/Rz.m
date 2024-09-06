function [MatrixZ] = Rz(z)
    z = deg2rad(z);
    MatrixZ = [cos(z),-sin(z),0;
        sin(z),cos(z),0;
        0,0,1];
end