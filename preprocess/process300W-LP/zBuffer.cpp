#include <mex.h>
#include <matrix.h>
#include <math.h>
#include <algorithm>
#define min(i, j) (((i) < (j)) ? (i) : (j))
#define max(i, j) (((i) > (j)) ? (i) : (j))
void zBuffer(double* triangle, double* vertex, double* vertexColor, double * img_input,int nver, int ntri, int height, int width, int channel, double* img_output, double* tri_output);
void mexFunction(int nlhs,mxArray *plhs[], int nrhs,const mxArray *prhs[])
{
	double* triangle;
	double* vertex;
	double* vertexColor;
	double* img_input;
	int nver;
	int ntri;
	int width;
	int height;
	int channel;

	triangle = mxGetPr(prhs[0]);
	vertex = mxGetPr(prhs[1]);
	vertexColor = mxGetPr(prhs[2]);
	img_input = mxGetPr(prhs[3]);	
	nver = (int)*mxGetPr(prhs[4]);
	ntri = (int)*mxGetPr(prhs[5]);
	height = (int)*mxGetPr(prhs[6]);
	width = (int)*mxGetPr(prhs[7]);
	channel = (int)*mxGetPr(prhs[8]);
	
	double* img_output;
	double* tri_output;

	const mwSize dims[3]={height, width, channel};
    plhs[0] = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);
    plhs[1] = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);
	img_output = mxGetPr(plhs[0]);
    tri_output = mxGetPr(plhs[1]);

    for(int x = 0; x <= width; x++)
	{
		for (int y = 0; y <= height; y++)
		{
	    	img_output[0 * width * height + x * height + y] = img_input[0 * width * height + x * height + y];
            img_output[1 * width * height + x * height + y] = img_input[1 * width * height + x * height + y];
            img_output[2 * width * height + x * height + y] = img_input[2 * width * height + x * height + y];
		    tri_output[0 * width * height + x * height + y] = -1;
        }
	}
    

    zBuffer(triangle, vertex, vertexColor, img_input, nver, ntri, height, width, channel, img_output, tri_output);
 
}
void zBuffer(double* triangle, double* vertex, double* vertexColor, double * img_input,int nver, int ntri, int height, int width, int channel, double* img_output, double* tri_output)
{
    struct Point2D{
        double x;
        double y;
    };
    int i,j,k;
    double centerZ;
	double* point1 = new double[2 * ntri];
	double* point2 = new double[2 * ntri];
	double* point3 = new double[2 * ntri];
    double* imgh = new double[width * height];
    double* trih = new double[ntri];
    double* tritex = new double[ntri * 3];
    
    struct Point2D pt0,pt1,pt2,pt3;
    double L1,L2,L3;
    bool flag;
    int xmin,xmax,ymin,ymax;
	for(int i = 0; i < width * height; i++)
    {
		imgh[i] = -99999999999999;
    }
    
    for(i = 0; i < ntri; i++)
	{
		int p1 = int(triangle[3*i]);
		int p2 = int(triangle[3*i + 1]);
		int p3 = int(triangle[3*i + 2]);
        int x1 = vertex[3*p1];
        int x2 = vertex[3*p2];
        int x3 = vertex[3*p3];
		point1[2*i] = vertex[3*p1];	point1[2*i+1] = vertex[3*p1+1];
		point2[2*i] = vertex[3*p2];	point2[2*i+1] = vertex[3*p2+1];
		point3[2*i] = vertex[3*p3];	point3[2*i+1] = vertex[3*p3+1];
        centerZ = (vertex[3*p1+2] + vertex[3*p2+2] + vertex[3*p3+2]) / 3;
        trih[i] = centerZ;

        tritex[3*i+0] = (vertexColor[3*p1+0] + vertexColor[3*p2+0] + vertexColor[3*p3+0]) / 3;
		tritex[3*i+1] = (vertexColor[3*p1+1] + vertexColor[3*p2+1] + vertexColor[3*p3+1]) / 3;
		tritex[3*i+2] = (vertexColor[3*p1+2] + vertexColor[3*p2+2] + vertexColor[3*p3+2]) / 3;

        
    }
    for(i = 0; i < ntri; i++)
	{
        
        pt1.x = point1[2*i]; pt1.y = point1[2*i+1];
		pt2.x = point2[2*i]; pt2.y= point2[2*i+1];
		pt3.x = point3[2*i]; pt3.y = point3[2*i+1];
        xmin = (int)ceil(min(min(pt1.x,pt2.x),pt3.x));
        xmax = (int)floor(max(max(pt1.x,pt2.x),pt3.x));
        ymin = (int)ceil(min(min(pt1.y,pt2.y),pt3.y));
        ymax = (int)floor(max(max(pt1.y,pt2.y),pt3.y));
        if(xmax < xmin || ymax < ymin || xmax > width-1 || xmin < 0 || ymax > height-1 || ymin < 0)
			continue;
        for(int x = xmin; x <= xmax; x++)
	    {
		    for (int y = ymin; y <= ymax; y++)
		    {
                pt0.x = x;
                pt0.y = y;
            
                L1 = -((pt0.x-pt1.x)*(pt2.y-pt1.y)) + ((pt0.y-pt1.y)*(pt2.x-pt1.x));
                L2 = -((pt0.x-pt2.x)*(pt3.y-pt2.y)) + ((pt0,y-pt2.y)*(pt3.x-pt2.x));;
                L3 = -((pt0.x-pt3.x)*(pt1.y-pt3.y)) + ((pt0.y-pt3.y)*(pt1.x-pt3.x));
                flag = ((L1>0)&&(L2>0)&&(L3>0))||((L1<0)&&(L2<0)&&(L3<0));
                if( imgh[x * height + y] < trih[i] && flag)
			    {
                    imgh[x * height + y] = trih[i];
                    img_output[0 * width * height + x * height + y] =  tritex[3 * i + 0];
                    img_output[1 * width * height + x * height + y] =  tritex[3 * i + 1];
                    img_output[2 * width * height + x * height + y] =  tritex[3 * i + 2];
                }
		    }
	    }
    }	
	
	delete[] point1;
	delete[] point2;
	delete[] point3;
    delete[] imgh;
    delete[] trih;
    delete[] tritex;
}
