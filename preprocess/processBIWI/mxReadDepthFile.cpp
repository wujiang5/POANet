#include <fstream>
#include <stdint.h>
#include "mex.h"
#include <sstream>
#include <string>
/**
 * Usage: [x,y,z] = mxReadDepthFile('frame_XXXXX_depth.bin',cam_intrinsic,zThresh);
 * zThresh is the max Z val in mm. All point beyond zThresh
 * are discarded.
 */

using namespace std;
void mexFunction (int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[])
{
    /*-----------------------------------------------------------*/
    /* Check the number of input arguments*/
    if (nrhs != 2 )
        mexErrMsgTxt("Incorrect number of input arguments!");
    
    /* Get the file name for the compressed depth image*/
    char* fname = mxArrayToString(prhs[0]);    
    
    /* Read the intrinsic camera parameters*/
	double* depth_intrinsic = mxGetPr(prhs[1]);;	
    
    /* Open the compressed binary depth image*/
 	FILE* pFile = fopen(fname, "rb");
	if(!pFile)
    {
		mexErrMsgTxt("Could not open file!");
		return;
	}
    
    /* Get the image width and height*/
  	int im_width = 0; int im_height = 0;
 	fread(&im_width,sizeof(int),1,pFile); 
  	fread(&im_height,sizeof(int),1,pFile);
 
    /* Storage for the depth data */
 	int16_t* depth_img = new int16_t[im_width*im_height];

    /* Read the binary depth file */
 	int numempty;
 	int numfull;
 	int p = 0;
    while(p < im_width*im_height )
    {
  		fread( &numempty,sizeof(int),1,pFile);
		for(int i = 0; i < numempty; i++)
			depth_img[ p + i ] = 0;

		fread( &numfull,sizeof(int), 1, pFile);
		fread( &depth_img[ p + numempty ], sizeof(int16_t), numfull, pFile) ;
		p += numempty+numfull;
	}

	fclose(pFile);
    
    /*-----------------------------------------------------------*/
    /* Get the 3D data from depth file */
    plhs[0] = mxCreateDoubleMatrix(im_height,im_width,mxREAL);
    plhs[1] = mxCreateDoubleMatrix(im_height,im_width,mxREAL);
    plhs[2] = mxCreateDoubleMatrix(im_height,im_width,mxREAL);
    double *X = mxGetPr(plhs[0]);
    double *Y = mxGetPr(plhs[1]);
    double *Z = mxGetPr(plhs[2]);
    
    int g_max_z = 10000;
    
	for(int y = 0; y < im_height; y++)
	{
        for(int x = 0; x < im_width; x++)
        {
            // Indexing is according to row order format in C
            float d = depth_img[y*im_width+x];
			if ( d < g_max_z && d > 0 )
            {
                // Indexing is according to col order format in Matlab
				X[x*im_height+y] = d * (float(x) - depth_intrinsic[2])/depth_intrinsic[0];
				Y[x*im_height+y] = d * (float(y) - depth_intrinsic[5])/depth_intrinsic[4];
				Z[x*im_height+y] = d;
			}
			else
            {
				X[x*im_height+y] = 0.0;
				Y[x*im_height+y] = 0.0;
				Z[x*im_height+y] = 0.0;
			}
		}
	}
   
    delete [] depth_img;
    mxFree(fname);
    return;
}