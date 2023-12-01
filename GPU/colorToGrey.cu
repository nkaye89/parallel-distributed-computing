#include <iostream>
#include <time.h>
#include <stdio.h>
#include "opencv2/opencv.hpp"
using namespace cv;
using namespace std;

// compile: nvcc colorToGrey.cu `pkg-config opencv --cflags --libs`
// run    : ./a.out

// Make your code change only to this function to implement lab problem 3.
__global__ void rgb2grayincuda(uchar3 *d_in, unsigned char *d_out, 
                                uint imgheight, uint imgwidth)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < imgwidth/2 && idy < imgheight)
    {
        uchar3 rgb = d_in[idy * imgwidth + idx];
        d_out[idy * imgwidth + idx] = 0.299f * rgb.x + 0.587f * rgb.y + 0.114f * rgb.z;
    }
    else
    {
	d_out[idy * imgwidth + idx] = 0;
    }
}

int main(void)
{
    // Read an image from the jpg file stored on hard disk file
    // Mat is a matrix of numbers containing (Red, Green, Blue) pixels of the image
    Mat srcImage = imread("./e1.jpg");
    
    // Dimensions of the image
    uint imgheight = srcImage.rows;
    uint imgwidth = srcImage.cols;

    // This is where we will store the output grayscale image corresponding to the input image
    Mat grayImage(imgheight, imgwidth, CV_8UC1, Scalar(0));

    // Now we allocate memory on the GPU for input image 
    uchar3 *d_in;
    cudaMalloc((void**)&d_in, imgheight*imgwidth*sizeof(uchar3));
    
    // Now we allocate memory on the GPU for output image 
    unsigned char *d_out;
    cudaMalloc((void**)&d_out, imgheight*imgwidth*sizeof(unsigned char));

    // Now we copy the input image data to the gpu memory ( using variable d_in) 
    cudaMemcpy(d_in, srcImage.data, imgheight*imgwidth*sizeof(uchar3), cudaMemcpyHostToDevice);
    
    // Now we create thread blocks: 32 threads along the x-axis and 32 threads along the y-axis
    dim3 threadsPerBlock(32, 32);
    
    // Now we find out how many blocks are needed along the x-axis and y-axis
    // In 2D, there are rows of blocks and columns of blocks
    int blockColumns = (imgwidth + threadsPerBlock.x - 1) / threadsPerBlock.x;
    int blockRows    = (imgheight + threadsPerBlock.y - 1) / threadsPerBlock.y;
   
    // Specify the block dimensions
    dim3 blocksPerImage( blockColumns, blockRows );

    // Finally, call the function to do image processing on GPU defined before main function.
    rgb2grayincuda<< <blocksPerImage, threadsPerBlock>> >(d_in, d_out, imgheight, imgwidth);

    // GPU has done its job, let us copy the output image's data to CPU
    cudaMemcpy(grayImage.data, d_out, imgheight*imgwidth*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    
    // Let us now create an image so that we can see it on CPU
	imwrite("greyImage.jpg", grayImage);
	
	// Type ls after the program finishes execution, then, you will find a new image "greyImage.jpg"
	// Copy the image to your laptop from Everest server to see the image.
	
	// Let us release the memory, we don't need it any more
    cudaFree(d_in);	
    cudaFree(d_out);
    
    return 0;

}
