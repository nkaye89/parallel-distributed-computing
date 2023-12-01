#include <iostream>
#include <time.h>
#include <stdio.h>
#include "opencv2/opencv.hpp"
using namespace cv;
using namespace std;

// compile: nvcc colorToGreyWithRot.cu `pkg-config opencv --cflags --libs`
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

__global__ void rotateincuda(uchar3 *d_in, uchar3 *d_out,
                            uint imgheight, uint imgwidth)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if(idx < imgheight && idy < imgwidth)
    {
        int rotx = imgwidth - idy;
        int roty = idx;

        uchar3 pixel = d_in[idx * imgwidth + idy];

        d_out[rotx * imgheight + roty] = pixel;
    }

}

int main(void)
{
    // Read an image from the jpg file stored on hard disk file
    // Mat is a matrix of numbers containing (Red, Green, Blue) pixels of the image
    Mat srcImage = imread("./e1.jpg");

    // Dimensions of the source image
    uint imgheight = srcImage.rows;
    uint imgwidth = srcImage.cols;

    // Now we allocate memory on the GPU for input image
    uchar3 *d_in;
    cudaMalloc((void**)&d_in, imgheight*imgwidth*sizeof(uchar3));

    // Now we copy the input image data to the gpu memory ( using variable d_in)
    cudaMemcpy(d_in, srcImage.data, imgheight*imgwidth*sizeof(uchar3), cudaMemcpyHostToDevice);

    // Create thread blocks
    dim3 threadsPerBlock(32, 32);
    int blockColumns = (imgwidth + threadsPerBlock.x - 1) / threadsPerBlock.x;
    int blockRows    = (imgheight + threadsPerBlock.y - 1) / threadsPerBlock.y;
    dim3 blocksPerImage(blockColumns, blockRows);

    // Allocate memory for rotated image
    uchar3 *d_out;
    cudaMalloc((void**)&d_out, imgheight*imgwidth*sizeof(uchar3));

    // Rotate input image 90 degrees
    rotateincuda<<<blocksPerImage, threadsPerBlock>>>(d_in, d_out,
                                                     imgheight, imgwidth);

    // Copy rotated image from GPU and save
    Mat rotatedImage(imgheight, imgwidth, CV_8UC3);
    cudaMemcpy(rotatedImage.data, d_out,
               imgheight*imgwidth*sizeof(uchar3), cudaMemcpyDeviceToHost);
    imwrite("rotatedImage.jpg", rotatedImage);

    // Release memory
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
