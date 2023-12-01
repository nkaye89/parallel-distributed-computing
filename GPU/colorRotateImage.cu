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
		uchar3 rgb = {255, 255, 255};
		d_out[idy * imgwidth + idx] = 255;
	}
}

__global__ void rotateImage(uchar3 *d_in, uchar3 *d_out,
                            uint imgheight, uint imgwidth)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < imgheight && y < imgwidth) {
        // Calculate new row and column indices by rotating (x, y)
        int new_x = imgwidth - 1 - y;
        int new_y = x;

        // Read from input image at (x, y)
        uchar3 color = d_in[x * imgwidth + y];

        // Write to output image at (new_x, new_y)
        d_out[new_x * imgheight + new_y] = color;
    }
}


int main(void)
{
    // Read an image from the jpg file stored on hard disk file
    Mat srcImage = imread("./e1.jpg");

    // Dimensions of the image
    uint imgheight = srcImage.rows;
    uint imgwidth = srcImage.cols;

    // Allocate memory on the GPU for input image
    uchar3 *d_in;
    cudaMalloc((void**)&d_in, imgheight*imgwidth*sizeof(uchar3));

    // Allocate memory on the GPU for output grayscale image
    unsigned char *d_out;
    cudaMalloc((void**)&d_out, imgheight*imgwidth*sizeof(unsigned char));

    // Copy the input image data to the gpu memory
    cudaMemcpy(d_in, srcImage.data, imgheight*imgwidth*sizeof(uchar3), cudaMemcpyHostToDevice);

    // Create thread blocks
    dim3 threadsPerBlock(32, 32);
    int blockColumns = (imgwidth + threadsPerBlock.x - 1) / threadsPerBlock.x;
    int blockRows    = (imgheight + threadsPerBlock.y - 1) / threadsPerBlock.y;
    dim3 blocksPerImage(blockColumns, blockRows);

    // Convert left half of input image to grayscale
    rgb2grayincuda<<<blocksPerImage, threadsPerBlock>>>(d_in, d_out, imgheight, imgwidth);

    // Allocate memory for rotated image
    uchar3 *d_out_rotated;
    cudaMalloc((void**)&d_out_rotated, imgheight*imgwidth*sizeof(uchar3));

    // Rotate input image 90 degrees
    rotateImage<<<blocksPerImage, threadsPerBlock>>>(d_in, d_out_rotated,
                                                     imgheight, imgwidth);

    // Copy rotated image from GPU and save
    Mat rotatedImage(imgheight, imgwidth, CV_8UC3);
    cudaMemcpy(rotatedImage.data, d_out_rotated,
               imgheight*imgwidth*sizeof(uchar3), cudaMemcpyDeviceToHost);
    imwrite("rotatedImage.jpg", rotatedImage);

    // Release memory
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_out_rotated);
}