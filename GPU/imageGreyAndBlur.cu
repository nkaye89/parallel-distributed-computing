#include <iostream>
#include <time.h>
#include <stdio.h>
#include "opencv2/opencv.hpp"
using namespace cv;
using namespace std;

#define BLUR_SIZE 3
// 5:34 pm
__global__ void rgb2grayincuda(uchar3 * const d_in, unsigned char * const d_out, 
                                uint imgheight, uint imgwidth)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < imgwidth && idy < imgheight)
    {
        uchar3 rgb = d_in[idy * imgwidth + idx];
        d_out[idy * imgwidth + idx] = 0.299f * rgb.x + 0.587f * rgb.y + 0.114f * rgb.z;
    }
}

__global__ void blurKernel(uchar3 * const d_in, uchar3 * const d_out, 
                                uint imgheight, uint imgwidth)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;  // column
    const unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;  // row

	if (idx < imgwidth && idy < imgheight) {
		int pixVal1 = 0;
		int pixVal2 = 0;
		int pixVal3 = 0;
		int pixels = 0;

		// Get the average of the surrounding BLUR_SIZE x BLUR_SIZE box
		for (int blurrow = -BLUR_SIZE; blurrow < BLUR_SIZE + 1; ++blurrow) {
			for (int blurcol = -BLUR_SIZE; blurcol < BLUR_SIZE + 1; ++blurcol) {
				int currow = idy + blurrow;
				int curcol = idx + blurcol;
				// Verify we have a valid image pixel
				if (currow > -1 && currow < imgheight && curcol > -1 && curcol < imgwidth) {
					pixVal1 += d_in[currow * imgwidth + curcol].x;
					pixVal2 += d_in[currow * imgwidth + curcol].y;
					pixVal3 += d_in[currow * imgwidth + curcol].z;
					pixels++; // Keep track of number of pixels in the avg
				}
			}
		}
		// Write our new pixel value out
		d_out[idy * imgwidth + idx].x = (unsigned char)(pixVal1 / pixels);
		d_out[idy * imgwidth + idx].y = (unsigned char)(pixVal2 / pixels);
		d_out[idy * imgwidth + idx].z = (unsigned char)(pixVal3 / pixels);
	}
}

int main(void)
{
    //Mat srcImage = imread("input_images/e1.jpg");
    Mat srcImage = imread("input_images/lenna.png");
    const uint imgheight = srcImage.rows;
    const uint imgwidth = srcImage.cols;

    Mat grayImage(imgheight, imgwidth, CV_8UC1, Scalar(0));

    uchar3 *d_in;
    unsigned char *d_out;

    cudaMalloc((void**)&d_in, imgheight*imgwidth*sizeof(uchar3));
    cudaMalloc((void**)&d_out, imgheight*imgwidth*sizeof(unsigned char));

    cudaMemcpy(d_in, srcImage.data, imgheight*imgwidth*sizeof(uchar3), cudaMemcpyHostToDevice);
    
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((imgwidth + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (imgheight + threadsPerBlock.y - 1) / threadsPerBlock.y);

    rgb2grayincuda<< <blocksPerGrid, threadsPerBlock>> >(d_in, d_out, imgheight, imgwidth);

    cudaMemcpy(grayImage.data, d_out, imgheight*imgwidth*sizeof(unsigned char), cudaMemcpyDeviceToHost);
	
    cudaFree(d_out);

	//imwrite("output_images/lennaGreyImage.jpg",grayImage);
	
  
	Mat blurImage(imgheight, imgwidth, CV_8UC3);
	uchar3 *d_out2;
	cudaMalloc((void**)&d_out2, imgheight*imgwidth*sizeof(uchar3));
	blurKernel<< <blocksPerGrid, threadsPerBlock>> >(d_in, d_out2, imgheight, imgwidth);
	cudaMemcpy(blurImage.data, d_out2, imgheight*imgwidth*sizeof(uchar3), cudaMemcpyDeviceToHost);
	cudaFree(d_in);
	cudaFree(d_out2);
	imwrite("output_images/lennaBlurImage.jpg",blurImage);
	
    return 0;

}
