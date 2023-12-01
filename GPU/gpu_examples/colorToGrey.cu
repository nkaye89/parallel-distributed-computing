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
    
    // if condition makes sure that the threads that are outside the range of image 
    // do not enter the if-block and do not write to d_out pixel array.
    if (idx < imgwidth && idy < imgheight)
    {
        uchar3 rgb = d_in[idy * imgwidth + idx];
        // rgb.x is the red component
        // rgb.y is the green component
        // rgb.z is the blue component
        
        // Apply the formula to get a single number (shade of grey) from RGB values
        d_out[idy * imgwidth + idx] = 0.299f * rgb.x + 0.587f * rgb.y + 0.114f * rgb.z;
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
    
    /*
     Now we find out how many thread blocks are needed along the x-axis and y-axis
     In 2D, we want to create rows of blocks and columns of blocks
     As discussed in class, the number of threads to be created will be more
     than the number of pixels. 
     The calculation below does the same thing as a ceiling function.
     which maps a floating point number x to the least integer greater than or equal to x
     */
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
