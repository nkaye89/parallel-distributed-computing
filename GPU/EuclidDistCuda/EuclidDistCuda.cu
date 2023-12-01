#include <iostream>
#include <time.h>
#include <stdio.h>
#include "opencv2/opencv.hpp"
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <sys/time.h>
using namespace cv;
using namespace std;

// compile: nvcc EuclidDistCuda.cu `pkg-config opencv --cflags --libs`
// run    : ./a.out

const int SIZE = 255;

void read_ints (const char* file_name, int arr[])
{
  FILE* file = fopen (file_name, "r");
  
  if (NULL == file) 
  {
        printf("file can't be opened \n");
        exit(1);
  }
  int i = 0;
  int counter = 0;
  fscanf (file, "%d", &i);    
  while (!feof (file))
    {  
      //printf ("%d ", i);
      arr[counter] = i;
      fscanf (file, "%d", &i);  
      counter++;    
    }
  fclose (file);        
}

__global__ void EuclidDist(int* dTest, int* dReference, int *dist)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    int d;

    if (i < SIZE) {
        d = (dTest[i] - dReference[i]) * (dTest[i] - dReference[i]);
        atomicAdd(dist, d);
    }
}


int main(void)
{
    //  create host arrays
    int *hTest, *hReference, hDist;
    hDist = 0;
    //  create device arrays
    int *dTest, *dReference, *dDist;

    //  allocate host mem
    hTest = (int*) malloc(SIZE * sizeof(int));
    hReference = (int*) malloc(SIZE * sizeof(int));

    //  allocate mem on gpu
    cudaMalloc((void**)&dTest, SIZE * sizeof(int));
    cudaMalloc((void**)&dReference, SIZE * sizeof(int));
    cudaMalloc((void**)&dDist, sizeof(int));

    //  read in test histogram
    read_ints("histograms/test/frog1.txt", hTest);

    //  set block and threads
    int blkNum = 8;
    int threadsPerBlk = 32;

//  ITSELF
    //  read in histogram
	read_ints("histograms/test/frog1.txt", hReference);
    //  Copy the arrays to the gpu memory
    cudaMemcpy(dTest, hTest, SIZE*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dReference, hReference, SIZE*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dDist, &hDist, sizeof(int), cudaMemcpyHostToDevice);
    //  launch kernel
    EuclidDist<<< blkNum , threadsPerBlk >>>(dTest, dReference, dDist);
    cudaDeviceSynchronize();
    cudaMemcpy(&hDist, dDist, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Distance from itself = %d \n", hDist);
    hDist = 0;

//  FROG2
    //  read in histogram
	read_ints("histograms/reference/frog2.txt", hReference);
    //  Copy the arrays to the gpu memory
    cudaMemcpy(dTest, hTest, SIZE*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dReference, hReference, SIZE*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dDist, &hDist, sizeof(int), cudaMemcpyHostToDevice);
    //  launch kernel
    EuclidDist<<< blkNum , threadsPerBlk >>>(dTest, dReference, dDist);
    cudaDeviceSynchronize();
    cudaMemcpy(&hDist, dDist, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Distance from frog2 = %d \n", hDist);
    hDist = 0;

//  ROTATED FROG
    //  read in histogram
	read_ints("histograms/reference/frogRotated.txt", hReference);
    //  Copy the arrays to the gpu memory
    cudaMemcpy(dTest, hTest, SIZE*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dReference, hReference, SIZE*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dDist, &hDist, sizeof(int), cudaMemcpyHostToDevice);
    //  launch kernel
    EuclidDist<<< blkNum , threadsPerBlk >>>(dTest, dReference, dDist);
    cudaDeviceSynchronize();
    cudaMemcpy(&hDist, dDist, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Distance from rotatedFrog = %d \n", hDist);
    hDist = 0;

//  BUCK
    //  read in histogram
	read_ints("histograms/reference/buck.txt", hReference);
    //  Copy the arrays to the gpu memory
    cudaMemcpy(dTest, hTest, SIZE*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dReference, hReference, SIZE*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dDist, &hDist, sizeof(int), cudaMemcpyHostToDevice);
    //  launch kernel
    EuclidDist<<< blkNum , threadsPerBlk >>>(dTest, dReference, dDist);
    cudaDeviceSynchronize();
    cudaMemcpy(&hDist, dDist, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Distance from buck = %d \n", hDist);
    hDist = 0;

//  MARQUETTE
    //  read in histogram
	read_ints("histograms/reference/marquette.txt", hReference);
    //  Copy the arrays to the gpu memory
    cudaMemcpy(dTest, hTest, SIZE*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dReference, hReference, SIZE*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dDist, &hDist, sizeof(int), cudaMemcpyHostToDevice);
    //  launch kernel
    EuclidDist<<< blkNum , threadsPerBlk >>>(dTest, dReference, dDist);
    cudaDeviceSynchronize();
    cudaMemcpy(&hDist, dDist, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Distance from marquette = %d \n", hDist);
    hDist = 0;


//  free mem
    cudaFree(dTest);
    cudaFree(dReference);
    cudaFree(dDist);

    free(hTest);
    free(hReference);

    return 0;    
}