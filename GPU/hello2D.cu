#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

//  2 blocks, 3 threads in each block
__global__ void HelloCuda1D()
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    printf("Local thread Id = %d Block id = %d Global thread Id = ( %d )\n",
        threadIdx.x, blockIdx.x, i);
}

__global__ void HelloCuda2D()
{
  int i = threadIdx.x; 
  int j = threadIdx.y;
  
  //if(blockIdx.x == 0 && blockIdx.y == 0)
    printf(" blockIdx.x = %d, blockIdy.y = %d, thread Id along x = %d along y = %d \n",
    															blockIdx.x, blockIdx.y, i, j);
}


/**
 * Host main routine
 */
int main(void)
{
  /* 
   HelloCuda1D<<<2,3>>>();
   cudaDeviceSynchronize();
   
   printf("1 done \n");
  */ 
    
   dim3 block(2,3);
   dim3 grid(2,2);
   
   HelloCuda2D<<<grid, block>>>();
   cudaDeviceSynchronize();
   
   printf("2 done \n");
  
}
