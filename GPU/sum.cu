#include <stdio.h>
#define N 70
//  N = 100
//  4 blocks * 24 threads = 96
//  5 blocks * 24 threads = 120
// 100 items in the array but 120 threads
__global__ void add(int *a, int *b, int *c)
{	
	int id;
	cudaGetDevice(&id);
	printf("%d\n", id);
    /* insert code to calculate the index properly using blockIdx.x, blockDim.x, threadIdx.x */
	int index = blockIdx.x * blockDim.x + threadIdx.x;
  // index = 100, 101, 102,.. 119
  if(index < N)
  	 c[index] = a[index] + b[index];
}

/* experiment with N */
/* how large can it be? */
//#define N (2048*2048)
//#define N 70
#define THREADS_PER_BLOCK 8

int main()
{
	cudaSetDevice(1);
  int *a, *b, *c; // Arrays in CPU
	int *d_a, *d_b, *d_c;  // Arrays to be allocated in a GPU
	int size = N * sizeof( int );

	/* allocate space for device copies of a, b, c */

	cudaMalloc( (void **) &d_a, size );
	cudaMalloc( (void **) &d_b, size );
	cudaMalloc( (void **) &d_c, size );

	/* allocate space for host copies of a, b, c and setup input values */

	a = (int *)malloc( size );
	b = (int *)malloc( size );
	c = (int *)malloc( size );

	for( int i = 0; i < N; i++ )
	{
		a[i] = b[i] = i;
		c[i] = 0;
	}

	/* copy inputs to device */
	/* fix the parameters needed to copy data to the device */
	cudaMemcpy( d_a, a, size, cudaMemcpyHostToDevice );
	cudaMemcpy( d_b, b, size, cudaMemcpyHostToDevice );

	/* launch the kernel on the GPU */
	/* insert the launch parameters to launch the kernel properly using blocks and threads */ 
	int id;
	cudaGetDevice(&id);
	printf("%d\n", id);
	add<<< (N + (THREADS_PER_BLOCK-1)) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>( d_a, d_b, d_c );
  //  10/3 = 3 blocks are not enough= 3*3 = 9
  //  ceiling   = 4 blocks are created = (4*3) = 12 threads

	/* copy result back to host */
	/* fix the parameters needed to copy data back to the host */
	cudaMemcpy( c, d_c, size, cudaMemcpyDeviceToHost );

  int index = 0; 
	printf( "c[%d] = %d\n",index+0, c[0] );
	printf( "c[%d] = %d\n",index+1, c[1] );
	printf( "c[%d] = %d\n",index+2, c[2] );
	printf( "c[%d] = %d\n",index+3, c[3] );
	printf( "c[%d] = %d\n",index+4, c[4] );

	printf( "c[%d] = %d\n",N-1, c[N-1] );

	/* clean up */

	free(a);
	free(b);
	free(c);
	cudaFree( d_a );
	cudaFree( d_b );
	cudaFree( d_c );
	
	return 0;
} /* end main */

