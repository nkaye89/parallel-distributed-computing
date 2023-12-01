#include <stdio.h>
#include <openacc.h>
#define N 2000

void populate_array(int arr[], int multiple) {
	int i;
	for(i=0;i<N;i++) {
		arr[i] = (i+1)*multiple;
	}
}

void openacc_vector_sum(int a[], int b[], int c[]) {
	int i;
	#pragma acc parallel loop
	for(i=0;i<N;i++) {
		c[i] = a[i]+b[i];
	}
}

void openacc_vector_sum2(int a[], int b[], int c[]) {
	int i;
	#pragma acc data copyin(a[0:N], b[0:N]) copyout(c[0:N])
	{
		#pragma acc parallel loop
		for(i=0;i<N;i++) {
			c[i] = a[i]+b[i];
		}
	}
	
}

void print_array(int arr[]) {
	int i;
	for(i=0;i<10;i++) {
		printf("%d,",arr[i]);
	}
	printf("...");
	for(i=N-5;i<N;i++) {
		printf(",%d",arr[i]);
	}
	printf("\n");
}

int main(int argc, char* argv[]) {

	int a[N], b[N], c[N];
	populate_array(a, 4);
	populate_array(b, 9);
	
	cudaSetDevice(1);

	openacc_vector_sum(a,b,c);
	//openacc_vector_sum2(a,b,c);

	print_array(a);
	print_array(b);
	print_array(c);

	return 0;
}