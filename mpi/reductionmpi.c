#include<mpi.h>
#include<stdio.h>

//compile using: mpicc reduction.c
//fun using:
//

int main(int argc, char **argv)
{





	// round-robin work distribution
	int arr[10] = {1,2,3,4,5,6,7,8,9,10}
	
	int localSum = 0;
	int index;
	for(index = rank; index < 10; index = index+numProcs)
	{
		localSum = localSum + arr[index];
	}
	
	printf("localSum by rank %d = %d \n", rank, localSum);
	fflush(stdout);

	int globalSum = 0;
	int root = 0;
	//MPI_Reduce is collective function because all of the processes
	//
	MPI_Reduce(&localSum, &globalSum, 1, MPI_INT, MPI_SUM, root, MPI_COMM_WORLD);

	if(rank == root)
	{
		print("Sum of the array is %d \n", globalSum);
	}
