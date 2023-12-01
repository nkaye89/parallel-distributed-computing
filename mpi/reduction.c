#include<mpi.h>
#include<stdio.h>

// compile using:  mpicc message.c
// run using: mpirun -np 2 ./a.out
// -np flag is to specify how many MPI processes should be created

int main(int argc, char **argv)
{
 int rank;
 int numProcs;
 
 MPI_Init(&argc, &argv);
 
 MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // rank is process id
 
 MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
 
 // send side
 int sendCode = 123;  int destination = 1; int tag = 100;
 
 // receive side
 int receivedCode; MPI_Status status;
 
 
 // round-robin work distribution
 int arr[10] = {1,2,3,4,5,6,7,8,9,10};
 
 int index; int localSum = 0;
 for(index = rank; index < 10; index = index + numProcs )
 {
    localSum = localSum + arr[index];
 }
 printf("localSum by rank %d = %d \n", rank, localSum);
 fflush(stdout);

 int globalSum = 0; int root = 0;
 // MPI_Reduce is collective function because all of the processes
 // participate in calculating the result.
 MPI_Reduce(&localSum, &globalSum, 1, MPI_INT, MPI_SUM, root, MPI_COMM_WORLD);
 
 if(rank == root)
  printf("Sum of the array is %d \n", globalSum);
 
 MPI_Finalize();
 return 0;  
}
