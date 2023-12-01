#include<mpi.h>
#include<stdio.h>

#define MAX 10

int globalSum = 0;
// mpirun -np 3 ./a.out

int main(int argc, char **argv)
{
 int rank;
 int numProcs;
 
 MPI_Init(&argc, &argv);
 MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // rank is process id
 MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

 int arr[MAX] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
 
 int i;
 int localSum = 0;
   
 int startIndex = (rank * MAX)/numProcs;
 int endIndex = ((rank + 1) * MAX)/numProcs;

 // printf("startIndex = %d endIndex = %d \n", startIndex, endIndex);
 // fflush(stdout);
 
 if(rank == (numProcs - 1))
     endIndex = MAX;
   
 for( i = startIndex; i < endIndex; i++)
 {
    localSum = localSum + arr[i];     
 }
 
 // globalSum = globalSum + localSum;
 MPI_Reduce(&localSum, &globalSum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

 if(rank == 0) 
 {
  printf("globalSum [rank %d] = %d \n", rank, globalSum);
  fflush(stdout);
 }
 MPI_Finalize();
 return 0;  
}