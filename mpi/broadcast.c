#include<mpi.h>
#include<stdio.h>

#define MAX 10

// compile using:  mpicc ExerciseMPI.c
// run using: mpirun -np 4 ./a.out
// -np flag is to specify how many MPI processes should be created

int main(int argc, char **argv)
{
 int rank;
 int numProcs;
 
 MPI_Init(&argc, &argv);
 MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // rank is process id
 MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

 int A[MAX];
 if(rank == 0)
 {
    int i;
    for( i = 0; i<MAX; i++ )
    {
        A[i] = i;
    }
 }
 
// int MPI_Bcast( void *buffer, int count, MPI_Datatype datatype, int root, 
//               MPI_Comm comm )
    MPI_Bcast( A, 10, MPI_INT, 0, MPI_COMM_WORLD );
  
  if(rank == 2)
  {
    int i;
    for( i = 0; i<MAX; i++ )
    {
        printf(", %d, ", A[i]);
    }
  } 

  MPI_Finalize();
  return 0;  
}