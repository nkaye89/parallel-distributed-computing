#include<mpi.h>
#include<stdio.h>

// compile using:  mpicc test.c
// run using: mpirun -np 3 ./a.out
// -np flag is to specify how many MPI processes should be created

int main(int argc, char **argv)
{
 int rank;
 int numProcs;
 
 MPI_Init(&argc, &argv);
 
 MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // rank is process id
 
 MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
 
 if(rank == 2)
    printf("%d \n", rank + numProcs);
    
 /*
  1) What is the output after the following command is executed?
   ./a.out (dont run MPI programs like this)
ANSWER: mpirun -np 1 a.out

  2) What is the output after the following command is executed?
     mpirun -np 3 ./a.out
ANSWER: 5

  3) What is the output after the following command is executed?
     mpirun -np 2 ./a.out 
    Process id = 0, 1    
ANSWER: 0
 */
    
 MPI_Finalize();
 return 0;  
}

