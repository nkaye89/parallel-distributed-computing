#include<mpi.h>
#include<stdio.h>

#define MAX 10

// compile using:  mpicc ExerciseMPI.c
// run using: mpirun -np 10 ./a.out
// -np flag is to specify how many MPI processes should be created

int matrix[MAX][MAX];

void populateMatrix( int ROWS, int COLS ) 
{
   int i, j;
   for(i = 0; i < ROWS; i++)
   {
       for(j = 0; j < COLS; j++)
       {
         matrix[i][j] = i;
       }
   }
}

int main(int argc, char **argv)
{
 int rank;
 int numProcs;
 
 MPI_Init(&argc, &argv);
 MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // rank is process id
 MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

 int A[MAX] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
 
 // Write an MPI program that runs using 10 MPI processes and prints one number from the array
 printf("Process %d: ArrayNum: %d ", rank, A[rank]);
 fflush(stdout);

 populateMatrix(10, 10);
 
 // Write MPI code below so that each MPI process adds the elements in a row given by its rank/id
 // e.g., P0 adds the elements from 0th row
 //       P1 adds the elements from 1st row  and so on
 int col;
 int result = 0;
 for(col = 0; col < 10; col++) {
    result += matrix[rank][col];
 }
 printf("MatrixResult: %d\n", result);
 fflush(stdout);
 
 MPI_Finalize();
 return 0;  
}