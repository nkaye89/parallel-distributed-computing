#include<stdio.h>
#include<stdlib.h>
#include<mpi.h>

#define MAX 500
#define COEFFICIENT 1


double power(double x, int degree)
{     
      if(degree == 0)  return 1;
      
      if(degree == 1)  return x;

      return x * power(x, degree - 1);
}

double sequential(int coeffArr[], double x)
{
   int maxDegree = MAX - 1;
   int i;
   double  answer = 0;
   
   for( i = 0; i < maxDegree;  i++)
   {
      
      double powerX = power(x, i);

      //printf("%f ", powerX);
      answer = answer + coeffArr[i] * powerX;
   }
   return answer;
}

void initialize(int coeffArr[])
{
   int maxDegree = MAX - 1;
   int i;
   for( i = 0; i < maxDegree; i++)
   {
      coeffArr[i] = COEFFICIENT;
   }
}

// Driver Code
int main(int argc, char **argv)
{
    int *coeffArr = (int *)malloc(sizeof(int) * MAX);
    
    initialize(coeffArr);
    double x = 0.99;
    int rank, numProcs;
    
    MPI_Init (&argc, &argv);
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // rank is process id
 
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    
    /* Start timer */
    double elapsed_time = - MPI_Wtime();
    
    double value = sequential(coeffArr, x);
    
    /* End timer */
    elapsed_time = elapsed_time + MPI_Wtime();
        
    if(rank == 0)    
    {
     printf(" sequential value %f wall clock time %8.6f \n", value, elapsed_time);
     fflush(stdout);
    }
    
    double partialResult = 0.0;
    
    int maxDegree = MAX - 1;
    int i;
    for( i = rank; i < maxDegree;  i= i+numProcs)
    {
      
      double powerX = power(x, i);

      //printf("%f ", powerX);
      partialResult = partialResult + coeffArr[i] * powerX;
    }
    printf("Rank = %d and partialResult = %f \n", rank, partialResult);
    fflush(stdout);
    free(coeffArr);

    //send to master process
    int tag = 100;
    MPI_Status status;
    double finalResult = partialResult;
    if(rank != 0) {
        MPI_Send(&partialResult, 1, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);
    }else { //receive
        int i;
        for(i = 1; i < numProcs; i++) {
            double partial;
            MPI_Recv(&partial, 1, MPI_DOUBLE, MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &status);
            finalResult += partial;
        }
        printf("Rank = %d and master final answer = %f \n", rank, finalResult);
    }
    
    MPI_Finalize();
    
	return 0;
}
