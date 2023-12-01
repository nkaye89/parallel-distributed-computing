#include<stdio.h>
#include<stdlib.h>
#include<mpi.h>

#define MAX 50000
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
    
    MPI_Init (&argc, &argv);
    
    /* Start timer */
    double elapsed_time = - MPI_Wtime();
    
    double value = sequential(coeffArr, x);
    
    /* End timer */
    elapsed_time = elapsed_time + MPI_Wtime();
        
    printf(" sequential value %f wall clock time %8.6f \n", value, elapsed_time);
    fflush(stdout);
    
    free(coeffArr);
    
    MPI_Finalize();
    
	return 0;
}
