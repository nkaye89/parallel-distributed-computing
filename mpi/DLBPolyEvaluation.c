#include<stdio.h>
#include<stdlib.h>
#include<mpi.h>

#define MAX 50003
//#define MAX 1000
#define COEFFICIENT 1

//make number double the amount of terms you want
#define TERMS 20

// Use Pascal server for this assignment. Use Everest only when you can not access Pascal.
// mpicc DLBPolyEvaluation.c
// mpirun -np 3 ./a.out

// Two tags to distinguish work messages from process-termination messages
#define WORKTAG 1
#define DIETAG 2

double power(double x, int degree)
{     
      if(degree == 0)  return 1;
      
      if(degree == 1)  return x;

      return x * power(x, degree - 1);
}

// to be called from worker process
double evaluateTerm(int coefficient, int degree, double x)
{
      double powerX = power(x, degree);

      double answer = 0;
      answer = answer + coefficient * powerX;
      
      return answer;
}

double sequential(int coeffArr[], double x)
{
   int i;
   double  answer = 0;
   
   // updated the stopping value of loop to MAX-1 
   for( i = 0; i < MAX;  i++)
   {
      double powerX = power(x, i);

      //printf("%f ", powerX);
      answer = answer + coeffArr[i] * powerX;
   }
   return answer;
}

void initialize(int coeffArr[])
{
   int i;
   for( i = 0; i < MAX; i++)
   {
      coeffArr[i] = COEFFICIENT;
   }
}

void master(int coeffArr[], int numTerms)
{
      int numProcs;
	  MPI_Comm_size(MPI_COMM_WORLD, &numProcs); 
	
	  MPI_Status status;
	
	  // MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)
      int tCount = 0;
      int workerRank;
      //int term[2];             // FOR ONE TERM
      int term[TERMS];         // FOR MULTIPLE TERMS
      
      for(workerRank = 1; workerRank < numProcs; workerRank++)
      {   
         /*    UNCOMMENT FOR ONE TERM
         term[0] = tCount;  // degree
         term[1] = coeffArr[tCount];  // coefficient
         MPI_Send(term, 2, MPI_INT, workerRank, WORKTAG, MPI_COMM_WORLD); 
         tCount++;
         // */

         //*    UNCOMMENT FOR MULTIPLE TERMS
         int i;
         double answer = 0;
         //printf("master is filling term array for worker %d\n", workerRank);
         for(i = 1; i < TERMS; i+=2) {
            term[i-1] = tCount;
            term[i] = coeffArr[tCount];
            tCount++;
         }
         MPI_Send(term, TERMS, MPI_INT, workerRank, WORKTAG, MPI_COMM_WORLD); 
         //printf("master sent terms to worker %d\n", workerRank);
         // */
      }
      // receive the work done by workers
      double termAnswer;
      double finalAnswer = 0;
      
      while(tCount < MAX)
      {    
          // Receive the results from workers and accumulate the result
          MPI_Recv(&termAnswer, 1, MPI_DOUBLE, MPI_ANY_SOURCE, WORKTAG, MPI_COMM_WORLD, &status);
          //printf("master received answer from a worker\n");
          finalAnswer = finalAnswer + termAnswer;
          workerRank = status.MPI_SOURCE;

          /*   UNCOMMENT FOR ONE TERM
          term[0] = tCount;  // degree
          term[1] = coeffArr[tCount];  // coefficient
          MPI_Send(term, 2, MPI_INT, workerRank, WORKTAG, MPI_COMM_WORLD); 
          tCount++;
          //   */

          //*    UNCOMMENT FOR MULTIPLE TERMS
          int i;
          double answer = 0;
          for(i = 1; i < TERMS; i+=2) {
             if(tCount >= MAX) {
               term[i-1] = 0;
               term[i] = 0;
             }else {
               term[i-1] = tCount;
               term[i] = coeffArr[tCount];
               tCount++;
             }
          }
          MPI_Send(term, TERMS, MPI_INT, workerRank, WORKTAG, MPI_COMM_WORLD); 
          // */
      }
      	
      // Send (tag = DIETAG) to all workers
      for(workerRank = 1; workerRank < numProcs; workerRank++)
      {
        // sending garbage value because it will be ignored at worker
        /*    UNCOMMENT FOR ONE TERM
        term[0] = -111;  term[1] = -111;
        //  */

        //*    UNCOMMENT FOR MULTIPLE TERMS
        int i;
        for(i = 0; i < TERMS; i++) {
             term[i] = -111;
        }
        //  */

        //MPI_Send(term, 2, MPI_INT, workerRank, DIETAG, MPI_COMM_WORLD);          // FOR ONE TERM
        MPI_Send(term, TERMS, MPI_INT, workerRank, DIETAG, MPI_COMM_WORLD);      // FOR MULTIPLE TERMS
      }
     
      // Do pending receives for outstanding messages from workers
      for(workerRank = 1; workerRank < numProcs; workerRank++)
      {
        MPI_Recv(&termAnswer, 1, MPI_DOUBLE, workerRank, WORKTAG, MPI_COMM_WORLD, &status);
        finalAnswer = finalAnswer + termAnswer;
      }
      printf("Master summed up to %f", finalAnswer);
      fflush(stdout);
     // MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status)
}
  
void worker(double x)
{
	MPI_Status status;
    int workerRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &workerRank);
    int val; 
        
    // worker keeps looping; waiting for work items from master process, 
    // unless it gets termination message
    while(1)
    { 
       //*      UNCOMMENT FOR MULTIPLE TERMS
       int term[(TERMS)];
       MPI_Recv(term, TERMS, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
       //printf("worker %d received terms\n", workerRank);
       //   */

       /*      UNCOMMENT FOR ONE TERM
       int term[2];
       MPI_Recv(term, 2, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
       //   */

       if(status.MPI_TAG == DIETAG)
       {
           //printf("TERMINATING. BYE \n");
           fflush(stdout);
           return;
       }
       else    // (status.MPI_TAG = WORKTAG)
       {
         //*    UNCOMMENT FOR MULTIPLE TERMS
         int i;
         double answer = 0;
         //printf("worker %d is evaluating terms\n", workerRank);
         for(i = 1; i < (TERMS); i+=2) {
            //printf("worker %d: i = %d\n", workerRank, i);
            answer += evaluateTerm(term[i], term[i-1], x);
            //printf("worker %d: answer = %f, term: %d\n", workerRank, answer, i);
         }
         //printf("worker %d finished evaluating terms\n", workerRank);
         // */

         /*    UNCOMMENT FOR ONE TERM
         // evaluateTerm(int coefficient, int degree, double x)
         double answer = evaluateTerm(term[1], term[0], x);
         // */

         //    THIS STAYS SAME
         //printf("worker %d attempting to send answer\n", workerRank);
         MPI_Send(&answer, 1, MPI_DOUBLE, 0, WORKTAG, MPI_COMM_WORLD);
         //printf("worker %d sent answer to master\n", workerRank);
         //printf(" degree %d coeff %d value %0.12f \n", term[0], term[1], answer);
         fflush(stdout);
       }
    } 
}

int main1()
{
    int *coeffArr = (int *)malloc(sizeof(int) * MAX);
    double x = 0.99;
    //double x = 1.000000000001;
    
    initialize(coeffArr);
    double value = sequential(coeffArr, x);
    printf(" sequential value %f \n", value);
    free(coeffArr);
    
    return 0;
} 

// Driver Code
int main(int argc, char **argv)
{ 
    int *coeffArr = (int *)malloc(sizeof(int) * MAX);
    double x = 0.99;
    //double x = 1.000000000001;
    
    initialize(coeffArr);
    
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   //*     
    if(rank == 0)
    {
       double sequentialAnswer = sequential(coeffArr, x);
       printf("squential Answer = %f \n", sequentialAnswer);
    }
   // */ 

    if (rank == 0) 
    {
       master(coeffArr, MAX);
    } 
    else 
    {
       worker(x);
    }
    
    free(coeffArr);
    
    MPI_Finalize();
    
	return 0;
}

/*
    sequential code measurement

    // Start timer
    double elapsed_time = - MPI_Wtime();
    
    double value = sequential(coeffArr, x);
    
    // End timer 
    elapsed_time = elapsed_time + MPI_Wtime();
    
    printf(" sequential value %f wall clock time %8.6f \n", value, elapsed_time);
    fflush(stdout);

*/
