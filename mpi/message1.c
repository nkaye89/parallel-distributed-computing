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
 int destination = 1; int tag = 100;
 
 // receive side
  int source = 0; 
 
//  double doubleData = 3.142123;
//  double doubleRecv;  
 
 
 if(rank == 0)
 {
   double stockPrice[5] = {2202.1, 3300.0, 232.12, 65.12, 0};
   //MPI_Send(stockPrice, 5, MPI_DOUBLE, destination, tag, MPI_COMM_WORLD);
   //int sendCode = 123;
   //MPI_Send( &sendCode, 1, MPI_INT, destination, tag, MPI_COMM_WORLD);
   int dest;
   
   for(dest = 1; dest < numProcs; dest++) {
    MPI_Send(stockPrice, 5, MPI_DOUBLE, destination, tag, MPI_COMM_WORLD);
   }
 }
 else 
 {
   //int receivedCode;       
   MPI_Status status;
   //MPI_Recv(&receivedCode, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
   //printf("Received code is %d \n", receivedCode);
   double receivedStock[5];
   MPI_Recv(receivedStock, 5, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD, &status);  
   printf("Received  %f \n", receivedStock[1]);
 }
   
 MPI_Finalize();
 return 0;  
}