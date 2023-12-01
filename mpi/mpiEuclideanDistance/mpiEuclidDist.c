#include<mpi.h>
#include<stdio.h>
#include <stdlib.h>
#include <string.h>

//compile using mpicc mpiEuclidDist.c
// run using: mpirun -np 4 ./a.out
// -np flag is to specify how many MPI processes should be created

const int SIZE = 255;

void read_ints (const char* file_name, int arr[])
{
  FILE* file = fopen (file_name, "r");
  
  if (NULL == file) 
  {
        printf("file can't be opened \n");
        exit(1);
  }
  int i = 0;
  int counter = 0;
  fscanf (file, "%d", &i);    
  while (!feof (file))
    {  
      //printf ("%d ", i);
      arr[counter] = i;
      fscanf (file, "%d", &i);  
      counter++;    
    }
  fclose (file);        
}

int main(int argc, char **argv)
{
    int rank;
    int numProcs;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // rank is process id
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    int test[SIZE];
    int reference[SIZE];
	read_ints("histograms/test/frog1.txt", test);


//  ----> TEST FOR ITSELF <----
    read_ints("histograms/test/frog1.txt", reference);
    int frog1 = 101;

    int partialDistance = 0;
    int i;
    for(i = rank; i < SIZE; i += numProcs) {
        partialDistance += (test[i] - reference[i]) * (test[i] - reference[i]);
    }
    printf("rank: %d\tpartialDistance = %d\n", rank, partialDistance);
    fflush(stdout);
    
    if(rank != 0) {
        MPI_Send(&partialDistance, 1, MPI_INT, 0, frog1, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }else {
        int finalDistance = partialDistance;
        int partial;
        MPI_Barrier(MPI_COMM_WORLD);
        for(i = 1; i < numProcs; i++) {
            MPI_Recv(&partial, 1, MPI_DOUBLE, MPI_ANY_SOURCE, frog1, MPI_COMM_WORLD, &status);
            finalDistance += partial;
        }
        printf("Rank: %d Final Distance From Itself = %d\n\n", rank, finalDistance);
        fflush(stdout);
    }


//  ----> TEST FOR FROG2 <----
	read_ints("histograms/reference/frog2.txt", reference);
    int frog2 = 102;

    partialDistance = 0;
    for(i = rank; i < SIZE; i += numProcs) {
        partialDistance += (test[i] - reference[i]) * (test[i] - reference[i]);
    }
    printf("rank: %d\tpartialDistance = %d\n", rank, partialDistance);
    fflush(stdout);
    
    if(rank != 0) {
        MPI_Send(&partialDistance, 1, MPI_INT, 0, frog2, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }else {
        int finalDistance = partialDistance;
        int partial;
        MPI_Barrier(MPI_COMM_WORLD);
        for(i = 1; i < numProcs; i++) {
            MPI_Recv(&partial, 1, MPI_DOUBLE, MPI_ANY_SOURCE, frog2, MPI_COMM_WORLD, &status);
            finalDistance += partial;
        }
        printf("Rank: %d Final Distance From Frog2 = %d\n\n", rank, finalDistance);
        fflush(stdout);
    }


//  ----> TEST FOR ROTATED FROG <----
	read_ints("histograms/reference/frogRotated.txt", reference);
    int rotatedFrog = 103;

    partialDistance = 0;
    for(i = rank; i < SIZE; i += numProcs) {
        partialDistance += (test[i] - reference[i]) * (test[i] - reference[i]);
    }
    printf("rank: %d\tpartialDistance = %d\n", rank, partialDistance);
    fflush(stdout);
    
    if(rank != 0) {
        MPI_Send(&partialDistance, 1, MPI_INT, 0, rotatedFrog, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }else {
        int finalDistance = partialDistance;
        int partial;
        MPI_Barrier(MPI_COMM_WORLD);
        for(i = 1; i < numProcs; i++) {
            MPI_Recv(&partial, 1, MPI_DOUBLE, MPI_ANY_SOURCE, rotatedFrog, MPI_COMM_WORLD, &status);
            finalDistance += partial;
        }
        printf("Rank: %d Final Distance From Rotated Frog = %d\n\n", rank, finalDistance);
        fflush(stdout);
    }


//  ----> TEST FOR BUCK <----	
	read_ints("histograms/reference/buck.txt", reference);
    int buck = 104;

    partialDistance = 0;
    for(i = rank; i < SIZE; i += numProcs) {
        partialDistance += (test[i] - reference[i]) * (test[i] - reference[i]);
    }
    printf("rank: %d\tpartialDistance = %d\n", rank, partialDistance);
    fflush(stdout);
    
    if(rank != 0) {
        MPI_Send(&partialDistance, 1, MPI_INT, 0, buck, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }else {
        int finalDistance = partialDistance;
        int partial;
        MPI_Barrier(MPI_COMM_WORLD);
        for(i = 1; i < numProcs; i++) {
            MPI_Recv(&partial, 1, MPI_DOUBLE, MPI_ANY_SOURCE, buck, MPI_COMM_WORLD, &status);
            finalDistance += partial;
        }
        printf("Rank: %d Final Distance From Buck = %d\n\n", rank, finalDistance);
        fflush(stdout);
    }



//  ----> TEST FOR MARQUETTE <----
    read_ints("histograms/reference/marquette.txt", reference);
    int marquette = 105;

    partialDistance = 0;
    for(i = rank; i < SIZE; i += numProcs) {
        partialDistance += (test[i] - reference[i]) * (test[i] - reference[i]);
    }
    printf("rank: %d\tpartialDistance = %d\n", rank, partialDistance);
    fflush(stdout);
    
    if(rank != 0) {
        MPI_Send(&partialDistance, 1, MPI_INT, 0, marquette, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }else {
        int finalDistance = partialDistance;
        int partial;
        MPI_Barrier(MPI_COMM_WORLD);
        for(i = 1; i < numProcs; i++) {
            MPI_Recv(&partial, 1, MPI_DOUBLE, MPI_ANY_SOURCE, marquette, MPI_COMM_WORLD, &status);
            finalDistance += partial;
        }
        printf("Rank: %d Final Distance From Marquette = %d\n\n", rank, finalDistance);
        fflush(stdout);
    }


    MPI_Finalize();
    return 0;
}