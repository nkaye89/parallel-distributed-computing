// C++ program to implement
// g++ -o distance EuclideanDistance.cpp
// g++ -std=c++11 -lpthread -o distance EuclideanDistance.cpp
// ./distance

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <thread>

using namespace std;

const int SIZE = 255;

int *partialDistArr = NULL;

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

int euclidean_distance(int test[], int reference[], int SIZE)
{
  int counter;
  int distance = 0;
  for(counter = 0; counter<SIZE; counter++)
  {
     distance += (test[counter] - reference[counter]) * (test[counter] - reference[counter]);
  }
  return distance;
}

int euclidean_distance_static_partitioning(int test[], int reference[], int SIZE, int threadId, int numThreads)
{
  int counter;
  int start = (threadId * SIZE)/numThreads;
  int end = ( ( threadId + 1) * SIZE)/numThreads;
  
  if( threadId == (numThreads - 1) )
    end = SIZE;
  
  int distance = 0;

  for(counter = start; counter < end; counter++) {
    distance += (test[counter] - reference[counter]) * (test[counter] - reference[counter]);
  }

  partialDistArr[threadId] = distance;
}

int euclidean_distance_round_robin(int test[], int reference[], int SIZE, int threadId, int numThreads)
{
  int counter;
  
  int distance = 0;

  for(counter = threadId; counter < SIZE; counter += numThreads) {
    distance += (test[counter] - reference[counter]) * (test[counter] - reference[counter]);
  }

  partialDistArr[threadId] = distance;
}

// Driver code
int main()
{
  //get number of threads
  int numThreads;
  printf("Enter number of threads from main thread \n");
  scanf("%d", &numThreads);

  //create arrays
  partialDistArr = new int[numThreads];
  thread *threads = new thread[numThreads];

    int test[SIZE];
    int reference[SIZE];
	read_ints("histograms/test/frog1.txt", test);
	
  int dist1 = 0;


//  ----> TEST FOR ITSELF <----

	read_ints("histograms/test/frog1.txt", reference);
//SEQUENTIAL
  printf("\nSequential...\n");
	dist1 = euclidean_distance(test, reference, SIZE);
  printf("Distance from itself = %d \n", dist1);
  dist1 = 0;

//STATIC PARTITIONING
  printf("Static Partitioning...\n");
  //create threads
  for(int id = 0; id < numThreads; id++) {
    threads[id] = thread(euclidean_distance_static_partitioning, test, reference, SIZE, id, numThreads);
  }
  //join threads
  for(int id = 0; id < numThreads; id++) {
    threads[id].join();
  }
  //sum partialDistArr and clear it
  for(int num = 0; num < numThreads; num++) {
    dist1 += partialDistArr[num];
    partialDistArr[num] = 0;
  }
  printf("Distance from itself = %d \n", dist1);
  dist1 = 0;

//ROUND ROBIN
  printf("Round Robin...\n");
  //create threads
  for(int id = 0; id < numThreads; id++) {
    threads[id] = thread(euclidean_distance_round_robin, test, reference, SIZE, id, numThreads);
  }
  //join threads
  for(int id = 0; id < numThreads; id++) {
    threads[id].join();
  }
  //sum partialDistArr and clear it
  for(int num = 0; num < numThreads; num++) {
    dist1 += partialDistArr[num];
    partialDistArr[num] = 0;
  }
	printf("Distance from itself = %d \n", dist1);
  dist1 = 0;
	

//  ----> TEST FOR FROG2 <----

	read_ints("histograms/reference/frog2.txt", reference);
//SEQUENTIAL
  printf("\nSequential...\n");
	dist1 = euclidean_distance(test, reference, SIZE);
  printf("Distance from frog2 = %d \n", dist1);
  dist1 = 0;

//STATIC PARTITIONING
  printf("Partial Ordering...\n");
  //create threads
  for(int id = 0; id < numThreads; id++) {
    threads[id] = thread(euclidean_distance_static_partitioning, test, reference, SIZE, id, numThreads);
  }
  //join threads
  for(int id = 0; id < numThreads; id++) {
    threads[id].join();
  }
  //sum partialDistArr and clear it
  for(int num = 0; num < numThreads; num++) {
    dist1 += partialDistArr[num];
    partialDistArr[num] = 0;
  }
  printf("Distance from frog2 = %d \n", dist1);
  dist1 = 0;

//ROUND ROBIN
  printf("Round Robin...\n");
  //create threads
  for(int id = 0; id < numThreads; id++) {
    threads[id] = thread(euclidean_distance_round_robin, test, reference, SIZE, id, numThreads);
  }
  //join threads
  for(int id = 0; id < numThreads; id++) {
    threads[id].join();
  }
  //sum partialDistArr and clear it
  for(int num = 0; num < numThreads; num++) {
    dist1 += partialDistArr[num];
    partialDistArr[num] = 0;
  }
	printf("Distance from frog2 = %d \n", dist1);
  dist1 = 0;


//  ----> TEST FOR ROTATED FROG <----
	
	read_ints("histograms/reference/frogRotated.txt", reference);
//SEQUENTIAL
  printf("\nSequential...\n");
	dist1 = euclidean_distance(test, reference, SIZE);
  printf("Distance from rotated frog = %d \n", dist1);
  dist1 = 0;

//STATIC PARTITIONING
  printf("Static Partitioning...\n");
  //create threads
  for(int id = 0; id < numThreads; id++) {
    threads[id] = thread(euclidean_distance_static_partitioning, test, reference, SIZE, id, numThreads);
  }
  //join threads
  for(int id = 0; id < numThreads; id++) {
    threads[id].join();
  }
  //sum partialDistArr and clear it
  for(int num = 0; num < numThreads; num++) {
    dist1 += partialDistArr[num];
    //printf("\nThread %d: %d\n", num, partialDistArr[num]);
    partialDistArr[num] = 0;
  }
  printf("Distance from rotated frog = %d \n", dist1);
  dist1 = 0;

//ROUND ROBIN
  printf("Round Robin...\n");
  //create threads
  for(int id = 0; id < numThreads; id++) {
    threads[id] = thread(euclidean_distance_round_robin, test, reference, SIZE, id, numThreads);
  }
  //join threads
  for(int id = 0; id < numThreads; id++) {
    threads[id].join();
  }
  //sum partialDistArr and clear it
  for(int num = 0; num < numThreads; num++) {
    dist1 += partialDistArr[num];
    partialDistArr[num] = 0;
  }
	printf("Distance from rotated frog = %d \n", dist1);
  dist1 = 0;


//  ----> TEST FOR BUCK <----
		
	read_ints("histograms/reference/buck.txt", reference);
//SEQUENTIAL
  printf("\nSequential...\n");
	dist1 = euclidean_distance(test, reference, SIZE);
  printf("Distance from buck = %d \n", dist1);
  dist1 = 0;

//STATIC PARTITIONING
  printf("Static Partitioning...\n");
  //create threads
  for(int id = 0; id < numThreads; id++) {
    threads[id] = thread(euclidean_distance_static_partitioning, test, reference, SIZE, id, numThreads);
  }
  //join threads
  for(int id = 0; id < numThreads; id++) {
    threads[id].join();
  }
  //sum partialDistArr and clear it
  for(int num = 0; num < numThreads; num++) {
    dist1 += partialDistArr[num];
    partialDistArr[num] = 0;
  }
  printf("Distance from buck = %d \n", dist1);
  dist1 = 0;

//ROUND ROBIN
  printf("Round Robin...\n");
  //create threads
  for(int id = 0; id < numThreads; id++) {
    threads[id] = thread(euclidean_distance_round_robin, test, reference, SIZE, id, numThreads);
  }
  //join threads
  for(int id = 0; id < numThreads; id++) {
    threads[id].join();
  }
  //sum partialDistArr and clear it
  for(int num = 0; num < numThreads; num++) {
    dist1 += partialDistArr[num];
    partialDistArr[num] = 0;
  }
	printf("Distance from buck = %d \n", dist1);
  dist1 = 0;


//  ----> TEST FOR MARQUETTE <----

  read_ints("histograms/reference/marquette.txt", reference);
//SEQUENTIAL
  printf("\nSequential...\n");
	dist1 = euclidean_distance(test, reference, SIZE);
  printf("Distance from marquette = %d \n", dist1);
  dist1 = 0;

//STATIC PARTITIONING
  printf("Static Partitioning...\n");
  //create threads
  for(int id = 0; id < numThreads; id++) {
    threads[id] = thread(euclidean_distance_static_partitioning, test, reference, SIZE, id, numThreads);
  }
  //join threads
  for(int id = 0; id < numThreads; id++) {
    threads[id].join();
  }
  //sum partialDistArr and clear it
  for(int num = 0; num < numThreads; num++) {
    dist1 += partialDistArr[num];
    partialDistArr[num] = 0;
  }
  printf("Distance from marquette = %d \n", dist1);
  dist1 = 0;

//ROUND ROBIN
  printf("Round Robin...\n");
  //create threads
  for(int id = 0; id < numThreads; id++) {
    threads[id] = thread(euclidean_distance_round_robin, test, reference, SIZE, id, numThreads);
  }
  //join threads
  for(int id = 0; id < numThreads; id++) {
    threads[id].join();
  }
  //sum partialDistArr and clear it
  for(int num = 0; num < numThreads; num++) {
    dist1 += partialDistArr[num];
    partialDistArr[num] = 0;
  }
	printf("Distance from marquette = %d \n", dist1);
  dist1 = 0;
	
	return 0;
}
