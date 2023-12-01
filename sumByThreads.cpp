// compile: g++ -std=c++11 -lpthread -o parSum sumByThreads.cpp

// run: ./parSum

#include<stdio.h>
#include<thread>

using namespace std;

int *partialSumArr = NULL;

/* Function to print an array */
void printArray(int arr[], int start, int end)
{
	for (int i = start; i < end; i++)
		printf("%d ", arr[i]);
	printf("\n");
}

int sequentialSum(int arr[], int arraySize)
{
   int sum = 0;
   for (int i = 0; i < arraySize; i++)
   {
      sum = sum + arr[i];
   }
   return sum;
}

void findSum(int arr[], int threadId, int arraySize, int numThreads)
{
    int i, j;
    int start = (threadId * arraySize)/numThreads;
    int end = ( ( threadId + 1) * arraySize)/numThreads;
    
    if( threadId == (numThreads - 1) )
       end = arraySize;
    
    printf("threadId = %d, arraySize = %d start = %d end = %d \n", threadId, arraySize, start, end);
    
    int sum = 0;
	for (i = start; i < end; i++) {
       sum += arr[i];
	}
	printf("Partial sum = %d \n", sum );

	partialSumArr[threadId] = sum;
}

void findSumRoundRobin(int arr[], int threadId, int arraySize, int numThreads)
{
    int i;

    printf("threadId = %d, arraySize = %d numThreads = %d\n", threadId, arraySize, numThreads);
    
    int sum = 0;
	for (i = threadId; i < arraySize; i += numThreads) {
       sum += arr[i];
	}
	printf("Partial sum = %d \n", sum );

	partialSumArr[threadId] = sum;
}

int main() 
{
  
 int numThreads;
 printf("Enter number of threads from main thread \n");
 scanf("%d", &numThreads);
 
 thread *threads = new thread[numThreads];
 
 /* This array is to store partial sums by the threads; 
    thread i stores its partial sum in the array at index i */
 partialSumArr = new int[numThreads];
 
 int arr[] = { 64, 34, 25, 12, 22, 11, 90, 10, 15, 19, 7, 6, 16 };
 //int arr[] = { 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4 };
 int arraySize = sizeof(arr) / sizeof(arr[0]);
 
 for(int id = 0; id < numThreads; id++)
 {
   // the first parameter is the name of the method that a thread should call 
   // the rest of the parameters like arr, id, .. are for the method findSum itselfs
   threads[id] = thread( findSumRoundRobin, arr, id, arraySize, numThreads);
 }
 
 /* main thread waits for children threads to finish if you omit the loop below, 
    then main thread can exit before children threads finish running the findSum method */
 for(int id = 0; id < numThreads; id++)
 {
   threads[id].join();
 }
 
 // main thread adds the partial sums to get the final result
 int finalSum = 0;
 for(int id = 0; id < numThreads; id++)
 {
   finalSum = finalSum + partialSumArr[id]; 
 }
 
 //printArray(arr, 0, arraySize);
 
 printf(" Parallel Sum by main thread = %d \n", finalSum);
 
 // for verification, let us do the same findSum but sequentially
 printf("Sequential Sum by main thread = %d \n", sequentialSum(arr, arraySize) );
 
 return 0;
}