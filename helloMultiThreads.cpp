// compile: g++ -std=c++11 -lpthread -o hello helloMultiThreads.cpp

// run: ./hello

#include<thread>
#include<iostream>

using namespace std;

void hello (int n)
{
  printf("Hello, World #%d \n", n);
}

int main() 
{
  
 int numThreads;
 printf("Enter number of threads from main thread \n");
 scanf("%d", &numThreads);
 
 thread *threads = new thread[numThreads];
 for(int i = 0; i<numThreads; i++)
 {
   threads[i] = thread(hello, i);
 }
 
 for(int i = 0; i<numThreads; i++)
 {
   threads[i].join();
 }
 
 printf("Goodbye, World from main thread!\n");
 
 return 0;
}