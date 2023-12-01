// compile: g++ -std=c++11 -lpthread -o hello threadProgram.cpp

// run: ./hello

#include<thread>
#include<iostream>

using namespace std;

void hello( int n )
{
  printf("Hello, World #%d \n", n);
}

void bye( int n )
{
  printf("Bye, World #%d \n", n);
}

int main() 
{
   thread first(hello, 1);
  
   thread second(hello, 2);

   thread third(hello, 3);
 
   thread fourth(bye, 4);

   first.join();
  
   second.join();
   
   third.join();

   fourth.join();

   return 0;
}
