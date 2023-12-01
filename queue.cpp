// CPP code to illustrate
// Queue in Standard Template Library (STL)
#include <iostream>
#include <queue>

#define MAX 50000
#define COEFFICIENT 1

using namespace std;

class Term
{
  public:  
   int coefficient;
   int degree;
  
   Term(int coef, int deg)
   {
    coefficient = coef;
    degree = deg;
   }
};

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


// Print the queue
void showq(queue<Term> gq)
{
	queue<Term> g = gq;
	while (!g.empty()) {
       Term t = g.front();
       cout<<t.coefficient<<" "<<t.degree<<endl;
	   g.pop();
	}
	cout << '\n';
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
int main()
{
    int coeffArr[MAX]; 
 
    initialize(coeffArr);
    double x = 0.99;
    
	queue<Term> gquiz;
	Term *t = new Term(3,2);
	gquiz.push(*t);
	
	Term *t1 = new Term(13,12);
	gquiz.push(*t1);
	
	cout << "The queue gquiz is : ";
	showq(gquiz);

	cout << "\ngquiz.size() : " << gquiz.size();

 	Term firstTermInQ = gquiz.front();
 	cout << "\ngquiz.front() : " << firstTermInQ.coefficient<<" "<<firstTermInQ.degree<<endl;

	return 0;
}
