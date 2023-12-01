// C++ program to implement
// g++ -o distance EuclideanDistance.cpp
// ./distance

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <sys/time.h>

const int SIZE = 255;

void read_ints (const char* file_name, double arr[])
{
  FILE* file = fopen (file_name, "r");
  
  if (NULL == file) 
  {
        printf("file can't be opened \n");
        exit(1);
  }
  double i = 0;
  int counter = 0;
  fscanf(file, "%lf", &i);    
  while (!feof (file))
    {  
      //printf ("%.2f ", i);
      arr[counter] = i;
      fscanf (file, "%lf", &i);  
      counter++;    
    }
  fclose (file);        
}

double euclidean_distance(double *test, double *reference, int SIZE)
{
  int counter;
  __m256d dSIMD = _mm256_set_pd( 0, 0, 0, 0 );
  __m256d neg = _mm256_set_pd( -1, -1, -1, -1 );
  double distance = 0;
  for(counter = 0; counter<SIZE; counter+=4)
  {
     __m256d testVec = _mm256_load_pd( &test[counter] );
     __m256d refVec = _mm256_load_pd( &reference[counter] );
     __m256d negRefVec = _mm256_mul_pd( refVec, neg);
     __m256d tempVec = _mm256_add_pd( testVec, negRefVec );
     tempVec = _mm256_mul_pd(tempVec, tempVec);
     dSIMD = _mm256_add_pd(dSIMD, tempVec);

     //distance += (test[counter] - reference[counter]) * (test[counter] - reference[counter]);
  }

  double *dVec = (double*)_mm_malloc( 4 * sizeof(double), 32);
  _mm256_store_pd( &dVec[0], dSIMD );
  //printf("dvec[0] = %f, dvec[1] = %f, dvec[2] = %f, dvec[3] = %f\n", dVec[0], dVec[1], dVec[2], dVec[3]);
  distance = dVec[0]+dVec[1]+dVec[2]+dVec[3];
  
  return distance;
}

// Driver code
int main()
{
    //double test[SIZE];
    double *test = (double*)_mm_malloc( SIZE * sizeof(double), 32);
    //double reference[SIZE];
    double *reference = (double*)_mm_malloc( SIZE * sizeof(double), 32);
	read_ints("histograms/test/frog1.txt", test);
	
	read_ints("histograms/test/frog1.txt", reference);
	double dist1 = euclidean_distance(test, reference, SIZE);
	printf("Distance from itself = %f \n", dist1);
	
	read_ints("histograms/reference/frog2.txt", reference);
    dist1 = euclidean_distance(test, reference, SIZE);
	printf("Distance from frog2 = %f \n", dist1);
	
	read_ints("histograms/reference/frogRotated.txt", reference);
	dist1 = euclidean_distance(test, reference, SIZE);
	printf("Distance from rotated frog = %f \n", dist1);
		
	read_ints("histograms/reference/buck.txt", reference);
	dist1 = euclidean_distance(test, reference, SIZE);
	printf("Distance from buck = %f \n", dist1);

    read_ints("histograms/reference/marquette.txt", reference);
	dist1 = euclidean_distance(test, reference, SIZE);
	printf("Distance from marquette = %f \n", dist1);

	return 0;
}