#include <mpi.h>
#include <stdio.h>
#include <unistd.h>

int main(int argc, char **argv)
{
	int rank;
	int numProcesses;
	
	char hostname[256];
	
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
	
	gethostname(hostname, 255);
	//print only odd processes:
	if(rank %2 != 0)
	{
		printf("Hello World! I am process number %d on host %s\n ",rank, hostname);
	}

	MPI_Finalize();

	return 0;
}
