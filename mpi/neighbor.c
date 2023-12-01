//RING COMMUNICATION

MPI_Send(, destination = (rank+1)%nprocs, );

if(rank = 0) {
    MPI_Recv(, source = nprocs-1,);
}else {
    MPI_Recv(, source = rank - 1,);
}