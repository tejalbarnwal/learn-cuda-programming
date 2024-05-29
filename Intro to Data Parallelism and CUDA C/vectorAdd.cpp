#include <stdio.h>

void vectorAdd(double* h_A, double* h_B, double* h_C, int n)
{
    for(int i=0; i<n; i++)
    {
        h_C[i] = h_A[i] + h_B[i];
    }
}

void printVector(double* h_C, int n)
{
    for(int i=0; i<n; i++)
    {
        printf("%f , ", h_C[i]);
    }
    printf("\n");
}

int main()
{
    printf("hello world\n");

    double h_A[] = {1.0, 2.0, 3.0};
    double h_B[] = {4.0, 5.0, 6.0};
    int n = sizeof(h_A)/sizeof(h_A[0]);
    // printf("%f", (double)n);

    double h_C[] = {0.0, 0.0, 0.0};

    vectorAdd(h_A, h_B, h_C, n);
    printVector(h_C, n);

    return 0;
}