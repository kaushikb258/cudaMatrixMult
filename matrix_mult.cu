#include <iostream>
#include <stdio.h>
#include <math.h>

using namespace std;

#define TILE_WIDTH 2

__global__ void MatrixMult(int m, int n, int k, float *a, float *b, float *c)
{

 int row = threadIdx.y + blockIdx.y*blockDim.y;  
 int col = threadIdx.x + blockIdx.x*blockDim.x;  
 
 if((row < m) && (col < k))
 {
  float temp = 0.0;
  for (int i = 0; i < n; ++i)
  {
   temp += a[row*n+i]*b[col+i*k];
  }
  c[row*k+col] = temp; 
 }

}


// main fn
int main(void)
{
 
 int m = 4;
 int n = 6;
 int k = 7;
   
 float* a = new float[m*n];
 float* b = new float[n*k];
 float* c = new float[m*k];
 float *dev_a, *dev_b, *dev_c;
 
 dim3 dimGrid((k-1)/TILE_WIDTH+1,(m-1)/TILE_WIDTH+1,1);
 dim3 dimBlock(TILE_WIDTH,TILE_WIDTH,1);


 cudaMalloc((void**)&dev_a, m*n*sizeof(float));
 cudaMalloc((void**)&dev_b, n*k*sizeof(float));
 cudaMalloc((void**)&dev_c, m*k*sizeof(float));

 for (int i=0; i<m*n; i++)
 {
  a[i] = sin((float) i);
 }
 
 for (int i=0; i<n*k; i++)
 {
  b[i] = cos((float) i);
 }


 cudaMemcpy(dev_a, a, m*n*sizeof(float), cudaMemcpyHostToDevice);
 cudaMemcpy(dev_b, b, n*k*sizeof(float), cudaMemcpyHostToDevice);

 MatrixMult<<<dimGrid,dimBlock>>>(m,n,k,dev_a,dev_b,dev_c);
 
 cudaMemcpy(c, dev_c, m*k*sizeof(float), cudaMemcpyDeviceToHost);


 cout<<"a matrix: \n";
 for (int i=0; i<m; i++)
 {
  for (int j=0; j<n; j++)
  {
   cout<<a[n*i+j]<<" ";
  }
  cout<<"\n";
 }

 cout<<"b matrix: \n";
 for (int i=0; i<n; i++)
 {
  for (int j=0; j<k; j++)
  {
   cout<<b[k*i+j]<<" ";
  }
  cout<<"\n";
 }

 cout<<"c matrix: \n";
 for (int i=0; i<m; i++)
 {
  for (int j=0; j<k; j++)
  {
   cout<<c[k*i+j]<<" ";
  }
  cout<<"\n";
 }


 cudaFree(dev_a);
 cudaFree(dev_b);
 cudaFree(dev_c);
 
 delete [] a;
 delete [] b;
 delete [] c;

}
