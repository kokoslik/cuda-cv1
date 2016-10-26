#include <iostream>
#include <cstring>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>
using namespace std;
typedef unsigned char uchar;

#define NX 32
#define NY 32

static void HandleError(cudaError_t err,
                        const char *file,
                        int line)
{
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err),
            file, line);
        exit(EXIT_FAILURE);
    }
}

#define HANDLE_ERROR( error ) (HandleError( error, __FILE__, __LINE__ ))

__global__ void gauss_filter(uchar* Imdev,uchar* Imnewdev,int oldwidth,int oldheight)
{
	uchar GK[]={2,4,5,4,2,
				4,9,12,9,4,
				5,12,15,12,5,
				4,9,12,9,4,
				2,4,5,4,2};
	int i=threadIdx.x+blockDim.x*blockIdx.x+2;
	int j=threadIdx.y+blockDim.y*blockIdx.y+2;
	if((i<oldheight-2)&&(j<oldwidth-2))
	{
		int res=0;
		for(int ii=0;ii<5;ii++)
			for(int jj=0;jj<5;jj++)
				res+=Imdev[(i-2+ii)*oldwidth+(j-2+jj)]*GK[ii*5+jj];
		Imnewdev[i*oldwidth+j]=res/159;
	}
}

__global__ void gradient(uchar* Imdev,uchar* Imnewdev,int oldwidth,int oldheight)
{
	int Gx[]={-1,0,1,
			  -2,0,2,
			  -1,0,1};
	int Gy[]={-1,-2,-1,
	           0, 0, 0,
			   1, 2, 1};
	int i=threadIdx.x+blockDim.x*blockIdx.x+1;
	int j=threadIdx.y+blockDim.y*blockIdx.y+1;
	if((i<oldheight-1)&&(j<oldwidth-1))
	{
		int res1=0,res2=0;
		for(int ii=0;ii<3;ii++)
			for(int jj=0;jj<3;jj++)
			{
				res1+=Imdev[(i-1+ii)*oldwidth+(j-1+jj)]*Gx[ii*3+jj];
				res2+=Imdev[(i-1+ii)*oldwidth+(j-1+jj)]*Gy[ii*3+jj];
			}
		Imnewdev[i*oldwidth+j]= __fsqrt_ru(res1*res1+res2*res2);
	}

}

float improcGPU(const char* infilename, const char* outfilename, int width,int height,int Glower,int Gupper)
{
	ifstream ifile(infilename);
	ofstream ofile(outfilename);
	float time;
	width+=4;
	height+=4;
	uchar *Im,*Imdev,*Imnewdev;
	size_t size=width*height*sizeof(uchar);
	Im=new uchar[width*height];
	memset(Im,255,size);
	cudaEvent_t start,stop;
	HANDLE_ERROR( cudaMalloc(&Imdev,size) );
	HANDLE_ERROR( cudaMalloc(&Imnewdev,size) );
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	dim3 threads(NX,NY,1),blocks(height%NX==0?height/NX:height/NX+1,width%NY==0?width/NY:width/NY+1);
	for(int i=2;i<height-2;i++)
		for(int j=2;j<width-2;j++)
			ifile>>(int&)Im[i*width+j];
	ifile.close();
	cudaEventRecord(start);
	HANDLE_ERROR( cudaMemcpy(Imdev,Im,size,cudaMemcpyHostToDevice) );
	gauss_filter<<<blocks,threads>>>(Imdev,Imnewdev,width,height);
	HANDLE_ERROR( cudaGetLastError() );
	HANDLE_ERROR( cudaDeviceSynchronize() );
	swap(Imdev,Imnewdev);
	gradient<<<blocks,threads>>>(Imdev,Imnewdev,width,height);
	HANDLE_ERROR( cudaMemcpy(Im,Imnewdev,size,cudaMemcpyDeviceToHost) );
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time,start,stop);
	for(int i=2;i<height-2;i++)
	{
			for(int j=2;j<width-2;j++)
				ofile<<(((Glower<Im[i*width+j])&&(Gupper>Im[i*width+j]))?255:0 )<<' ';
			ofile<<endl;
	}
	ofile.close();
	delete[] Im;
	HANDLE_ERROR( cudaFree(Imdev) );
	HANDLE_ERROR( cudaFree(Imnewdev) );
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	return time*1e-3;
}

int main()
{
	cout<<"Time: "<<improcGPU("img.txt","imfilt.txt",960,512,100,200)<<endl;
return 0;
}
