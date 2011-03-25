/* Author: Anand Madhavan */

#include <iostream>
#include "l1ls_coord_descent.hh"
#include "coreutils.hh"
#include "math.h"
#include "cutil.h"
#include "cuda.h"
#include "GPU.hh"
#include "cublas.h"


__device__ float norm1_cu(const float* x, int n) {
	float x1norm = 0;
	for(int i=0;i<n; ++i) {
		x1norm += fabs(x[i]);
	}
	return x1norm;
}

static __device__ __constant__ int const_k;
static __device__ __constant__ int const_m;
static __device__ __constant__ int const_n;
static __device__ __constant__ float const_gamma;
static __device__ __constant__ float alphas[5];

__device__ int num_alphas = 5;
__device__ float tol = 1e-6;

__device__ float setyta(float* yta, int j, float val) {
	return yta[blockIdx.x*const_n+j] = val; 
	// blockIdx.x is the col index and yta is row-contiguous
}

__device__ float getyta(const float* yta, int j) {
	return yta[blockIdx.x*const_n+j]; 
	// blockIdx.x is the col index and yta is row-contiguous
}

__device__ float gety(const float* y, int j) {
	return y[blockIdx.x*const_k+j]; 
	// blockIdx.x is the col index and y is col-contiguous
}

__device__ float geta(const float* a_on_dev, int row, int col) {
	return a_on_dev[col*const_k+row];
	// A is row-contiguous
}

__device__ float getata(const float* ata_on_dev, int row, int col) {
	return ata_on_dev[row*const_n+col];
	// AtA is row-contiguous
}

extern __shared__ float shared_buf[];

__shared__ float af;
__shared__ float bf;
__shared__ int imina;
__shared__ bool found_flag;
__shared__ bool abort_flag;
__shared__ float minhx;
__shared__ float hx;
__shared__ int iter;

inline __device__ int idx(int ib) {
	return blockDim.x*ib + threadIdx.x;
}

__global__ void kernel_l1ls_coord_descent_sub
(int* gcount, const float* a, const float* y, const float* ata, const float* yta, float* xout) 
{
  int tpb = (const_n>512)?512:const_n;
  // This block, handles one training data 
  // * Called with n threads per block, and m blocks

  float* x = shared_buf; // needs local, take norm
  float* buff = x+const_n; // needs local
  float* d = buff+const_n; // needs local
	
  int i=-1;
  for(i = threadIdx.x; i<const_n; i+=tpb) {
    x[i] = 0;
  }
  int j;
  if(threadIdx.x==0) {
    abort_flag=false;
    iter=0;
  }
  __syncthreads();	
  float xstar=0;
  while(iter<150 && !abort_flag) {
    for(i=threadIdx.x; i<const_n; i+=tpb) {
      buff[i] = 0;
      for(j=0;j<const_n;++j) {
        buff[i] += x[j]*getata(ata,j,i);
      }
      buff[i] = getyta(yta,i)-buff[i];
      if(fabs(-(buff[i] + getata(ata,i,i)*x[i])) <= const_gamma) {
        xstar = 0;
      } else if(-(buff[i] + getata(ata,i,i)*x[i]) > const_gamma) {
        xstar = (buff[i] + getata(ata,i,i)*x[i] + const_gamma)/getata(ata,i,i);
      } else {
        xstar = (buff[i] + getata(ata,i,i)*x[i] - const_gamma)/getata(ata,i,i);
      }
      d[i] = xstar-x[i];
    }
    __syncthreads(); // below depends on all being updated...
    if(threadIdx.x==0) {
      bf=0;
      for(j=0;j<const_n; ++j) {
        bf -= buff[j]*d[j];
      }
    }
    // ...DONE with buff as y_minus_Ax_t_A
    // REUSE buff... as dtAtA..
    __syncthreads();
    for(i=threadIdx.x; i<const_n; i+=tpb) {
      buff[i]=0;
      // below access pattern is also source of bank conflicts i'd imagine...
      for(j=0;j<const_n;++j) {
        buff[i] += getata(ata,j,i)*d[j];
      }
    }
    __syncthreads();
    if(threadIdx.x==0) { 
      af = 0; // compute a = 0.5*d'*AtA*d
      for(j=0;j<const_n;++j) {
        af += 0.5*buff[j]*d[j];
      }
    }
    __syncthreads(); // below uses all of x...
    if(threadIdx.x==0) {
      imina = -1;
      found_flag = false;
      minhx = const_gamma*norm1_cu(x,const_n);
    }
    __syncthreads(); // below uses 'found'...
    // REUSE buff as xn...
    j=0; // reuse j instead of ia...
    while(j<num_alphas && !found_flag) {
      for(i=threadIdx.x; i<const_n; i+=tpb) {
        buff[i] = x[i]+alphas[j]*d[i]; // reuse xstar
        //    buff[i] = x[i]+alpha*d[i]; // reuse xstar
      }
      __syncthreads(); // norm uses all the values...
      if(threadIdx.x==0) {
        hx = alphas[j]*(af*alphas[j] + bf) + const_gamma*norm1_cu(buff,const_n);
        //      hx = af*alpha*alpha + bf*alpha + const_gamma*norm1(buff,const_n);
        if(hx < minhx*(1-tol)) {
          imina = j;
	  minhx = hx;
	  found_flag = true;
	}
      }
      j++;
      __syncthreads(); // found used again at beginning of loop
    }
    __syncthreads();
    if(threadIdx.x==0) {
      if(imina==-1) {
        abort_flag=true;
      }
      iter++;
    }
    __syncthreads();
    if(!abort_flag) {
      for(i=threadIdx.x; i<const_n; i+=tpb) {
        x[i] += alphas[imina]*d[i];
      }
    }
    __syncthreads();		
  } // for each iteration
  __syncthreads();
  for(i=threadIdx.x; i<const_n; i+=tpb) {
    *(xout+blockIdx.x*const_n+i) = x[i];
  }
  // xout is of size nxm and is col-contiguous
  if(threadIdx.x==0) {
    if(iter>=150) (*gcount)++;
  }
  __syncthreads();
}

void onetime_teardown(float* A_on_dev, float* AtA_on_dev,
		float* Y_on_dev, float* YtA_on_dev, float* Xn_on_dev)
{
	cublasFree(A_on_dev);
	cublasFree(Y_on_dev);
	cublasFree(Xn_on_dev);

	cudaFree(AtA_on_dev);
	cudaFree(YtA_on_dev);
}

void onetime_setup(int k, int m, int n, float gamma, float** A_on_dev, float** AtA_on_dev, 
		float** Y_on_dev, float** YtA_on_dev, float** Xn_on_dev)
{
	int asize = k*n;
	int ysize = k*m;
	int xoutsize = n*m;
	cublasAlloc(asize,sizeof(float),(void**)A_on_dev);
	cublasAlloc(ysize,sizeof(float),(void**)Y_on_dev);
	cublasAlloc(xoutsize,sizeof(float),(void**)Xn_on_dev);

	int atasize = n*n*sizeof(float);
	int ytasize = m*n*sizeof(float);
	cutilSafeCall(cudaMalloc((void**)(AtA_on_dev),atasize));
	cutilSafeCall(cudaMalloc((void**)(YtA_on_dev),ytasize));

	float ha[] = {3, 1, 1e-1, 3e-2, 1e-2};
	cutilSafeCall(cudaMemcpyToSymbol(const_k,(const void*)(&k),sizeof(k)));
	cutilSafeCall(cudaMemcpyToSymbol(const_m,(const void*)(&m),sizeof(m)));
	cutilSafeCall(cudaMemcpyToSymbol(const_n,(const void*)(&n),sizeof(n)));
	cutilSafeCall(cudaMemcpyToSymbol(const_gamma,(const void*)(&gamma),sizeof(gamma)));
	cutilSafeCall(cudaMemcpyToSymbol(alphas,(const void*)(&ha),5*sizeof(float)));
}

void setup_device_memory(const Matrix& A, const Matrix& Y, 
		float** A_on_dev, float** Y_on_dev, 
		float** AtA_on_dev, float** YtA_on_dev, float** Xn_on_dev, float gamma)
{
	int k = num_rows(Y);
	int m = num_cols(Y);
	int n = num_cols(A);
	int asize = k*n*sizeof(float);
	int ysize = k*m*sizeof(float);
	int atasize = n*n*sizeof(float);
	int ytasize = m*n*sizeof(float);
	int xoutsize = n*m*sizeof(float);
	
	cutilSafeCall(cudaMalloc((void**)(A_on_dev),asize));
	cutilSafeCall(cudaMalloc((void**)(AtA_on_dev),atasize));
	cutilSafeCall(cudaMalloc((void**)(Y_on_dev),ysize));
	cutilSafeCall(cudaMalloc((void**)(YtA_on_dev),ytasize));
	cutilSafeCall(cudaMalloc((void**)(Xn_on_dev),xoutsize));
	
	float ha[] = {3, 1, 1e-1, 3e-2, 1e-2};
	cutilSafeCall(cudaMemcpyToSymbol(const_k,(const void*)(&k),sizeof(k)));
	cutilSafeCall(cudaMemcpyToSymbol(const_m,(const void*)(&m),sizeof(m)));
	cutilSafeCall(cudaMemcpyToSymbol(const_n,(const void*)(&n),sizeof(n)));
	cutilSafeCall(cudaMemcpyToSymbol(const_gamma,(const void*)(&gamma),sizeof(gamma)));
	cutilSafeCall(cudaMemcpyToSymbol(alphas,(const void*)(&ha),5*sizeof(float)));

	cutilSafeCall(cudaMemcpy(*A_on_dev,A.values,asize,cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(*Y_on_dev,Y.values,ysize,cudaMemcpyHostToDevice));
	gpu::checkErrors(); 
}

__global__ void kernel_compute_AtA(const float* a, float* AtA_on_dev) 
{
	int i, j;
	i = threadIdx.x+blockIdx.x*blockDim.x;
	j = threadIdx.y+blockIdx.y*blockDim.y;
	if((i<const_n) && (j<const_n)) {
		float temp=0.0;
		for(int l=0; l<const_k; ++l) {
			temp += geta(a,l,i)*geta(a,l,j);
		}
		*(AtA_on_dev+const_n*i+j) = temp;
	}
}

__global__ void kernel_compute_YtA(const float* A_on_dev, float* Y_on_dev, 
		float* YtA_on_dev) 
{
  int tpb = (const_n>512)?512:const_n;
  for(int t = threadIdx.x; t<const_n; t+=tpb) {
    setyta(YtA_on_dev,t,0);
    int j;
    for(j=0;j<const_k;++j) {
      setyta(YtA_on_dev,t,
      getyta(YtA_on_dev,t)+gety(Y_on_dev,j)*geta(A_on_dev,j,t));
    }
  }
}

void compute_YtA_on_device(const float* A_on_dev, float* Y_on_dev, float* YtA_on_dev, int m, int n)
{
        int tpb = (n>512)?512:n;
	dim3 blocks(m);
	dim3 threads_per_block(tpb);
	kernel_compute_YtA<<<blocks,threads_per_block>>>(A_on_dev,Y_on_dev,YtA_on_dev);
	gpu::checkErrors();
}

void compute_AtA_on_device(const float* A_on_dev, float* AtA_on_dev, int k, int n)
{
	// replace with cublas impl..
	// this is a trivial impl, should have a lot of bank conflicts and such...
	dim3 blocks(((n-1)/16) + 1,((n-1)/16) + 1);
	dim3 threads_per_block(16,16);
	kernel_compute_AtA<<<blocks,threads_per_block>>>(A_on_dev,AtA_on_dev);
	gpu::checkErrors();
}

void teardown_device_memory(float* A_on_dev, float* Y_on_dev, 
		float* AtA_on_dev, float* YtA_on_dev, 
		float* Xn_on_dev, Matrix& Xout, int n, int m) 
{
	cutilSafeCall(cudaFree(A_on_dev));
	cutilSafeCall(cudaFree(AtA_on_dev));
	cutilSafeCall(cudaFree(Y_on_dev));
	cutilSafeCall(cudaFree(YtA_on_dev));
	init(Xout,n,m,false); // NOTE: here we use internal data knowledge
	cutilSafeCall(cudaMemcpy((void *)(Xout.values),(const void *)(Xn_on_dev),
			n*m*sizeof(float),cudaMemcpyDeviceToHost));	
	cutilSafeCall(cudaFree(Xn_on_dev));
	gpu::checkErrors(); 
}

void l1ls_coord_descent_cu (Matrix& Xout, /* : output, size: n, m */ 
		float gamma, /* : input */
		const Matrix& A, /* : input, size: k, n */
		const Matrix& Y)
{		
	int k = num_rows(Y);
	int m = num_cols(Y);
	int n = num_cols(A);

	// * Transfer A, Y data (use Xinit=0), allocate device memory
	float* A_on_dev;
	float* AtA_on_dev;
	float* Y_on_dev;
	float* Xn_on_dev;
	float* YtA_on_dev;
	setup_device_memory(A,Y,&A_on_dev,&Y_on_dev,&AtA_on_dev,
			&YtA_on_dev,&Xn_on_dev,gamma);
	gpu::checkErrors(); 
	
	compute_AtA_on_device(A_on_dev,AtA_on_dev,k,n);
	gpu::checkErrors(); 
	compute_YtA_on_device(A_on_dev,Y_on_dev,YtA_on_dev,m,n);
	gpu::checkErrors(); 

        int tpb = (n>512)?512:n;
	// * Allocate sizeof(float)*(7*n) bytes per block shared memory
	// * Then call with n threads per block, and m blocks
	dim3 blocks(m);
	dim3 threads_per_block(tpb);
	int shared_mem_size = sizeof(float)*3*n; 
	cudaDeviceProp prop;
	gpu::get_device_infos(0,prop);
	
        if(tpb>prop.maxThreadsPerBlock) {
		std::cerr << "Cannot launch " << n << 
		" threads per block. Max " << prop.maxThreadsPerBlock << " allowed\n";
		exit(EXIT_FAILURE);
	}
        int* gcounts;
	int counts=0;
	cutilSafeCall(cudaMalloc((void**)(&gcounts),sizeof(int)));
    cutilSafeCall(cudaMemcpy(gcounts,&counts,sizeof(int),cudaMemcpyHostToDevice));
	kernel_l1ls_coord_descent_sub<<<blocks,threads_per_block,shared_mem_size>>>
	                  (gcounts,A_on_dev,Y_on_dev,AtA_on_dev,YtA_on_dev,Xn_on_dev);
	cutilSafeCall(cudaMemcpy((void *)(&counts),(const void *)(gcounts),
			sizeof(int),cudaMemcpyDeviceToHost));	
	cutilSafeCall(cudaFree(gcounts));
	if(!gpu::checkErrors()) {
		teardown_device_memory(A_on_dev,Y_on_dev,AtA_on_dev,
				YtA_on_dev,Xn_on_dev,Xout,n,m);	
		exit(EXIT_FAILURE);
	}

	// * Free device memory
	teardown_device_memory(A_on_dev,Y_on_dev,AtA_on_dev,YtA_on_dev,Xn_on_dev,
			Xout,n,m);	
	return;
}

void l1ls_coord_descent_cu_basic (int k, int m, int n,
		const float* A_on_dev,
		float* AtA_on_dev,
		float* Y_on_dev, 
		float* YtA_on_dev, 
		float* Xn_on_dev)
{		
	compute_AtA_on_device(A_on_dev,AtA_on_dev,k,n);	
	compute_YtA_on_device(A_on_dev,Y_on_dev,YtA_on_dev,m,n);
        int tpb = (n>512)?512:n;

	dim3 blocks(m);
	dim3 threads_per_block(tpb);
	int shared_mem_size = sizeof(float)*3*n; 
	
	cudaDeviceProp prop;
	gpu::get_device_infos(0,prop);
	if(tpb > prop.maxThreadsPerBlock) {
		std::cerr << "Cannot launch " << tpb << 
		" threads per block. Max " << prop.maxThreadsPerBlock << " allowed\n";
		exit(EXIT_FAILURE);
	}
    int* gcounts;
	int counts=0;
	cutilSafeCall(cudaMalloc((void**)(&gcounts),sizeof(int)));
    cutilSafeCall(cudaMemcpy(gcounts,&counts,sizeof(int),cudaMemcpyHostToDevice));
	kernel_l1ls_coord_descent_sub<<<blocks,threads_per_block,shared_mem_size>>>
	                  (gcounts,A_on_dev,Y_on_dev,AtA_on_dev,YtA_on_dev,Xn_on_dev);
	cutilSafeCall(cudaMemcpy((void *)(&counts),(const void *)(gcounts),
			sizeof(int),cudaMemcpyDeviceToHost));	
	cutilSafeCall(cudaFree(gcounts));
	return;
}
