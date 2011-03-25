/* Author: Anand Madhavan */

#include <iostream>
#include "cublas.h"
#include "Matrix.hh"
#include "proj_grad_descent.h"
#include "GPU.hh"
float proj_grad_descent_cu (Matrix& Bout, /* : output, size: k, n */ 
		float c, /* : input */
		float sigma, /* : input */
		float eta, 
		float beta,
		float tol,
		int niters,
		/*const*/ Matrix& Binit, /* : input, size: k, n */
		/*const*/ Matrix& X, /* : input, size: k, m */
		/*const*/ Matrix& S) /* : input, size: size n, m */ 
{
	cublasInit();
	gpu::checkCublasError();
	
	// allocate Binit, X and S, Bout on GPU.
	int k = num_rows(X); // A is B
	int m = num_cols(X); // X is S
	int n = num_cols(Binit); // Y is X
	
	float* B_on_dev;
	float* X_on_dev;
	float* S_on_dev;
	
	cublasAlloc(n*k,sizeof(float),(void**)&B_on_dev);
	cublasAlloc(k*m,sizeof(float),(void**)&X_on_dev);
	cublasAlloc(n*m,sizeof(float),(void**)&S_on_dev);
	
	// copy over values...
	cublasSetMatrix(k,n,sizeof(float),Binit.values,k,B_on_dev,k);
	cublasSetMatrix(k,m,sizeof(float),X.values,k,X_on_dev,k);
	cublasSetMatrix(n,m,sizeof(float),S.values,n,S_on_dev,n);
	
	// compute...
	float* SSt2_on_dev;
	float* XSt2_on_dev;
	float* G_on_dev;
	float* X_BS_on_dev;
	cublasAlloc(n*n,sizeof(float),(void**)&SSt2_on_dev);
	cublasAlloc(k*n,sizeof(float),(void**)&XSt2_on_dev);
	cublasAlloc(k*n,sizeof(float),(void**)&G_on_dev);
	cublasAlloc(k*m,sizeof(float),(void**)&X_BS_on_dev);

	//	ss2 = 2*S*S' = 2*S*S' + 0*SS2
	cublasSgemm('n','t',n,n,m,2.0,S_on_dev,n,S_on_dev,n,0,SSt2_on_dev,n);
	//	xs2 = 2*X*S' = 2*X*S' + 0*XS2
	cublasSgemm('n','t',k,n,m,2.0,X_on_dev,k,S_on_dev,n,0,XSt2_on_dev,k);

	std::vector<float>fobj;
	int mv_avg_win = 30;
	float mv_avg=0;
	int iter = 0;
	
	// scale down B
	for(int j=0;j<n;++j) {
		float sumsq = cublasSnrm2(k,B_on_dev+k*j,1);
		if (sumsq>c) {
			cublasSscal(k,sqrt(c)/sumsq,B_on_dev+k*j,1);			
		}
	}
	float nfobj,nfobj2;
	cublasSgemm('n','n',k,m,n,-1.0,B_on_dev,k,S_on_dev,n,0,X_BS_on_dev,k);
	cublasSaxpy(k*m,1.0,X_on_dev,1,X_BS_on_dev,1);
	nfobj = cublasSnrm2(k*m,X_BS_on_dev,1);
	nfobj2 = cublasSasum(k*m,S_on_dev,1);  
	nfobj = nfobj*nfobj/(2*sigma*sigma)+beta*nfobj2;
	std::cerr << "Initial objective: " << nfobj<<std::endl;
	fobj.push_back(nfobj);
	
	while(true && iter>niters) {
		iter++;
		float eta_this = 10*eta/(10+iter);
		cublasSgemm('n','n',k,n,n,1.0,B_on_dev,k,SSt2_on_dev,n,0,G_on_dev,k);
		// G = B*2SS'
		// G = G-2XS' (ie. G = B*2SS'-2XS')
		cublasSaxpy(k*n,-1,XSt2_on_dev,1,G_on_dev,1);
		// B = B - eta_this*G
		cublasSaxpy(k*n,-eta_this,G_on_dev,1,B_on_dev,1);
		for(int j=0;j<n; ++j) {
			float sumsq = cublasSnrm2(k,B_on_dev+k*j,1);
			if (sumsq>c) {
				cublasSscal(k,sqrt(c)/sumsq,B_on_dev+k*j,1);			
			}
		}
		cublasSgemm('n','n',k,m,n,-1.0,B_on_dev,k,S_on_dev,n,0,X_BS_on_dev,k);
		cublasSaxpy(k*m,1.0,X_on_dev,1,X_BS_on_dev,1);
		nfobj = cublasSnrm2(k*m,X_BS_on_dev,1);
		nfobj2 = cublasSasum(k*m,S_on_dev,1);  
		nfobj = nfobj*nfobj/(2*sigma*sigma)+beta*nfobj2;
		fobj.push_back(nfobj);
		if(iter>=mv_avg_win) {
			mv_avg = 0;
			for(int i=iter-mv_avg_win; i<iter; ++i) {
				mv_avg += fobj[i];
			}
			mv_avg = mv_avg/(float)mv_avg_win;
			float criteria = fabs((nfobj-mv_avg)/mv_avg);
			if(criteria<tol) 
				break;
		}
		std::cerr << std::endl;
	}
	std::cerr << std::endl;
	cublasFree(X_BS_on_dev);
	cublasFree(G_on_dev);
	cublasFree(XSt2_on_dev);
	cublasFree(SSt2_on_dev);
	// get results...
	cublasGetMatrix(k,n,sizeof(float),B_on_dev,k,Bout.values,k);
	
	// free up...
	cublasFree(B_on_dev);
	cublasFree(X_on_dev);
	cublasFree(S_on_dev);
	
	cublasShutdown();
	return fobj[fobj.size()-1];
}

void onetime_setup_pg(int k, int m, int n,
		float** SSt2_on_dev,
		float** XSt2_on_dev,
		float** G_on_dev,
		float** X_BS_on_dev) 
{
	cublasAlloc(n*n,sizeof(float),(void**)SSt2_on_dev);
	cublasAlloc(k*n,sizeof(float),(void**)XSt2_on_dev);
	cublasAlloc(k*n,sizeof(float),(void**)G_on_dev);
	cublasAlloc(k*m,sizeof(float),(void**)X_BS_on_dev);
}

void onetime_teardown_pg(
		float* SSt2_on_dev,
		float* XSt2_on_dev,
		float* G_on_dev,
		float* X_BS_on_dev) 
{
	cublasFree(X_BS_on_dev);
	cublasFree(G_on_dev);
	cublasFree(XSt2_on_dev);
	cublasFree(SSt2_on_dev);
}

float proj_grad_descent_cu_basic (float c, /* : input */
		float sigma, /* : input */
		float eta, float beta, int niter,
		int k, int m, int n,
		float* B_on_dev,
		float* X_on_dev,
		float* S_on_dev,
		float* SSt2_on_dev,
		float* XSt2_on_dev,
		float* G_on_dev,
		float* X_BS_on_dev) /* : input, size: size n, m */ 
{	
	// compute...
	//	ss2 = 2*S*S' = 2*S*S' + 0*SS2
	cublasSgemm('n','t',n,n,m,2.0,S_on_dev,n,S_on_dev,n,0,SSt2_on_dev,n);
	//	xs2 = 2*X*S' = 2*X*S' + 0*XS2
	cublasSgemm('n','t',k,n,m,2.0,X_on_dev,k,S_on_dev,n,0,XSt2_on_dev,k);

	std::vector<float>fobj;
	int iter = 0;
	
	// scale down B
	scale_down_b(n,k,c,B_on_dev);

	for(iter=0;iter<niter;++iter) {
		float eta_this = 10*eta/(10+iter);
		cublasSgemm('n','n',k,n,n,1.0,B_on_dev,k,SSt2_on_dev,n,0,G_on_dev,k);
		// G = B*2SS'
		// G = G-2XS' (ie. G = B*2SS'-2XS')
		cublasSaxpy(k*n,-1,XSt2_on_dev,1,G_on_dev,1);
		// B = B - eta_this*G
		cublasSaxpy(k*n,-eta_this,G_on_dev,1,B_on_dev,1);
		scale_down_b(n,k,c,B_on_dev);
	}
	return calc_objective(k,m,n,sigma,beta,B_on_dev,S_on_dev,X_on_dev,X_BS_on_dev);
}

float calc_objective(int k, int m, int n, float sigma, float beta, 
		float* B_on_dev, float* S_on_dev, float* X_on_dev, float* X_BS_on_dev) 
{
	cublasSgemm('n','n',k,m,n,-1.0,B_on_dev,k,S_on_dev,n,0,X_BS_on_dev,k);
	gpu::checkErrors(); gpu::checkCublasError();
	cublasSaxpy(k*m,1.0,X_on_dev,1,X_BS_on_dev,1);
	gpu::checkErrors(); gpu::checkCublasError();
	float nfobj = cublasSnrm2(k*m,X_BS_on_dev,1);
	gpu::checkErrors(); gpu::checkCublasError();
	float nfobj2 = cublasSasum(n*m,S_on_dev,1);  
	gpu::checkErrors(); gpu::checkCublasError();
        float fresidue, fsparsity;
        fresidue = nfobj*nfobj/(2*sigma*sigma);
        fsparsity = beta*nfobj2;
	nfobj = fresidue + fsparsity;
	return nfobj;
}

// scale down B
void scale_down_b(int n, int k, float c, float* B_on_dev) 
{
	for(int j=0;j<n;++j) {
		float sumsq = cublasSnrm2(k,B_on_dev+k*j,1);
		gpu::checkErrors(); gpu::checkCublasError();
		if (sumsq>c) {
			cublasSscal(k,sqrt(c)/sumsq,B_on_dev+k*j,1);			
			gpu::checkErrors(); gpu::checkCublasError();
		}
	}
}
