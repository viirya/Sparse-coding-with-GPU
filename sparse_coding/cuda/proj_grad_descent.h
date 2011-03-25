/* Author: Anand Madhavan */
#ifndef __PROJ_GRAD_DESCENT_H__
#define __PROJ_GRAD_DESCENT_H__

#include "Matrix.hh"

float proj_grad_descent_cu (Matrix& Bout, /* : output, size: k, n */ 
		float c, /* : input */
		float sigma, /* : input */
		float eta, 
		float beta,
		float tol,
		int niters,
		/*const*/ Matrix& Binit, /* : input, size: k, n */
		/*const*/ Matrix& X, /* : input, size: k, m */
		/*const*/ Matrix& S); /* : input, size: size n, m */

// Version of above used in conjunction with onetime_setup_pg and onetime_teardown_pg..
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
		float* X_BS_on_dev);

void onetime_setup_pg(int k, int m, int n,
		float** SSt2_on_dev,
		float** XSt2_on_dev,
		float** G_on_dev,
		float** X_BS_on_dev);
void onetime_teardown_pg(float* SSt2_on_dev,
		float* XSt2_on_dev,
		float* G_on_dev,
		float* X_BS_on_dev);

float calc_objective(int k, int m, int n, float sigma, float beta, 
		float* B_on_dev, float* S_on_dev, float* X_on_dev, float* X_BS_on_dev);
void scale_down_b(int n, int k, float c, float* B_on_dev);

#endif
