/* Author: Anand Madhavan */

#ifndef __L1LS_COORD_DESCENT_H__
#define __L1LS_COORD_DESCENT_H__

#include "Matrix.hh"

// Plain CPU version...
void l1ls_coord_descent (Matrix& Xout, /* : output, size: n, m */ 
		float gamma, /* : input */
		const Matrix& A, /* : input, size: k, n */
		const Matrix& Y, /* : input, size: k, m */
		const Matrix& Xinit); /* : input, size: size n, m */

// CUDA version...
void l1ls_coord_descent_cu (Matrix& Xout, /* : output, size: n, m */ 
		float gamma, /* : input */
		const Matrix& A, /* : input, size: k, n */
		const Matrix& Y);

// This is used along with projected gradient descent and 
// memory reuse...
void l1ls_coord_descent_cu_basic (int k, int m, int n,
		const float* A_on_dev,
		float* AtA_on_dev,
		float* Y_on_dev, 
		float* YtA_on_dev, 
		float* Xn_on_dev);

// Ctor/Dtor type methods used with above...
void onetime_setup(int k, int m, int n, float gamma, 
		float** A_on_dev, float** AtA_on_dev, 
		float** Y_on_dev, float** YtA_on_dev, float** Xn_on_dev);
void onetime_teardown(float* A_on_dev, float* AtA_on_dev,
		float* Y_on_dev, float* YtA_on_dev, float* Xn_on_dev);

#endif
