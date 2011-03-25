/* Author: Anand Madhavan */
/*  Pure C version of the coordinate descent algorithm without cuda */

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "l1ls_coord_descent.hh"
#include "coreutils.hh"
#include "math.h"

float norm1(const float* x, int n) {
	float x1norm = 0;
	for(int i=0;i<n; ++i) {
		x1norm += fabs(x[i]);
	}
	return x1norm;
}

void l1ls_coord_descent_sub(float* xout, 
		float gamma, 
		const Matrix& A,
		const float* y,
	    const float* xinit,
	    const Matrix& AtA) {
	const int num_alphas = 5;
	float alphas[num_alphas] = { 1, 3e-1, 1e-1, 3e-2, 1e-2 };
	float tol = 1e-6;
	int n = num_cols(A);
	int k = num_rows(A);
	
	float* x = (float*) malloc(sizeof(float)*n);
	for(int i=0;i<n;++i) {
		x[i] = 0;//xinit[i];
	}
	float* ytA = (float*) malloc(sizeof(float)*n);
	for(int i=0;i<n;++i) {
		float temp=0;
		for(int j=0;j<k;++j) {
			temp+=y[j]*get_val(A,j,i);
		}
		ytA[i]=temp;
	}
	float* xstar = (float*) malloc(sizeof(float)*n);
	float* y_minus_Ax_t_A = (float*)malloc(sizeof(float)*n);
	float* d = (float*) malloc(sizeof(float)*n);
	float* xn = (float*) malloc(sizeof(float)*n);
	float* dtAtA = (float*)malloc(sizeof(float)*n);
	for(int iter=0;iter<700;++iter) {
		for(int i=0;i<n;++i) {
			float temp = 0;
			for(int j=0;j<n;++j)
				temp += x[j]*get_val(AtA,j,i);
			y_minus_Ax_t_A[i] = ytA[i]-temp;
		}
		for(int j=0; j<n; ++j) {
			float Pj = 0.5*get_val(AtA,j,j);
			float Qj = y_minus_Ax_t_A[j] + get_val(AtA,j,j)*x[j];
			float Dj = -Qj;
			if(fabs(Dj)< gamma)
				xstar[j] = 0;
			else if(Dj > gamma)
				xstar[j] = (Qj + gamma)/(2.0*Pj);
			else
				xstar[j] = (Qj - gamma)/(2.0*Pj);
		}
		for(int i=0;i<n;++i) {
			d[i] = xstar[i]-x[i];
		}
		float a = 0; // compute a = 0.5*d'*AtA*d
		for(int i=0;i<n; ++i) {
			float temp=0.0;
			for(int j=0;j<n;++j)
				temp += get_val(AtA,j,i)*d[j];
			dtAtA[i] = temp;
		}
		for(int i=0;i<n;++i) {
			a += 0.5*dtAtA[i]*d[i];
		}

		float b = 0;
		// compute b = - y_minus_Ax_t_A*d
		for(int i=0;i<n; ++i) {
			b = b - y_minus_Ax_t_A[i]*d[i];
		}
		float minhx = gamma*norm1(x,n);
		int imina = -1;
		bool found = false;
		int ia=0;
		while(ia<num_alphas && !found) {
			float alpha = alphas[ia];
			for(int i=0;i<n;++i) {
				xn[i] = x[i]+alpha*d[i];
			}
			float hx = a*alpha*alpha + b*alpha + gamma*norm1(xn,n);
			if(hx < minhx*(1-tol)) {
				imina = ia;
				minhx = hx;
				found = true;
			}
			ia++;
		}
		if(imina==-1) {
			printf("breaking becahse imina==-1");
			break;
		}
		for(int i=0;i<n;++i) {
			x[i] = x[i] + alphas[imina]*d[i];
		}
	}
	free(dtAtA);
	free(xn);
	free(d);
	free(y_minus_Ax_t_A);

	for(int i=0;i<n;++i) {
		xout[i] = x[i];
	}
	free(x);
	free(xstar);
	free(ytA);
}

void l1ls_coord_descent (Matrix& Xout, /* : output, size: n, m */ 
		float gamma, /* : input */
		const Matrix& A, /* : input, size: k, n */
		const Matrix& Y, /* : input, size: k, m */
		const Matrix& Xinit) { /*: input, size: n, m */
	// AtA...
	Matrix AtA;
	init(AtA,num_cols(A),num_cols(A));
	init(Xout, num_rows(Xinit),num_cols(Xinit),false);
	for(int i=0; i<num_cols(A); ++i) {
		for(int j=0; j<num_cols(A); ++j) {
			float temp=0.0;
			for(int l=0; l<num_rows(A); ++l) {
				temp+=get_val(A,l,i)*get_val(A,l,j);
			}
			set_val(AtA,i,j,temp);
		}
	}
	//... AtA
	for(int i=0; i<num_cols(Y); i++) {
		const float* xinit = get_col(Xinit,i);
		float* xout = get_col(Xout,i);
		l1ls_coord_descent_sub(xout,gamma,A,get_col(Y,i),xinit,AtA);
	}
	freeup(AtA);
}
