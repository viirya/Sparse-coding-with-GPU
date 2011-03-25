/* Author: Anand Madhavan */

#include <stdio.h>
#include <stdlib.h>
#include "Matrix.hh"
#include <iostream>
#include "coreutils.hh"
#include "math.h"
using namespace std;

int size_in_bytes(const Matrix& A) 
{
	return num_cols(A)*num_rows(A)*sizeof(float);
}

void assert_bounds(const Matrix& m, int row, int col) {
	if(m.row_contiguous) {
		if(row>=m.num_ptrs || (col >=m.num_vals) || (row < 0) || (col < 0) ) {
			DEBUGIFY(std::cerr << "Accessing out of bound indeces: r: " << row << " c: " << col <<std::endl);
		}	
	} else {
		if(col>=m.num_ptrs || (row >=m.num_vals) || (col < 0) || (row < 0) ) {
			DEBUGIFY(std::cerr << "Accessing out of bound indeces: r: " << row << " c: " << col <<std::endl);
		}
	}
}

float get_val(const Matrix& m, int row, int col) {
	assert_bounds(m,row,col);
	if(!m.row_contiguous)
		return m.values[m.num_vals*col+row];
	else 
		return m.values[m.num_vals*row+col];
}

void set_val(Matrix& m, int row, int col, float value) {
	assert_bounds(m,row,col);
	if(!m.row_contiguous) 
		m.values[m.num_vals*col+row] = value;
	else
		m.values[m.num_vals*row+col] = value;
}

void assign(Matrix&m, const Matrix& from) {
	freeup(m);
	init(m,from.num_ptrs,from.num_vals);
	for(int i=0;i<from.num_ptrs;i++) {
		for(int j=0;j<from.num_vals;j++) {
			set_val(m,i,j,get_val(from,i,j));
		}
	}
}

int num_cols(const Matrix& m) {
	if(!m.row_contiguous)
		return m.num_ptrs;
	return m.num_vals;
}

int num_rows(const Matrix& m) {
	if(!m.row_contiguous)
		return m.num_vals;
	return m.num_ptrs;
}

int nnz(const Matrix& m) {
	int nonzeros = 0;
	for(int j=0;j<num_cols(m);j++) {
		for(int i=0;i<num_rows(m);i++) {
			if(fabs(get_val(m,i,j))>1e-14)
				nonzeros++;
		}
	}
	return nonzeros;
}

bool init(Matrix& m, int num_rows, int num_cols, bool row_contiguous) {
	m.row_contiguous = row_contiguous;	
	if(row_contiguous) {
		m.num_vals = num_cols;
		m.num_ptrs = num_rows;
	} else {
		m.num_vals = num_rows;
		m.num_ptrs = num_cols;
	}
	m.values = (float*) malloc(m.num_ptrs*m.num_vals*sizeof(float));
	if(m.values==0) {
		cerr << "Cannot allocate enough memory through malloc for matrix: [" << num_rows << "x" << num_cols << "]\n";
		exit(1);
	}
	return true;
}

void freeup(Matrix& m) {
	free(m.values);
	m.values = 0;
	m.num_ptrs = 0;
	m.num_vals = 0;
}

bool equals(const Matrix& m1, const Matrix& m2) {
	if(m1.num_ptrs!=m2.num_ptrs)
		return false;
	if(m1.num_vals!=m2.num_vals)
		return false;
	if(m1.row_contiguous!=m2.row_contiguous)
		return false; // for now.
	for(int i=0;i<m1.num_ptrs;i++) {
		for(int j=0;j<m1.num_vals;j++) {
			if(get_val(m1,i,j)!=get_val(m2,i,j)) {
				DEBUGIFY(std::cerr << "Not equal: " << get_val(m1,i,j) << " " << get_val(m2,i,j)<<std::endl);
				return false;
			}
		}
	}
	return true;
}

bool read_matrix(FILE* inf, int row, int col, Matrix& m, bool row_contiguous)
{
	init(m,row,col,row_contiguous);
	for(int i=0;i<row;i++) {
		for(int j=0;j<col;j++) {
			float val;
			fscanf(inf,"%g ",&val);
			set_val(m,i,j,val);
			if(j==col-1) {
				fscanf(inf,"\n");
			}
		}
	}
	return true;
}

bool read_matrix(FILE* inf, Matrix& m, const std::string& tag, bool row_contiguous) {
	std::string head = tag;
	if(tag.empty())
		head.append("%d %d\n");
	else 
		head.append(" %d %d\n");
	int row, col;
	fscanf(inf,head.c_str(),&row,&col);
	init(m,row,col,row_contiguous);
	for(int i=0;i<row;i++) {
		for(int j=0;j<col;j++) {
			float val;
			fscanf(inf,"%g ",&val);
			set_val(m,i,j,val);
			if(j==col-1) {
				fscanf(inf,"\n");
			}
		}
	}
	return true;
}

float avgdiff(const Matrix& m1, const Matrix& m2) {
	float diffsqr = 0.0;
	int nonzeros = 0;
	for(int j=0;j<num_cols(m2);j++) {
		for(int i=0;i<num_rows(m1);i++) {
			float diff = (get_val(m1,i,j)-get_val(m2,i,j));
			diff = diff*diff;
			diffsqr+=diff;
			if(fabs(get_val(m2,i,j))>1e-14)
				nonzeros++;
		}
	}
	return sqrt(diffsqr)/nonzeros;	
}

const float* get_col(const Matrix& m, int col) {
	return get_col(const_cast<Matrix&> (m),col);
}

float* get_col(Matrix& m, int col) {
	if(m.row_contiguous) {
		DEBUGIFY(std::cerr << "Can't do get_col on matrix initialized using row_contiguous\n");
		return 0;
	}
	assert_bounds(m,0,col);
	return m.values+m.num_vals*col;
}

void print_matrix(const Matrix& m, const std::string& tag) {
	std::cerr << tag << " " << num_rows(m) << " " << num_cols(m) << std::endl;
	for(int i = 0; i < num_rows(m); i++) {
		for(int j = 0; j < num_cols(m); j++) {
			std::cerr << get_val(m,i,j) << " ";
			if(j == num_cols(m)-1) 
				std::cerr << std::endl;
		}
	}
}
