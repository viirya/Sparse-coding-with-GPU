/* Author: Anand Madhavan */
#ifndef __MATRIX_H__
#define __MATRIX_H__
#include <vector>
#include <string>

struct Matrix {
	float* values;
	int num_ptrs;
	int num_vals; // #cols (if row contiguous) or  #rows (if col contiguous)
	bool row_contiguous; 
	// if true, values is a vector of row pointers
	// if false, values is a vector of col pointers
};

int num_cols(const Matrix& m);
int num_rows(const Matrix& m);

float get_val(const Matrix& m, int row, int col);
void set_val(Matrix& m, int row, int col, float value);
bool init(Matrix& m, int num_rows, int num_cols, bool row_contiguous=true);
float* get_col(Matrix& m, int col);
const float* get_col(const Matrix& m, int col);
void freeup(Matrix& m);
bool equals(const Matrix& m1, const Matrix& m2);
void assign(Matrix&m, const Matrix& from);

float avgdiff(const Matrix& m1, const Matrix& m2);
int nnz(const Matrix& m);

bool read_matrix(FILE* inf, Matrix& m, const std::string& name, bool row_contiguous=true); // false for cublas
bool read_matrix(FILE* inf, int row, int col, Matrix& m, bool row_contiguous=true);
void print_matrix(const Matrix& m, const std::string& name);

int size_in_bytes(const Matrix& A);

#endif // __MATRIX_H__
