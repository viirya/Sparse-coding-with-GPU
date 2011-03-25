/* Author: Anand Madhavan */
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include "utils.hh"
#include "GPU.hh"
#include <vector>
#include <string>
#include <set>
#include "cublas.h"
#include <sstream>

#include "Matrix.hh"
#include "l1ls_coord_descent.hh"
#include "proj_grad_descent.h"
using namespace std;

bool g_verbose;

void print_usage(const char** argv)
{
    std::cout << "Usage: " << argv[0] << "-runwhat=<natural|digits|basis|coeffs|findcoeffs> "<< 
    "-mb=<batch size> -nb=<number of batches> " << 
    "-n=<# basis vectors> -k=<num dims> -nepoch=<iterations> -imagesdir=<images dir>" << 
    "-filename=<coeffs/digits file> -basisdir=<basis output directory> " << 
    "-beta=<beta> -sigma=<sigma> -eta=<eta> " << 
    "-nbasis_iters=<number of iterations for basis computation> " << 
    "-basisfile=<basis file for coefficient computation> " << 
    "-labelfile= < label file for letters> " << std::endl;
}

struct Options {
  Options():device(0),sigma(1),c(1),nepoch(1),
  nbasis_iters(15),eta(0.01),beta(0.4),tol(1e-1),
  filename(0),runwhat(0),basisfile(0),labelfile(0),numtrain(100){ }

  ~Options(){}
  int device;
  bool verbose;
  bool run_tests;
  char *filename, *runwhat, *basisfile, *labelfile;
  char* imagesdir, *basisdir;
  int k;
  int mb,nb; // m for a batch, nb batchces
  int n, nepoch;
  float sigma, c;
  float eta, beta, tol;
  int nbasis_iters, numtrain;
  
};

bool read_data_coeff(const std::string& fname,
		float& gamma, 
		Matrix& A, Matrix& Y, 
		Matrix& Xinit, Matrix& Xout)
{
  Stats stats;
  // read file small_file.txt
  // read matrices
  FILE* inf = fopen(fname.c_str(),"rt");
  if(inf==0) {
	  std::cerr << "Cannot open file\n";
	  return false;
  }
  fscanf(inf,"gamma %g\n",&gamma);
//  std::cerr << "gamma " << gamma << std::endl;
  if(!read_matrix(inf,A,"A",false)) {
	  cout << "Error reading matrix A\n";
	  return false;
  }
//  DEBUGIFY(print_matrix(A,"A"));
  if(!read_matrix(inf,Y,"Y",false)) {
	  cout << "Error reading matrix Y\n";
	  return false;
  }
//  DEBUGIFY(print_matrix(Y,"Y"));
  if(!read_matrix(inf,Xinit,"Xinit",false)) {
	  cout << "Error reading matrix Xinit\n";
	  return false;
  }
//  DEBUGIFY(print_matrix(Xinit,"Xinit"));
  if(!read_matrix(inf,Xout,"Xout",false)) {
	  cout << "Error reading matrix Xout\n";
	  return false;
  }
//  DEBUGIFY(print_matrix(Xout,"Xout"));
  fclose(inf);
  return true;
}

bool test_cpu(float gamma, Matrix& A, Matrix& Y, Matrix& Xinit, Matrix& Xout) 
{
	Matrix c_xout;
	float expt_time=0;
	{
		cpu::CpuEventTimer timer(expt_time);
		l1ls_coord_descent(c_xout, gamma, A, Y, Xinit);
	}
	std::cerr<<"\nAvg error in xout: " << avgdiff(c_xout,Xout) << ", cpu time: " << expt_time << "[ms]\n";	
	freeup(c_xout);
	return true;
}

bool test_gpu(float gamma, Matrix& A, Matrix& Y, Matrix& Xinit, Matrix& Xout) 
{
	Matrix c_xout;
	float expt_time=0;
	{
		gpu::GpuEventTimer timer(expt_time);
		l1ls_coord_descent_cu(c_xout, gamma, A, Y);
	}
	std::cerr<<"\nAvg error in xout: " << avgdiff(c_xout,Xout) << ", gpu time: " << expt_time << "[ms]\n";	
	freeup(c_xout);
	return true;
}

bool test(bool cpu)
{
	std::string fname("data.txt");
	float gamma;
	Matrix A, Y, Xinit, Xout;
	if(!read_data_coeff(fname,gamma,A,Y,Xinit,Xout)) {
		std::cerr << "Error reading matrices\n";
		return false;
	}
	
	int k = num_rows(Y);
	int m = num_cols(Y);
	int n = num_cols(A);

	printf("k=%d, m=%d, n=%d\n",k,m,n);

	bool ret_val;
	if(cpu)
		ret_val = test_cpu(gamma,A,Y,Xinit,Xout);
	else
		ret_val = test_gpu(gamma,A,Y,Xinit,Xout);
	
	freeup(A);
	freeup(Y);
	freeup(Xinit);
	freeup(Xout);
	return ret_val;
}

void run_expt(const Options& opts)
{
	std::string fname(opts.filename?opts.filename:"data.txt");
	float gamma;
	Matrix A, Y, Xinit, Xout;
	if(!read_data_coeff(fname,gamma,A,Y,Xinit,Xout)) {
		std::cerr << "Error reading matrices\n";
		return;
	}
	
	int k = num_rows(Y);
	int m = num_cols(Y);
	int n = num_cols(A);

	Matrix c_xout;
	float expt_time=0;
	{
		gpu::GpuEventTimer timer(expt_time);
		l1ls_coord_descent_cu(c_xout, gamma, A, Y);
	}
	printf("gamma: %.5f\n",gamma);
	printf("k: %d\nm: %d\nn: %d\n",k,m,n);
	printf("avg_error_in_dd_cuda: %f\n", avgdiff(c_xout,Xout));
	printf("dd_cuda_time: %f\n",expt_time/1000.0);

	freeup(c_xout);	
	freeup(A);
	freeup(Y);
	freeup(Xinit);
	freeup(Xout);
	return;
}

bool run(const Options& opts, Stats& stats) {
  DEBUGIFY(std::cerr << "\nRunning\n";);
  if(opts.device<0) {
    cpu::CpuEventTimer timer(stats.total_time);
    test(true);
  }
  else {
     cpu::CpuEventTimer timer(stats.total_time);
  }
  return true;
}

void read_options(int argc, const char** argv, Options& opts) {
  if(cutCheckCmdLineFlag(argc, argv, "help")) {
    print_usage(argv);
    exit(1); 
  }
  if(cutCheckCmdLineFlag(argc,argv,"filename"))
	  cutGetCmdLineArgumentstr(argc, argv, "filename", &(opts.filename));
  
  if(cutCheckCmdLineFlag(argc,argv,"imagesdir"))
	  cutGetCmdLineArgumentstr(argc, argv, "imagesdir", &(opts.imagesdir));

  if(cutCheckCmdLineFlag(argc,argv,"basisdir"))
	  cutGetCmdLineArgumentstr(argc, argv, "basisdir", &(opts.basisdir));

  if(cutCheckCmdLineFlag(argc,argv,"runwhat"))
	  cutGetCmdLineArgumentstr(argc, argv, "runwhat", &(opts.runwhat));

  if(cutCheckCmdLineFlag(argc,argv,"basisfile"))
	  cutGetCmdLineArgumentstr(argc, argv, "basisfile", &(opts.basisfile));
  
  if(cutCheckCmdLineFlag(argc,argv,"labelfile"))
	  cutGetCmdLineArgumentstr(argc, argv, "labelfile", &(opts.labelfile));
  
#define GET_CMD_LINE_ARG_I(a)  cutGetCmdLineArgumenti(argc, argv, #a, &(opts.a)); 
  GET_CMD_LINE_ARG_I(k);
  GET_CMD_LINE_ARG_I(mb);
  GET_CMD_LINE_ARG_I(n);
  GET_CMD_LINE_ARG_I(nb);
  GET_CMD_LINE_ARG_I(nepoch);
  GET_CMD_LINE_ARG_I(numtrain);
  GET_CMD_LINE_ARG_I(device);
  GET_CMD_LINE_ARG_I(nbasis_iters);
#define GET_CMD_LINE_ARG_F(a)  cutGetCmdLineArgumentf(argc, argv, #a, &(opts.a)); 
  GET_CMD_LINE_ARG_F(sigma);
  GET_CMD_LINE_ARG_F(c);
  GET_CMD_LINE_ARG_F(eta);
  GET_CMD_LINE_ARG_F(beta);
  GET_CMD_LINE_ARG_F(tol);
#define GET_CMD_LINE_FLAG(a)  if(cutCheckCmdLineFlag(argc, argv, #a)) (opts.a) = true; 
  GET_CMD_LINE_FLAG(verbose);
  g_verbose = opts.verbose;
}

int main_coeffs(int argc, const char** argv)
{
	float expt_time=0;
	{ 
		cpu::CpuEventTimer timer(expt_time);

		Options opts;
		read_options(argc, argv, opts);

		std::string device_name = gpu::initialize_device(opts.device); 
		run_expt(opts);
	}
	return 0;
}

bool read_data_basis(const std::string& fname,
		float& c, float& sigma,  
		Matrix& Binit, Matrix& X, Matrix& S, Matrix& Bout) 
{ 
  // read matrices
  FILE* inf = fopen(fname.c_str(),"rt");
  if(inf==0) {
	  std::cerr << "Cannot open file\n";
	  return false;
  }
  fscanf(inf,"c %g\n",&c);
  std::cerr << "c " << c << std::endl;
  fscanf(inf,"sigma %g\n",&sigma);
  std::cerr << "sigma " << sigma << std::endl;
  if(!read_matrix(inf,Binit,"Binit",false)) {
	  cout << "Error reading matrix Binit\n";
	  return false;
  }
  if(!read_matrix(inf,X,"X",false)) {
	  cout << "Error reading matrix X\n";
	  return false;
  }
  if(!read_matrix(inf,S,"S",false)) {
	  cout << "Error reading matrix S\n";
	  return false;
  }
  if(!read_matrix(inf,Bout,"Bout",false)) {
	  cout << "Error reading matrix Bout\n";
	  return false;
  }
  fclose(inf);
  return true;
}

void run_expt_basis(const Options& opts)
{
	std::string fname(opts.filename?opts.filename:"data.txt");
	Matrix Binit; // A
	Matrix X; // Y
	Matrix S; // X dimensions
	Matrix Bout; // B
	float c, sigma;
	if(!read_data_basis(fname,c,sigma,Binit,X,S,Bout)) {
		std::cerr << "Error reading matrices\n";
		return;
	}
	int k = num_rows(X); // A is B
	int m = num_cols(X); // X is S
	int n = num_cols(Binit); // Y is X

	Matrix c_bout;
	init(c_bout,k,n,false);
	float expt_time=0;
	{
		gpu::GpuEventTimer timer(expt_time);
		proj_grad_descent_cu(c_bout, c, sigma, opts.eta, opts.beta, opts.tol, opts.nbasis_iters, Binit, X, S);
	}
	printf("c: %.5f\n",c);
	printf("sigma: %.5f\n",sigma);
	printf("k: %d\nm: %d\nn: %d\n",k,m,n);
	printf("avg_error_in_basis: %f\n", avgdiff(c_bout,Bout));
	printf("cublas_time: %f\n",expt_time/1000.0);

	freeup(c_bout);	
	freeup(Binit);
	freeup(X);
	freeup(S);
	freeup(Bout);
	return;
}

int main_basis(int argc, const char** argv)
{
	float expt_time=0;
	{ 
		cpu::CpuEventTimer timer(expt_time);

		Options opts;
		read_options(argc, argv, opts);

		std::string device_name = gpu::initialize_device(opts.device); 
		run_expt_basis(opts);
	}
	return 0;
}

int read_images(string dirname, vector<Matrix*>& imgs)
{
//	cerr << "Reading images\n";
	// store in 62 Matrix objects
	vector<string> img_files;

	DIR *dir;
	struct dirent *dirp;
	if((dir  = opendir(dirname.c_str())) == NULL) {
		cout << "Error(" << errno << ") opening directory: " 
                     << dirname<< endl;
		exit(-1);	
	}

	while ((dirp = readdir(dir)) != NULL) {
                string dirpname = string(dirp->d_name);
		if(dirpname==".")
			continue;
		if(dirpname=="..")
			continue;
                if(dirpname.find("IMAGES")==string::npos) {
                        cout << "Skipping file: " << dirpname << endl;
                        continue;
                }
		string tmp = dirname;
		tmp.append("/");
		tmp.append(dirpname);
		img_files.push_back(tmp);
	}
	closedir(dir);
	int ret_val=0;
	for(int i=0;i<img_files.size();i++) {
//		cerr << img_files[i]<<endl;
		Matrix* m = new Matrix();
		FILE* inf = fopen(img_files[i].c_str(),"rt");
		if(inf==0) {
			std::cerr << "Cannot open file\n";
			return false;
		}
		read_matrix(inf,*m,"",false);
//		print_matrix(*m,"Image");
		ret_val = num_rows(*m);
		imgs.push_back(m);
		fclose(inf);
	}
//	cerr << imgs.size() <<" images read\n";
//	cerr << "Done reading images\n";
	return ret_val;
}

bool write_basis(const Options& opts, string tag, Matrix& B, int iepoch) 
{
	stringstream tmpstr;
	tmpstr << opts.basisdir;
	tmpstr << "/basis_" << tag << iepoch << ".txt";
	FILE* ouf = fopen(tmpstr.str().c_str(),"wt");
	if(ouf==0) {
		cerr << "Cannot open file "<< tmpstr.str();
		return false;
	}
//	cerr << "B num rows: " << num_rows(B);
//	cerr << "B num cols: " << num_cols(B);
	for(int i=0;i<num_rows(B);++i) {
		for(int j=0;j<num_cols(B);++j) {
			fprintf(ouf,"%g ",get_val(B,i,j));
		}
		fprintf(ouf,"\n");
	}
	fclose(ouf);
	return true;
}

void get_images_as_input_matrix(Matrix& bigX, const Options& opts)
{
	int k = opts.k;
	int m = opts.mb;
	// 1) read all images into memory...
	vector<Matrix*> imgs;
	int img_size = read_images(opts.imagesdir,imgs);
	// 2) arrange images into X matrix: in random patches ...
	int dim = int(sqrt(opts.k));
	int nbatches = opts.nb;
	// first pick random image, then pick random patch in it
	// repeat this for 1000 images
	// repeat this 100 times or so
	int buffer = dim;
	// Initialize B and S to random values
	init(bigX, k, m*nbatches, false);
	for(size_t iim=0;iim< m*nbatches; ++iim) {
		// pick random image...
		int img_index = rand()%imgs.size(); // in the range 0 to imgs.size()
		// now pick in the range between buffer and img_size-buffer-dim
		const Matrix& img = *(imgs[img_index]);
		int upper = img_size-buffer-dim;
		int lower = buffer;
		int rr_pos = rand()%(upper-lower)+lower;
		int rc_pos = rand()%(upper-lower)+lower;
		int r_in_x=0;
		for(size_t c=0;c<dim;++c) {
			for(size_t r=0;r<dim;r++) {
				set_val(bigX,r_in_x++,iim,get_val(img,rr_pos+r,rc_pos+c));
			}
		}
	}
	for(size_t i=0;i<imgs.size();++i) {
		freeup(*imgs[i]);
		delete imgs[i];
	}
}


void run_expt_together(const Matrix& bigX, const Options& opts)
{
	int k = opts.k;
	int m = opts.mb;
	int n = opts.n;
	int nbatches = opts.nb;
	cerr << "basis_iters: " << opts.nbasis_iters << endl;
	cerr << "eta: " << opts.eta << endl;
	cerr << "beta: " << opts.beta << endl;
	cerr << "sigma: " << opts.sigma << endl;
	cerr << "k: " << k << endl;
	cerr << "m: " << m << endl;
	cerr << "n: " << n << endl;
	cerr << "nb: " << nbatches << endl;
	cublasInit();
	gpu::checkErrors();
	float gamma = 2*opts.sigma*opts.sigma*opts.beta;
	cerr << "gamma: " << gamma << endl;
	float *B_on_dev, *BtB_on_dev, *X_on_dev, *XtB_on_dev, *S_on_dev;
	onetime_setup(k,m,n,gamma,
			&B_on_dev, &BtB_on_dev, &X_on_dev, 
			&XtB_on_dev, &S_on_dev);
	gpu::checkErrors();
	Matrix B;
	init(B, k, n, false);
	for(int i=0;i<k;i++) {
          for(int j=0;j<n;j++) {
            set_val(B,i,j,((float)rand()/(float)RAND_MAX)-0.5);
	  }
        }
	for(int j=0;j<n;j++) {
          float col_avg = 0;
          for(int i=0;i<k;i++) {
            col_avg += get_val(B,i,j);
          }
          col_avg /= (float)(k);
          for(int i=0;i<k;i++) {
            set_val(B,i,j,get_val(B,i,j)-col_avg);
          }
          float col_norm = 0;
          for(int i=0;i<k;i++) {
            col_norm += get_val(B,i,j)*get_val(B,i,j);
          }
          col_norm = sqrt(col_norm);
          for(int i=0;i<k;i++) {
            set_val(B,i,j,get_val(B,i,j)/col_norm);
          }
	}
	cutilSafeCall(cudaMemcpy(B_on_dev,B.values,k*n*sizeof(float),cudaMemcpyHostToDevice));
	gpu::checkErrors();

	float *SSt2_on_dev, *XSt2_on_dev, *G_on_dev, *X_BS_on_dev;
	onetime_setup_pg(k, m, n, &SSt2_on_dev,&XSt2_on_dev,&G_on_dev,&X_BS_on_dev);
	gpu::checkErrors(); gpu::checkCublasError();
	
	std::vector<Matrix*> ss;
	cerr << "iepoch, fobj, coeffs_time, basis_time, avg_nnz\n";
	for(int iepoch=0;iepoch<opts.nepoch; iepoch++) {
		float fobj=0;
		float coeff_time=0, basis_time=0;
		int nonzeros = 0;
		for(int ibatch=0;ibatch<nbatches;++ibatch) {
			Matrix X;
			X.row_contiguous = false;
			X.num_ptrs = m;
			X.num_vals = k;
			X.values = bigX.values+(ibatch*m*k);
			Matrix* S=0;
			if(ss.size()<(ibatch+1)) {
				S = new Matrix(); 
				init(*S, n, m, false);
				ss.push_back(S);
				for(int i=0;i<n;i++) {
					for(int j=0;j<m;j++) {
						set_val(*S,i,j,0);
					}
				}
			} else {
                // Reusing S for batch...
				S = ss[ibatch];
			}
			// Repeat until convergence of original cost function:
			// load S, X
			cutilSafeCall(cudaMemcpy(S_on_dev,S->values,
					n*m*sizeof(float),cudaMemcpyHostToDevice));
			cutilSafeCall(cudaMemcpy(X_on_dev,X.values,
					k*m*sizeof(float),cudaMemcpyHostToDevice));
			gpu::checkErrors(); gpu::checkCublasError();
			// 3) solve for coefficients using fixed B
			float l1ls_time=0;
			{
				gpu::GpuEventTimer timer(l1ls_time);
				l1ls_coord_descent_cu_basic(k, m, n, B_on_dev,
						BtB_on_dev, X_on_dev, XtB_on_dev, S_on_dev);
				gpu::checkErrors(); gpu::checkCublasError();
			}
			coeff_time +=l1ls_time;

			cutilSafeCall(cudaMemcpy((void *)(S->values),(const void *)(S_on_dev),
					n*m*sizeof(float),cudaMemcpyDeviceToHost));
			gpu::checkErrors(); gpu::checkCublasError();
		        fobj += calc_objective(k,m,n,opts.sigma,opts.beta,B_on_dev,S_on_dev,X_on_dev,X_BS_on_dev);
			// 4) solve for basis using fixed S
			float b_time=0;
			{
				gpu::GpuEventTimer timer(b_time);
				float tmp = proj_grad_descent_cu_basic
				(opts.c,opts.sigma, opts.eta, opts.beta, opts.nbasis_iters,
						k,m,n,
						B_on_dev,X_on_dev,
						S_on_dev,SSt2_on_dev,XSt2_on_dev,G_on_dev,X_BS_on_dev);
				gpu::checkErrors(); gpu::checkCublasError();
			}

			basis_time +=b_time;
			nonzeros += nnz(*S);
		}
		cerr << iepoch+1 << ", " << fobj/((float)(nbatches*m)) << ", " << 
		coeff_time/1000.0 << ", " << basis_time/1000.0 << ", " <<
		(float)nonzeros/(float)(nbatches*m*n) << endl;
		cutilSafeCall(cudaMemcpy((void *)(B.values),(const void *)(B_on_dev),
				k*n*sizeof(float),cudaMemcpyDeviceToHost));	
		write_basis(opts,"",B,iepoch);
	}
	cutilSafeCall(cudaMemcpy((void *)(B.values),(const void *)(B_on_dev),
			k*n*sizeof(float),cudaMemcpyDeviceToHost));	
	onetime_teardown(B_on_dev, BtB_on_dev, X_on_dev, XtB_on_dev, S_on_dev);
	onetime_teardown_pg(SSt2_on_dev, XSt2_on_dev, G_on_dev, X_BS_on_dev);

	cublasShutdown();
	for(size_t i=0;i<ss.size();++i) {
		freeup(*ss[i]);
		delete ss[i];
	}
}

void run_expt_on_natural_images(const Options& opts) 
{
	cerr << "Getting basis from natural images\n";
	Matrix bigX;
	get_images_as_input_matrix(bigX,opts);
	run_expt_together(bigX,opts);
	freeup(bigX);
}

void get_digits_as_input_matrix(Matrix& bigX, const Options& opts)
{
	cerr << "Reading " << opts.filename << endl;
	FILE* inf = fopen(opts.filename,"rt");
	if(inf==0) {
		std::cerr << "Cannot open file\n";
		return;
	}
	read_matrix(inf,196,60000,bigX,false);
}

void run_expt_on_digits(const Options& opts)
{
	cerr << "Getting basis from digits\n";
	Matrix bigX;
	get_digits_as_input_matrix(bigX,opts);
	run_expt_together(bigX,opts);
	freeup(bigX);
}

void get_letters_as_input_matrix(Matrix& bigX, const Options& opts)
{
	cerr << "Reading " << opts.filename << endl;
	FILE* inf = fopen(opts.filename,"rt");
	if(inf==0) {
		std::cerr << "Cannot open file\n";
		return;
	}
	read_matrix(inf,196,52152,bigX,false);
}

void get_letters_labels(Matrix& labels, const Options& opts)
{
	cerr << "Reading letters label file: " << opts.labelfile << endl;
	if(opts.labelfile==0) {
		cerr << "Specify letters label file\n";
		return;
	}
	FILE* inf = fopen(opts.labelfile,"rt");
	if(inf==0) {
		std::cerr << "Cannot open file\n";
		return;
	}
	read_matrix(inf,52152,1,labels,false);
}

void read_basis(Matrix& B, const Options& opts) 
{
	cerr << "Reading basis file: " << opts.basisfile << endl;
	if(opts.basisfile==0) {
		cerr << "Please specify basis file for coefficient computation\n";
		return;
	}
	FILE* inf = fopen(opts.basisfile,"rt");
	if(inf==0) {
		std::cerr << "Cannot open file\n";
		return;
	}
	read_matrix(inf,opts.k,opts.n,B,false);
}

void write_svm_light_data(string filename, const Matrix& X, const Matrix& labels) 
{
	// write in svmlight recognizable format:
	FILE* inf = fopen(filename.c_str(),"w");
	for(size_t ic=0; ic<num_cols(X); ic++) {
//		printf("Label: %g\n",get_val(labels,ic,0)+1);
		fprintf(inf,"%g",get_val(labels,ic,0)+1); 
		for(size_t ir=0;ir<num_rows(X); ir++) {
			float val = get_val(X,ir,ic);
			if(val>1e-14) {
				// svmlight wants >=1 feature values.
				fprintf(inf," %d:%g",ir+1,val);
			}
		}
		fprintf(inf,"\n");
	}
	fclose(inf);
}

void write_test_train_data(const Matrix& testX, const Matrix& testLabels,
		const Matrix& trainX,const Matrix& trainLabels, const Options& opts)
{
	cerr << "Writing train.dat\n";
	write_svm_light_data("train.dat",trainX,trainLabels);
	cerr << "Writing test.dat\n";
	write_svm_light_data("test.dat",testX,testLabels);
}

int randint(int l, int u) 
{
	return l + rand()%(u-l+1);
}

void partition_into_train_test(Matrix& trainX, Matrix& trainLabels,
		Matrix& testX, Matrix& testLabels,
		const Matrix& inp, const Matrix& inpl, 
		const Options& opts)
{
	//	cerr << "RANDMAX: " << RAND_MAX << endl;
	// go through matrix and pick randomly 100 columns...
	int numtrain=opts.numtrain;
	int m = num_cols(inp);
	cerr << "Paritionning into " << numtrain << " train and " << m-numtrain << " test\n";
	int k = num_rows(inp);
	set<int> traincols;
	while(traincols.size()<numtrain) {
		int random = randint(0,m-1);
		traincols.insert(random);
	}
	init(trainX,k,numtrain,false);
	init(testX,k,m-numtrain,false);
	init(trainLabels,numtrain,1,false);
	init(testLabels,m-numtrain,1,false);
	int ictrain=0, ictest=0;
	for(size_t ic=0; ic<m; ++ic) {
		if(traincols.find(ic)!=traincols.end()) {
			for(size_t ir=0; ir<k; ++ir) {
				set_val(trainX,ir,ictrain,get_val(inp,ir,ic));
			}
			set_val(trainLabels,ictrain,0,get_val(inpl,ic,0));
			ictrain++;
		}
		else {
			for(size_t ir=0; ir<k; ++ir) {
				float val = get_val(inp,ir,ic);
				set_val(testX,ir,ictest,val);
			}
			set_val(testLabels,ictest,0,get_val(inpl,ic,0));
			ictest++;
		}
	}
}

void find_coeffs(Matrix& S, const Matrix& X, const Matrix& B, const Options& opts) 
{
	cerr << "Finding coefficients for given inputs\n";
	int m = num_cols(X);
	int n = num_cols(B);
	int k = num_rows(X);
	float gamma = 2*opts.sigma*opts.sigma*opts.beta;
	cerr << "k: " << k << endl;
	cerr << "m: " << m << endl;
	cerr << "n: " << n << endl;
	cerr << "gamma: " << gamma << endl;
	init(S,n,m,false);
	std::string device_name = gpu::initialize_device(opts.device); 
	float expt_time=0;
	{
		gpu::GpuEventTimer timer(expt_time);
		l1ls_coord_descent_cu(S, gamma, B, X);
		gpu::checkErrors(); 
	}
}

void find_coeffs_for_letters(const Options& opts)
{
	cerr << "Getting coeffs for letters\n";
	Matrix inp, inpl;
	get_letters_as_input_matrix(inp,opts);
	get_letters_labels(inpl, opts);
	
	Matrix testX, trainX, testLabels, trainLabels;
	partition_into_train_test(trainX,trainLabels,testX,testLabels,inp,inpl,opts);
	
	Matrix B,S;
	read_basis(B,opts);
	
	find_coeffs(S,trainX,B,opts);
	
	write_test_train_data(testX,testLabels,trainX,trainLabels,opts);

	freeup(testX);
	freeup(trainX);
	freeup(inp);
	freeup(testLabels);
	freeup(trainLabels);
	freeup(inpl);
	freeup(B);
	freeup(S);
}

int main(int argc, const char** argv)
{
	Options opts;
	read_options(argc,argv,opts);
	if(opts.runwhat==0) {
		cerr << "Run what?\n";
		print_usage(argv);
	    exit(1); 
	}
	if(strcmp(opts.runwhat,"coeffs")==0) {
		cerr << "Just coeffs...\n";
		main_coeffs(argc,argv);
		return 0;
	}
	else if(strcmp(opts.runwhat,"natural")==0) {
		cerr << "On natural images...\n";
		if(opts.imagesdir==0) {
			print_usage(argv);
			std::cerr << "Specify input images directory\n";
			return 0;
		}
		if(opts.basisdir==0) {
			print_usage(argv);
			std::cerr << "Specify output basis directory\n";
			return 0;
		}
		cerr << "Writing to basis dir: " << opts.basisdir << endl;
		std::string device_name = gpu::initialize_device(opts.device); 
		cerr << "Initialized device: " << device_name<<endl;
		run_expt_on_natural_images(opts);
		return 0;
	} else if(strcmp(opts.runwhat,"digits")==0) {
		cerr << "On digits...\n";
		if(opts.filename==0) {
			print_usage(argv);
			std::cerr << "Cannot open file\n";
			return 0;
		}
		if(opts.basisdir==0) {
			print_usage(argv);
			std::cerr << "Specify output basis directory\n";
			return 0;
		}
		cerr << "Writing to basis dir: " << opts.basisdir << endl;
		run_expt_on_digits(opts);
		return 0;
	} else if(strcmp(opts.runwhat,"basis")==0) {
		cerr << "Just basis...\n";
		main_basis(argc,argv);
	} else if(strcmp(opts.runwhat,"findcoeffs")==0) {
		cerr << "Finding coefficients\n";
		find_coeffs_for_letters(opts);
		return 0;
	} 
	return 0;
}
