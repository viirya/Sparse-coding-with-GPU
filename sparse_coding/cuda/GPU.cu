/* Author: Anand Madhavan */

#include "GPU.hh"
#include <iostream>
#include "cublas.h"

extern bool g_verbose;

void gpu::checkCublasError() {
	if(cublasGetError()!=CUBLAS_STATUS_SUCCESS) {
		std::cerr << "Cublas error, exiting\n";
		exit(-1);
	}
}

bool gpu::checkErrors() {
	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err) {
		fprintf(stderr, "GPU error: %s.\n", cudaGetErrorString( err) );
		return false; 
	}
	return true;
}

bool gpu::get_device_infos(int device,
		cudaDeviceProp& prop)
{
	if(device==-1) {
		return true;
	}
	int numdevices;
	CUDA_SAFE_CALL(cudaGetDeviceCount(&numdevices));
	if(numdevices==0 && device!=-1) {
		std::cerr << "No GPU devices!";
		return false;
	}
	if(device>=numdevices) {
		device = 0;
		std::cerr << "No device like that, defaulting to 0: ";
		return false;
	} 
	cudaDeviceProp props;
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&props, device));
	prop = props;
	return true;	
}

bool initialize_gpu_device(int device /*0, 1 etc*/, 
		std::string& name)
{
  if(device==-1) {
    name = "CPU-1-Core";
    return true;
  }
  int numdevices;
  CUDA_SAFE_CALL(cudaGetDeviceCount(&numdevices));
  if(numdevices==0 && device!=-1) {
    std::cerr << "No GPU devices!";
    return false;
  }
//  DEBUGIFY(std::cerr << "Initializing device...");
  if(device>=numdevices) {
    device = 0;
    std::cerr << "No device like that, defaulting to 0: ";
  } 
  cudaDeviceProp props;
  CUDA_SAFE_CALL(cudaGetDeviceProperties(&props, device));
  name = props.name;
  CUDA_SAFE_CALL(cudaSetDevice(device));
  //  DEBUGIFY(std::cerr << "Done." << std::endl;);
  return true;

}

std::string gpu::initialize_device(int device) {
  std::string device_name = "CPU-1-Core";
  if(device>-1) {
    if(!initialize_gpu_device(device,device_name)) {
      std::cerr << "ERROR: can't initialize gpu device\n";
      exit(1);
    }
  }
  return device_name;
}

void gpu::update_min_max_stats(const Stats& curr, Stats& min, Stats& max)
{
#define STATS_UPDATE_MAX(a)  if(curr.a>max.a) max.a = curr.a; if(curr.a<min.a) min.a = curr.a;
  STATS_UPDATE_MAX(compute_time);
  STATS_UPDATE_MAX(total_time);
  STATS_UPDATE_MAX(host_dev_global_transfer_time);
  STATS_UPDATE_MAX(dev_host_transfer_time);
  STATS_UPDATE_MAX(host_dev_mem_type_transfer_time);
}

