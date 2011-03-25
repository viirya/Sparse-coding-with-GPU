/* Author: Anand Madhavan */
#ifndef __GPU_H__
#define __GPU_H__

#include "cutil.h"
#include <string>
#include "utils.hh"
#include "Stats.hh"

namespace gpu {
	bool get_device_infos(int device, cudaDeviceProp& prop);
class GpuEventTimer {
  float& _time;
  cudaEvent_t _start_event;
  cudaEvent_t _stop_event;
  public:
  GpuEventTimer(float& time):_time(time) { 
    CUDA_SAFE_CALL( cudaEventCreate(&_start_event) );
    CUDA_SAFE_CALL( cudaEventCreate(&_stop_event) );

    CUDA_SAFE_CALL( cudaEventRecord(_start_event, 0) );
  }
  ~GpuEventTimer() { 
    CUDA_SAFE_CALL( cudaEventRecord(_stop_event, 0) );
    CUDA_SAFE_CALL( cudaEventSynchronize(_stop_event) );
    CUDA_SAFE_CALL( cudaEventElapsedTime(&_time, _start_event, _stop_event) );

    CUDA_SAFE_CALL( cudaEventDestroy(_stop_event) );
    CUDA_SAFE_CALL( cudaEventDestroy(_start_event) );
  }
};
  std::string initialize_device(int device); 
  void update_min_max_stats(const Stats& curr, Stats& min, Stats& max);
  bool checkErrors();
  void checkCublasError();
}

#endif
