/* Author: Anand Madhavan */
#ifndef __UTILS_H__
#define __UTILS_H__

#include "cutil_inline.h"
#include <string>
#include "coreutils.hh"

namespace cpu {
class CpuEventTimer {
  float& _time;
  unsigned int timer;
  public:
  CpuEventTimer(float& time):_time(time), timer(0) { 
    cutilCheckError( cutCreateTimer(&timer) );
    cutilCheckError( cutStartTimer(timer) );
  }
  ~CpuEventTimer() { 
    cutilCheckError( cutStopTimer(timer) );
    _time = cutGetTimerValue(timer);
    cutilCheckError(cutDeleteTimer(timer));
  }
};
}

#endif
