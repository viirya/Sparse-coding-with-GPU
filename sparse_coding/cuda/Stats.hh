/* Author: Anand Madhavan */
#ifndef __STATS_HH__
#define __STATS_HH__

struct Stats {
  Stats(float initt=0):compute_time(initt), total_time(initt),
  host_dev_global_transfer_time(initt), host_dev_global_bytes_transfered(0),
  dev_host_transfer_time(initt), dev_host_bytes_transfered(0),
  host_dev_mem_type_transfer_time(initt), host_dev_mem_type_bytes_transfered(0)
  {}
  const Stats& operator+(const Stats& with) { if(&with==this) return *this;
#define STATS_SUM(a) (a)+=(with).a;
    STATS_SUM(compute_time);
    STATS_SUM(total_time);
    STATS_SUM(host_dev_global_transfer_time);
    STATS_SUM(host_dev_global_bytes_transfered);
    STATS_SUM(dev_host_transfer_time);
    STATS_SUM(dev_host_bytes_transfered);
    STATS_SUM(host_dev_mem_type_transfer_time);
    STATS_SUM(host_dev_mem_type_bytes_transfered);
    return *this;
  }
  const Stats& operator/(int den) { if(den<=1) return *this;
#define STATS_DIV(a) (a)/=den;
    STATS_DIV(compute_time);
    STATS_DIV(total_time);
    STATS_DIV(host_dev_global_transfer_time);
    STATS_DIV(host_dev_global_bytes_transfered);
    STATS_DIV(dev_host_transfer_time);
    STATS_DIV(dev_host_bytes_transfered);
    STATS_DIV(host_dev_mem_type_transfer_time);
    STATS_DIV(host_dev_mem_type_bytes_transfered);
    return *this;
  }
  float compute_time; // in ms? pure decode time, either cpu or gpu
  float total_time; // in ms? total cpu time for decoding

  float host_dev_global_transfer_time;
  unsigned int host_dev_global_bytes_transfered;

  float dev_host_transfer_time; // from global
  unsigned int dev_host_bytes_transfered; // from global

  float host_dev_mem_type_transfer_time; // from host to specified mem type on device
  unsigned int host_dev_mem_type_bytes_transfered;
  
};

#endif
