#ifndef __CORE_UTILS_H__
#define __CORE_UTILS_H__

extern bool g_verbose;

#ifdef DEBUG
#define DEBUGIFY(x) x
#else
#define DEBUGIFY(x) if(g_verbose) x
#endif

#endif
