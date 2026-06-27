#pragma once

// CUDA 12.x headers declare a few C23 math names before glibc's math.h.
// With glibc 2.42, _GNU_SOURCE also enables those C23 prototypes and nvcc
// sees mismatched exception specifications.  setup.py includes this header
// after nvcc's implicit cuda_runtime.h include and after disabling
// _GNU_SOURCE for CUDA translation units.  Preloading c++config here lets us
// disable libstdc++'s pthread clockwait fast paths, which otherwise require
// GNU-only pthread prototypes hidden when _GNU_SOURCE is disabled.
#include <bits/c++config.h>

#undef _GLIBCXX_USE_PTHREAD_COND_CLOCKWAIT
#undef _GLIBCXX_USE_PTHREAD_MUTEX_CLOCKLOCK
#undef _GLIBCXX_USE_PTHREAD_RWLOCK_CLOCKLOCK
