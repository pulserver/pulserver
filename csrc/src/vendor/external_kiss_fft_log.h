/*
 *  Copyright (c) 2003-2010, Mark Borgerding. All rights reserved.
 *  This file is part of KISS FFT - https://github.com/mborgerding/kissfft
 *
 *  SPDX-License-Identifier: BSD-3-Clause
 *  See COPYING file for more information.
 */

#ifndef external_kiss_fft_log_h
#define external_kiss_fft_log_h

/* Avoid conflict with VxWorks ERROR macro */
#ifdef ERROR
#undef ERROR
#endif

#define ERROR 1
#define WARNING 2
#define INFO 3
#define DEBUG 4

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#if defined(NDEBUG)
# define KISS_FFT_LOG_MSG  (void)sizeof
#else
# define KISS_FFT_LOG_MSG  (void)sizeof
#endif

#define KISS_FFT_ERROR   KISS_FFT_LOG_MSG
#define KISS_FFT_WARNING KISS_FFT_LOG_MSG
#define KISS_FFT_INFO    KISS_FFT_LOG_MSG
#define KISS_FFT_DEBUG   KISS_FFT_LOG_MSG



#endif /* external_kiss_fft_log_h */