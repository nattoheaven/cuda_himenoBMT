/**
 * himenoBMTxps.gpudirect
 * Copyright (C)
 *  2012 NISHIMURA Ryohei
 *  2001 Originally written by Ryutaro HIMENO
 * All rights reserved.
 *
 * This software is licensed with GNU Lesser General Public License,
 * Version 2 or any later version.
 */

/********************************************************************

 This benchmark test program is measuring a cpu performance
 of floating point operation by a Poisson equation solver.

 If you have any question, please ask me via email.
 written by Ryutaro HIMENO, November 26, 2001.
 Version 3.0
 ----------------------------------------------
 Ryutaro Himeno, Dr. of Eng.
 Head of Computer Information Division,
 RIKEN (The Institute of Pysical and Chemical Research)
 Email : himeno@postman.riken.go.jp
 ---------------------------------------------------------------
 You can adjust the size of this benchmark code to fit your target
 computer. In that case, please chose following sets of
 (mimax,mjmax,mkmax):
 small : 33,33,65
 small : 65,65,129
 midium: 129,129,257
 large : 257,257,513
 ext.large: 513,513,1025
 This program is to measure a computer performance in MFLOPS
 by using a kernel which appears in a linear solver of pressure
 Poisson eq. which appears in an incompressible Navier-Stokes solver.
 A point-Jacobi method is employed in this solver as this method can 
 be easyly vectrized and be parallelized.
 ------------------
 Finite-difference method, curvilinear coodinate system
 Vectorizable and parallelizable on each grid point
 No. of grid points : imax x jmax x kmax including boundaries
 ------------------
 A,B,C:coefficient matrix, wrk1: source term of Poisson equation
 wrk2 : working area, OMEGA : relaxation parameter
 BND:control variable for boundaries and objects ( = 0 or 1)
 P: pressure
********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sys/time.h>

#ifdef SSMALL
#define MIMAX            33
#define MJMAX            33
#define MKMAX            65
#endif

#ifdef SMALL
#define MIMAX            65
#define MJMAX            65
#define MKMAX            129
#endif

#ifdef MIDDLE
#define MIMAX            129
#define MJMAX            129
#define MKMAX            257
#endif

#ifdef LARGE
#define MIMAX            257
#define MJMAX            257
#define MKMAX            513
#endif

#ifdef ELARGE
#define MIMAX            513
#define MJMAX            513
#define MKMAX            1025
#endif

double second();
float jacobi(int);
void initmt();
double fflop(int,int,int);
double mflops(int,double,double);

static int imax, jmax, kmax;

#define NTHREADS (256)
#define BLOCKI (4)
#define BLOCKJ (8)
#define BLOCKK (32)
#define BLOCKJK (BLOCKJ * BLOCKK)
#define BLOCKIJK (BLOCKI * BLOCKJK)

#define NBLOCKSI ((MIMAX - 1) / BLOCKI)
#define NBLOCKSJ ((MJMAX - 1) / BLOCKJ)
#define NBLOCKSK ((MKMAX - 1) / BLOCKK)

static int ngpus, *gpus, *ibounds;

static float (**d_p)[MJMAX - 1][MKMAX + 31];
static const float (**d_pprev)[MKMAX + 31], (**d_pnext)[MKMAX + 31];
static float (**d_a0)[MJMAX - 1][MKMAX + 31];
static float (**d_a1)[MJMAX - 1][MKMAX + 31];
static float (**d_a2)[MJMAX - 1][MKMAX + 31];
static float (**d_a3)[MJMAX - 1][MKMAX + 31];
static float (**d_b0)[MJMAX - 1][MKMAX + 31];
static float (**d_b1)[MJMAX - 1][MKMAX + 31];
static float (**d_b2)[MJMAX - 1][MKMAX + 31];
static float (**d_c0)[MJMAX - 1][MKMAX + 31];
static float (**d_c1)[MJMAX - 1][MKMAX + 31];
static float (**d_c2)[MJMAX - 1][MKMAX + 31];
static float (**d_bnd)[MJMAX - 1][MKMAX + 31];
static float (**d_wrk1)[MJMAX - 1][MKMAX + 31];
static float (**d_wrk2)[MJMAX - 1][MKMAX + 31];
static const float (**d_wrk2prev)[MKMAX + 31], (**d_wrk2next)[MKMAX + 31];
static float h_gosa[NBLOCKSI][NBLOCKSJ];
static float (**d_gosa)[NBLOCKSJ];

#define CUDACHECK(E) do {                                       \
    cudaError_t e = (E);                                        \
    if (e != cudaSuccess) {                                     \
      fprintf(stderr, "CUDA Error: " __FILE__ ": %d: %s\n",     \
              __LINE__, cudaGetErrorString(e));                 \
      exit(-1);                                                 \
    }                                                           \
  } while (0);

__global__ void
initmt_kernel0(float (*a)[MJMAX - 1][MKMAX + 31], float value)
{
  for (int ijk = threadIdx.x; ijk < BLOCKIJK; ijk += NTHREADS) {
    int i = blockIdx.y * BLOCKI + ijk / BLOCKJK;
    int j = (blockIdx.x / NBLOCKSK) * BLOCKJ + (ijk % BLOCKJK) / BLOCKK;
    int k = (blockIdx.x % NBLOCKSK) * BLOCKK + ijk % BLOCKK;
    a[i][j][k] = value;
  }
}

__global__ void
initmt_kernel1(float (*p)[MJMAX - 1][MKMAX + 31], int ibegin)
{
  for (int ijk = threadIdx.x; ijk < BLOCKIJK; ijk += NTHREADS) {
    int i = blockIdx.y * BLOCKI + ijk / BLOCKJK;
    int j = (blockIdx.x / NBLOCKSK) * BLOCKJ + (ijk % BLOCKJK) / BLOCKK;
    int k = (blockIdx.x % NBLOCKSK) * BLOCKK + ijk % BLOCKK;
    p[i][j][k] = (float) ((i + ibegin) * (i + ibegin)) /
      (float) ((MIMAX - 1) * (MIMAX - 1));
  }
}

template<bool HAS_PREV, bool HAS_NEXT>
__device__ float
jacobi_sub(float (*wrk2)[MJMAX - 1][MKMAX + 31],
           const float (*p)[MJMAX - 1][MKMAX + 31],
           const float (*pprev)[MKMAX + 31], const float (*pnext)[MKMAX + 31],
           const float (*a0)[MJMAX - 1][MKMAX + 31],
           const float (*a1)[MJMAX - 1][MKMAX + 31],
           const float (*a2)[MJMAX - 1][MKMAX + 31],
           const float (*a3)[MJMAX - 1][MKMAX + 31],
           const float (*b0)[MJMAX - 1][MKMAX + 31],
           const float (*b1)[MJMAX - 1][MKMAX + 31],
           const float (*b2)[MJMAX - 1][MKMAX + 31],
           const float (*c0)[MJMAX - 1][MKMAX + 31],
           const float (*c1)[MJMAX - 1][MKMAX + 31],
           const float (*c2)[MJMAX - 1][MKMAX + 31],
           const float (*bnd)[MJMAX - 1][MKMAX + 31],
           const float (*wrk1)[MJMAX - 1][MKMAX + 31],
           float (*s_p)[BLOCKJ + 2][BLOCKK + 2])
{
  const float omega = 0.8f;
  for (int tidij = threadIdx.x; tidij < (BLOCKI + 2) * (BLOCKJ + 2);
       tidij += NTHREADS) {
    int tidi = tidij / (BLOCKJ + 2);
    int tidj = tidij % (BLOCKJ + 2);
    int i = blockIdx.y * BLOCKI + tidi - 1;
    int j = blockIdx.x * BLOCKJ + tidj - 1;
    if (j >= 0 && j < MJMAX - 1) {
      const float (*pp)[MKMAX + 31];
      if (HAS_PREV && i < 0) {
        pp = pprev;
      } else if (HAS_NEXT && i >= gridDim.y * BLOCKI) {
        pp = pnext;
      } else {
        pp = p[i];
      }
      s_p[tidi][tidj][1] = pp[j][0];
    }
  }
  int tidk = threadIdx.x % BLOCKK;
  float gosa = 0.0f;
  for (int k0 = 0; k0 < MKMAX - 1; k0 += BLOCKK) {
    int k = k0 + tidk;
    for (int tidij = threadIdx.x / BLOCKK; tidij < (BLOCKI + 2) * (BLOCKJ + 2);
         tidij += NTHREADS / BLOCKK) {
      int tidi = tidij / (BLOCKJ + 2);
      int tidj = tidij % (BLOCKJ + 2);
      int i = blockIdx.y * BLOCKI + tidi - 1;
      int j = blockIdx.x * BLOCKJ + tidj - 1;
      if (j >= 0 && j < MJMAX - 1) {
        const float (*pp)[MKMAX + 31];
        if (HAS_PREV && i < 0) {
          pp = pprev;
        } else if (HAS_NEXT && i >= gridDim.y * BLOCKI) {
          pp = pnext;
        } else {
          pp = p[i];
        }
        if (k + 1 < MKMAX - 1) {
          s_p[tidi][tidj][tidk + 2] = pp[j][k + 1];
        }
      }
    }
    __syncthreads();
    for (int tidij = threadIdx.x / BLOCKK; tidij < BLOCKI * BLOCKJ;
         tidij += NTHREADS / BLOCKK) {
      int tidi = tidij / BLOCKJ;
      int tidj = tidij % BLOCKJ;
      int i = blockIdx.y * BLOCKI + tidi;
      int j = blockIdx.x * BLOCKJ + tidj;
      if ((HAS_PREV || i != 0) && (HAS_NEXT || i != gridDim.y * BLOCKI - 1) &&
          j != 0 && j != MJMAX - 2 && k != 0 && k != MKMAX - 2) {
        int tidii = tidi + 1;
        int tidjj = tidj + 1;
        int tidkk = tidk + 1;
        float s0 =
          a0[i][j][k] * s_p[tidii + 1][tidjj][tidkk]
          + a1[i][j][k] * s_p[tidii][tidjj + 1][tidkk]
          + a2[i][j][k] * s_p[tidii][tidjj][tidkk + 1]
          + b0[i][j][k] * ( s_p[tidii + 1][tidjj + 1][tidkk]
                            - s_p[tidii + 1][tidjj - 1][tidkk]
                            - s_p[tidii - 1][tidjj + 1][tidkk]
                            + s_p[tidii - 1][tidjj - 1][tidkk] )
          + b1[i][j][k] * ( s_p[tidii][tidjj + 1][tidkk + 1]
                            - s_p[tidii][tidjj - 1][tidkk + 1]
                            - s_p[tidii][tidjj + 1][tidkk - 1]
                            + s_p[tidii][tidjj - 1][tidkk - 1] )
          + b2[i][j][k] * ( s_p[tidii + 1][tidjj][tidkk + 1]
                            - s_p[tidii - 1][tidjj][tidkk + 1]
                            - s_p[tidii + 1][tidjj][tidkk - 1]
                            + s_p[tidii - 1][tidjj][tidkk - 1] )
          + c0[i][j][k] * s_p[tidii - 1][tidjj][tidkk]
          + c1[i][j][k] * s_p[tidii][tidjj - 1][tidkk]
          + c2[i][j][k] * s_p[tidii][tidjj][tidkk - 1]
          + wrk1[i][j][k];
        float ss =
          ( s0 * a3[i][j][k] - s_p[tidii][tidjj][tidkk] ) * bnd[i][j][k];
        gosa += ss * ss;
        wrk2[i][j][k] = s_p[tidii][tidjj][tidkk] + omega * ss;
      }
    }
    __syncthreads();
    for (int tidij = threadIdx.x / 2; tidij < (BLOCKI + 2) * (BLOCKJ + 2);
         tidij += NTHREADS / 2) {
      int tidi = tidij / (BLOCKJ + 2);
      int tidj = tidij % (BLOCKJ + 2);
      s_p[tidi][tidj][threadIdx.x % 2] =
        s_p[tidi][tidj][threadIdx.x % 2 + BLOCKK];
    }
    __syncthreads();
  }
  return gosa;
}

template<int I> __device__ void
calc_gosa(float &gosa, float gosai, float *s_gosa)
{
  if (I >= 32) {
    __syncthreads();
  }
  if (threadIdx.x < I) {
    gosai += s_gosa[threadIdx.x + I];
    s_gosa[threadIdx.x] = gosai;
  }
  calc_gosa<I / 2>(gosa, gosai, s_gosa);
}

template<> __device__ void
calc_gosa<1>(float &gosa, float gosai, float *s_gosa)
{
  if (threadIdx.x < 1) {
    gosai += s_gosa[threadIdx.x + 1];
    gosa = gosai;
  }
}

template<bool HAS_PREV, bool HAS_NEXT>
__global__ void
jacobi_kernel0(float (*wrk2)[MJMAX - 1][MKMAX + 31],
               const float (*p)[MJMAX - 1][MKMAX + 31],
               const float (*pprev)[MKMAX + 31], const float (*pnext)[MKMAX + 31],
               const float (*a0)[MJMAX - 1][MKMAX + 31],
               const float (*a1)[MJMAX - 1][MKMAX + 31],
               const float (*a2)[MJMAX - 1][MKMAX + 31],
               const float (*a3)[MJMAX - 1][MKMAX + 31],
               const float (*b0)[MJMAX - 1][MKMAX + 31],
               const float (*b1)[MJMAX - 1][MKMAX + 31],
               const float (*b2)[MJMAX - 1][MKMAX + 31],
               const float (*c0)[MJMAX - 1][MKMAX + 31],
               const float (*c1)[MJMAX - 1][MKMAX + 31],
               const float (*c2)[MJMAX - 1][MKMAX + 31],
               const float (*bnd)[MJMAX - 1][MKMAX + 31],
               const float (*wrk1)[MJMAX - 1][MKMAX + 31])
{
  __shared__ float shared[(BLOCKI + 2) * (BLOCKJ + 2) * (BLOCKK + 2)];
  float (*s_p)[BLOCKJ + 2][BLOCKK + 2] =
    (float (*)[BLOCKJ + 2][BLOCKK + 2])shared;
  jacobi_sub<HAS_PREV, HAS_NEXT>(wrk2,
				 p,
				 pprev, pnext,
				 a0,
				 a1,
				 a2,
				 a3,
				 b0,
				 b1,
				 b2,
				 c0,
				 c1,
				 c2,
				 bnd,
				 wrk1,
				 s_p);
}

template<bool HAS_PREV, bool HAS_NEXT>
__global__ void
jacobi_kernel1(float (*wrk2)[MJMAX - 1][MKMAX + 31],
               float (*gosa)[NBLOCKSJ],
               const float (*p)[MJMAX - 1][MKMAX + 31],
               const float (*pprev)[MKMAX + 31], const float (*pnext)[MKMAX + 31],
               const float (*a0)[MJMAX - 1][MKMAX + 31],
               const float (*a1)[MJMAX - 1][MKMAX + 31],
               const float (*a2)[MJMAX - 1][MKMAX + 31],
               const float (*a3)[MJMAX - 1][MKMAX + 31],
               const float (*b0)[MJMAX - 1][MKMAX + 31],
               const float (*b1)[MJMAX - 1][MKMAX + 31],
               const float (*b2)[MJMAX - 1][MKMAX + 31],
               const float (*c0)[MJMAX - 1][MKMAX + 31],
               const float (*c1)[MJMAX - 1][MKMAX + 31],
               const float (*c2)[MJMAX - 1][MKMAX + 31],
               const float (*bnd)[MJMAX - 1][MKMAX + 31],
               const float (*wrk1)[MJMAX - 1][MKMAX + 31])
{
  __shared__ float shared[(BLOCKI + 2) * (BLOCKJ + 2) * (BLOCKK + 2)];
  float (*s_p)[BLOCKJ + 2][BLOCKK + 2] =
    (float (*)[BLOCKJ + 2][BLOCKK + 2])shared;
  float gosai = jacobi_sub<HAS_PREV, HAS_NEXT>(wrk2,
					       p,
					       pprev, pnext,
					       a0,
					       a1,
					       a2,
					       a3,
					       b0,
					       b1,
					       b2,
					       c0,
					       c1,
					       c2,
					       bnd,
					       wrk1,
					       s_p);
  __syncthreads();
  float *s_gosa = shared;
  s_gosa[threadIdx.x] = gosai;
  calc_gosa<NTHREADS / 2>(gosa[blockIdx.y][blockIdx.x], gosai, s_gosa);
}

static void
*mymalloc(size_t n)
{
  void *p;
  p = malloc(n);
  if (p == NULL) {
    fprintf(stderr, "malloc failed!\n");
    exit(-2);
  }
  return p;
}

static int
testdirect(int n, int igpu, int *gpus)
{
  int ngpus = 0;
  gpus[ngpus++] = igpu;
  for (int jgpu = 0; jgpu < n; ++jgpu) {
    if (jgpu != igpu) {
      int canAccessPeer;
      CUDACHECK(cudaDeviceCanAccessPeer(&canAccessPeer, gpus[ngpus - 1], jgpu));
      if (canAccessPeer) {
        gpus[ngpus++] = jgpu;
      }
    }
  }
  return ngpus;
}

static void
initcuda()
{
  int n, startgpu, *tmpgpus;
  CUDACHECK(cudaGetDeviceCount(&n));
  if (n == 0) {
    fprintf(stderr, "There are no GPUs!\n");
    exit(-3);
  }
  ngpus = 0;
  tmpgpus = (int *)mymalloc(n * n * sizeof(int));
  for (int igpu = 0; igpu < n; ++igpu) {
    int ingpus = testdirect(n, igpu, &tmpgpus[igpu * n]);
    if (ingpus > ngpus) {
      startgpu = igpu;
      ngpus = ingpus;
    }
  }
  gpus = (int *)mymalloc(ngpus * sizeof(int));
  memcpy(gpus, &tmpgpus[startgpu * n], ngpus * sizeof(int));
  printf("This benchmark uses GPU");
  for (int igpu = 0; igpu < ngpus; ++igpu) {
    printf(" #%d", gpus[igpu]);
  }
  printf(".\n");
  ibounds = (int *)mymalloc((ngpus + 1) * sizeof(int));
  for (int igpu = 0; igpu <= ngpus; ++igpu) {
    ibounds[igpu] = igpu * NBLOCKSI / ngpus;
  }
  d_p = (float (**)[MJMAX - 1][MKMAX + 31])mymalloc
    (ngpus * sizeof(float (*)[MJMAX - 1][MKMAX + 31]));
  d_pprev = (const float (**)[MKMAX + 31])mymalloc
    (ngpus * sizeof(const float (*)[MKMAX + 31]));
  d_pnext = (const float (**)[MKMAX + 31])mymalloc
    (ngpus * sizeof(const float (*)[MKMAX + 31]));
  d_a0 = (float (**)[MJMAX - 1][MKMAX + 31])mymalloc
    (ngpus * sizeof(float (*)[MJMAX - 1][MKMAX + 31]));
  d_a1 = (float (**)[MJMAX - 1][MKMAX + 31])mymalloc
    (ngpus * sizeof(float (*)[MJMAX - 1][MKMAX + 31]));
  d_a2 = (float (**)[MJMAX - 1][MKMAX + 31])mymalloc
    (ngpus * sizeof(float (*)[MJMAX - 1][MKMAX + 31]));
  d_a3 = (float (**)[MJMAX - 1][MKMAX + 31])mymalloc
    (ngpus * sizeof(float (*)[MJMAX - 1][MKMAX + 31]));
  d_b0 = (float (**)[MJMAX - 1][MKMAX + 31])mymalloc
    (ngpus * sizeof(float (*)[MJMAX - 1][MKMAX + 31]));
  d_b1 = (float (**)[MJMAX - 1][MKMAX + 31])mymalloc
    (ngpus * sizeof(float (*)[MJMAX - 1][MKMAX + 31]));
  d_b2 = (float (**)[MJMAX - 1][MKMAX + 31])mymalloc
    (ngpus * sizeof(float (*)[MJMAX - 1][MKMAX + 31]));
  d_c0 = (float (**)[MJMAX - 1][MKMAX + 31])mymalloc
    (ngpus * sizeof(float (*)[MJMAX - 1][MKMAX + 31]));
  d_c1 = (float (**)[MJMAX - 1][MKMAX + 31])mymalloc
    (ngpus * sizeof(float (*)[MJMAX - 1][MKMAX + 31]));
  d_c2 = (float (**)[MJMAX - 1][MKMAX + 31])mymalloc
    (ngpus * sizeof(float (*)[MJMAX - 1][MKMAX + 31]));
  d_bnd = (float (**)[MJMAX - 1][MKMAX + 31])mymalloc
    (ngpus * sizeof(float (*)[MJMAX - 1][MKMAX + 31]));
  d_wrk1 = (float (**)[MJMAX - 1][MKMAX + 31])mymalloc
    (ngpus * sizeof(float (*)[MJMAX - 1][MKMAX + 31]));
  d_wrk2 = (float (**)[MJMAX - 1][MKMAX + 31])mymalloc
    (ngpus * sizeof(float (*)[MJMAX - 1][MKMAX + 31]));
  d_wrk2prev = (const float (**)[MKMAX + 31])mymalloc
    (ngpus * sizeof(const float (*)[MKMAX + 31]));
  d_wrk2next = (const float (**)[MKMAX + 31])mymalloc
    (ngpus * sizeof(const float (*)[MKMAX + 31]));
  d_gosa = (float (**)[NBLOCKSJ])mymalloc
    (ngpus * sizeof(float (*)[NBLOCKSJ]));
  for (int igpu = 0; igpu < ngpus; ++igpu) {
    int threadwidth = ibounds[igpu + 1] - ibounds[igpu];
    int iwidth = threadwidth * BLOCKI;
    if (iwidth > 0) {
      float *tmp;
      CUDACHECK(cudaSetDevice(gpus[igpu]));
      CUDACHECK(cudaFuncSetCacheConfig(jacobi_kernel0<false, false>,
                                       cudaFuncCachePreferShared));
      CUDACHECK(cudaFuncSetCacheConfig(jacobi_kernel0<false, true>,
                                       cudaFuncCachePreferShared));
      CUDACHECK(cudaFuncSetCacheConfig(jacobi_kernel0<true, false>,
                                       cudaFuncCachePreferShared));
      CUDACHECK(cudaFuncSetCacheConfig(jacobi_kernel0<true, true>,
                                       cudaFuncCachePreferShared));
      CUDACHECK(cudaFuncSetCacheConfig(jacobi_kernel1<false, false>,
                                       cudaFuncCachePreferShared));
      CUDACHECK(cudaFuncSetCacheConfig(jacobi_kernel1<false, true>,
                                       cudaFuncCachePreferShared));
      CUDACHECK(cudaFuncSetCacheConfig(jacobi_kernel1<true, false>,
                                       cudaFuncCachePreferShared));
      CUDACHECK(cudaFuncSetCacheConfig(jacobi_kernel1<true, true>,
                                       cudaFuncCachePreferShared));
      if (igpu != 0) {
        CUDACHECK(cudaDeviceEnablePeerAccess(gpus[igpu - 1], 0));
      }
      if (igpu + 1 != ngpus) {
        CUDACHECK(cudaDeviceEnablePeerAccess(gpus[igpu + 1], 0));
      }
      CUDACHECK(cudaMalloc((void **)&tmp,
                           (iwidth * (MJMAX - 1) * (MKMAX + 31) + BLOCKK - 1) *
                           sizeof(float)));
      d_p[igpu] = (float (*)[MJMAX - 1][MKMAX + 31])&tmp[BLOCKK - 1];
      CUDACHECK(cudaMalloc((void **)&d_a0[igpu],
                           iwidth * (MJMAX - 1) * (MKMAX + 31) * sizeof(float)));
      CUDACHECK(cudaMalloc((void **)&d_a1[igpu],
                           iwidth * (MJMAX - 1) * (MKMAX + 31) * sizeof(float)));
      CUDACHECK(cudaMalloc((void **)&d_a2[igpu],
                           iwidth * (MJMAX - 1) * (MKMAX + 31) * sizeof(float)));
      CUDACHECK(cudaMalloc((void **)&d_a3[igpu],
                           iwidth * (MJMAX - 1) * (MKMAX + 31) * sizeof(float)));
      CUDACHECK(cudaMalloc((void **)&d_b0[igpu],
                           iwidth * (MJMAX - 1) * (MKMAX + 31) * sizeof(float)));
      CUDACHECK(cudaMalloc((void **)&d_b1[igpu],
                           iwidth * (MJMAX - 1) * (MKMAX + 31) * sizeof(float)));
      CUDACHECK(cudaMalloc((void **)&d_b2[igpu],
                           iwidth * (MJMAX - 1) * (MKMAX + 31) * sizeof(float)));
      CUDACHECK(cudaMalloc((void **)&d_c0[igpu],
                           iwidth * (MJMAX - 1) * (MKMAX + 31) * sizeof(float)));
      CUDACHECK(cudaMalloc((void **)&d_c1[igpu],
                           iwidth * (MJMAX - 1) * (MKMAX + 31) * sizeof(float)));
      CUDACHECK(cudaMalloc((void **)&d_c2[igpu],
                           iwidth * (MJMAX - 1) * (MKMAX + 31) * sizeof(float)));
      CUDACHECK(cudaMalloc((void **)&d_bnd[igpu],
                           iwidth * (MJMAX - 1) * (MKMAX + 31) * sizeof(float)));
      CUDACHECK(cudaMalloc((void **)&d_wrk1[igpu],
                           iwidth * (MJMAX - 1) * (MKMAX + 31) * sizeof(float)));
      CUDACHECK(cudaMalloc((void **)&tmp,
                           (iwidth * (MJMAX - 1) * (MKMAX + 31) + BLOCKK - 1) *
                           sizeof(float)));
      d_wrk2[igpu] = (float (*)[MJMAX - 1][MKMAX + 31])&tmp[BLOCKK - 1];
      CUDACHECK(cudaMalloc((void **)&d_gosa[igpu],
                           threadwidth * NBLOCKSJ * sizeof(float)));
    }
  }
  for (int igpu = 0; igpu < ngpus; ++igpu) {
    d_wrk2prev[igpu] = d_pprev[igpu] = NULL;
    for (int jgpu = igpu - 1; jgpu >= 0; --jgpu) {
      int jwidth = ibounds[jgpu + 1] - ibounds[jgpu];
      if (jwidth > 0) {
        d_wrk2prev[igpu] = d_pprev[igpu] = d_p[jgpu][jwidth * BLOCKI - 1];
        break;
      }
    }
    d_wrk2next[igpu] = d_pnext[igpu] = NULL;
    for (int jgpu = igpu + 1; jgpu < ngpus; ++jgpu) {
      int jwidth = ibounds[jgpu + 1] - ibounds[jgpu];
      if (jwidth > 0) {
        d_wrk2next[igpu] = d_pnext[igpu] = d_p[jgpu][0];
        break;
      }
    }
  }
}

int
main()
{
  int    nn;
  float  gosa;
  double cpu,cpu0,cpu1,flop,target;

  target= 60.0;
  imax = MIMAX-1;
  jmax = MJMAX-1;
  kmax = MKMAX-1;

  /*
   *    Initializing matrixes
   */
  initcuda();
  initmt();
  printf("mimax = %d mjmax = %d mkmax = %d\n",MIMAX, MJMAX, MKMAX);
  printf("imax = %d jmax = %d kmax =%d\n",imax,jmax,kmax);

  nn= 3;
  printf(" Start rehearsal measurement process.\n");
  printf(" Measure the performance in %d times.\n\n",nn);

  cpu0= second();
  gosa= jacobi(nn);
  cpu1= second();
  cpu= cpu1 - cpu0;

  flop= fflop(imax,jmax,kmax);
  
  printf(" MFLOPS: %f time(s): %f %e\n\n",
         mflops(nn,cpu,flop),cpu,gosa);

  nn= (int)(target/(cpu/3.0));

  printf(" Now, start the actual measurement process.\n");
  printf(" The loop will be excuted in %d times\n",nn);
  printf(" This will take about one minute.\n");
  printf(" Wait for a while\n\n");

  /*
   *    Start measuring
   */
  cpu0 = second();
  gosa = jacobi(nn);
  cpu1 = second();

  cpu= cpu1 - cpu0;
  
  printf(" Loop executed for %d times\n",nn);
  printf(" Gosa : %e \n",gosa);
  printf(" MFLOPS measured : %f\tcpu : %f\n",mflops(nn,cpu,flop),cpu);
  printf(" Score based on Pentium III 600MHz : %f\n",
         mflops(nn,cpu,flop)/82,84);
  
  return (0);
}

void
initmt()
{
  for (int igpu = 0; igpu < ngpus; ++igpu) {
    int threadwidth = ibounds[igpu + 1] - ibounds[igpu];
    int iwidth = threadwidth * BLOCKI;
    if (iwidth > 0) {
      CUDACHECK(cudaSetDevice(gpus[igpu]));
      dim3 grid(NBLOCKSJ * NBLOCKSK, threadwidth);
      initmt_kernel0<<<grid, NTHREADS>>>(d_a0[igpu], 1.0f);
      initmt_kernel0<<<grid, NTHREADS>>>(d_a1[igpu], 1.0f);
      initmt_kernel0<<<grid, NTHREADS>>>(d_a2[igpu], 1.0f);
      initmt_kernel0<<<grid, NTHREADS>>>(d_a3[igpu], (float) (1.0 / 6.0));
      initmt_kernel0<<<grid, NTHREADS>>>(d_b0[igpu], 0.0f);
      initmt_kernel0<<<grid, NTHREADS>>>(d_b1[igpu], 0.0f);
      initmt_kernel0<<<grid, NTHREADS>>>(d_b2[igpu], 0.0f);
      initmt_kernel0<<<grid, NTHREADS>>>(d_c0[igpu], 1.0f);
      initmt_kernel0<<<grid, NTHREADS>>>(d_c1[igpu], 1.0f);
      initmt_kernel0<<<grid, NTHREADS>>>(d_c2[igpu], 1.0f);
      initmt_kernel1<<<grid, NTHREADS>>>(d_p[igpu], ibounds[igpu] * BLOCKI);
      initmt_kernel0<<<grid, NTHREADS>>>(d_wrk1[igpu], 0.0f);
      initmt_kernel0<<<grid, NTHREADS>>>(d_bnd[igpu], 1.0f);
      initmt_kernel1<<<grid, NTHREADS>>>(d_wrk2[igpu], ibounds[igpu] * BLOCKI);
    }
  }
}

float
jacobi(int nn)
{
  int i,j,n;
  float gosa;

  for(n=0 ; n<nn ; ++n){
    void *swap;

    gosa = 0.0;

    for (int igpu = 0; igpu < ngpus; ++igpu) {
      int threadwidth = ibounds[igpu + 1] - ibounds[igpu];
      int iwidth = threadwidth * BLOCKI;
      if (iwidth > 0) {
        CUDACHECK(cudaSetDevice(gpus[igpu]));
        dim3 grid(NBLOCKSJ, threadwidth);
        if (n + 1 == nn) {
	  if (igpu == 0) {
	    if (igpu + 1 == ngpus) {
	      jacobi_kernel1<false, false><<<grid, NTHREADS>>>
		(d_wrk2[igpu],
		 d_gosa[igpu],
		 d_p[igpu],
		 d_pprev[igpu], d_pnext[igpu],
		 d_a0[igpu],
		 d_a1[igpu],
		 d_a2[igpu],
		 d_a3[igpu],
		 d_b0[igpu],
		 d_b1[igpu],
		 d_b2[igpu],
		 d_c0[igpu],
		 d_c1[igpu],
		 d_c2[igpu],
		 d_bnd[igpu],
		 d_wrk1[igpu]);
	    } else {
	      jacobi_kernel1<false, true><<<grid, NTHREADS>>>
		(d_wrk2[igpu],
		 d_gosa[igpu],
		 d_p[igpu],
		 d_pprev[igpu], d_pnext[igpu],
		 d_a0[igpu],
		 d_a1[igpu],
		 d_a2[igpu],
		 d_a3[igpu],
		 d_b0[igpu],
		 d_b1[igpu],
		 d_b2[igpu],
		 d_c0[igpu],
		 d_c1[igpu],
		 d_c2[igpu],
		 d_bnd[igpu],
		 d_wrk1[igpu]);
	    }
	  } else {
	    if (igpu + 1 == ngpus) {
	      jacobi_kernel1<true, false><<<grid, NTHREADS>>>
		(d_wrk2[igpu],
		 d_gosa[igpu],
		 d_p[igpu],
		 d_pprev[igpu], d_pnext[igpu],
		 d_a0[igpu],
		 d_a1[igpu],
		 d_a2[igpu],
		 d_a3[igpu],
		 d_b0[igpu],
		 d_b1[igpu],
		 d_b2[igpu],
		 d_c0[igpu],
		 d_c1[igpu],
		 d_c2[igpu],
		 d_bnd[igpu],
		 d_wrk1[igpu]);
	    } else {
	      jacobi_kernel1<true, true><<<grid, NTHREADS>>>
		(d_wrk2[igpu],
		 d_gosa[igpu],
		 d_p[igpu],
		 d_pprev[igpu], d_pnext[igpu],
		 d_a0[igpu],
		 d_a1[igpu],
		 d_a2[igpu],
		 d_a3[igpu],
		 d_b0[igpu],
		 d_b1[igpu],
		 d_b2[igpu],
		 d_c0[igpu],
		 d_c1[igpu],
		 d_c2[igpu],
		 d_bnd[igpu],
		 d_wrk1[igpu]);
	    }
	  }
        } else {
	  if (igpu == 0) {
	    if (igpu + 1 == ngpus) {
	      jacobi_kernel0<false, false><<<grid, NTHREADS>>>
		(d_wrk2[igpu],
		 d_p[igpu],
		 d_pprev[igpu], d_pnext[igpu],
		 d_a0[igpu],
		 d_a1[igpu],
		 d_a2[igpu],
		 d_a3[igpu],
		 d_b0[igpu],
		 d_b1[igpu],
		 d_b2[igpu],
		 d_c0[igpu],
		 d_c1[igpu],
		 d_c2[igpu],
		 d_bnd[igpu],
		 d_wrk1[igpu]);
	    } else {
	      jacobi_kernel0<false, true><<<grid, NTHREADS>>>
		(d_wrk2[igpu],
		 d_p[igpu],
		 d_pprev[igpu], d_pnext[igpu],
		 d_a0[igpu],
		 d_a1[igpu],
		 d_a2[igpu],
		 d_a3[igpu],
		 d_b0[igpu],
		 d_b1[igpu],
		 d_b2[igpu],
		 d_c0[igpu],
		 d_c1[igpu],
		 d_c2[igpu],
		 d_bnd[igpu],
		 d_wrk1[igpu]);
	    }
	  } else {
	    if (igpu + 1 == ngpus) {
	      jacobi_kernel0<true, false><<<grid, NTHREADS>>>
		(d_wrk2[igpu],
		 d_p[igpu],
		 d_pprev[igpu], d_pnext[igpu],
		 d_a0[igpu],
		 d_a1[igpu],
		 d_a2[igpu],
		 d_a3[igpu],
		 d_b0[igpu],
		 d_b1[igpu],
		 d_b2[igpu],
		 d_c0[igpu],
		 d_c1[igpu],
		 d_c2[igpu],
		 d_bnd[igpu],
		 d_wrk1[igpu]);
	    } else {
	      jacobi_kernel0<true, true><<<grid, NTHREADS>>>
		(d_wrk2[igpu],
		 d_p[igpu],
		 d_pprev[igpu], d_pnext[igpu],
		 d_a0[igpu],
		 d_a1[igpu],
		 d_a2[igpu],
		 d_a3[igpu],
		 d_b0[igpu],
		 d_b1[igpu],
		 d_b2[igpu],
		 d_c0[igpu],
		 d_c1[igpu],
		 d_c2[igpu],
		 d_bnd[igpu],
		 d_wrk1[igpu]);
	    }
	  }
        }
      }
    }

    swap = (void *)d_p;
    d_p = d_wrk2;
    d_wrk2 = (float (**)[MJMAX - 1][MKMAX + 31])swap;
    swap = (void *)d_pprev;
    d_pprev = d_wrk2prev;
    d_wrk2prev = (const float (**)[MKMAX + 31])swap;
    swap = (void *)d_pnext;
    d_pnext = d_wrk2next;
    d_wrk2next = (const float (**)[MKMAX + 31])swap;
    for (int igpu = 0; igpu < ngpus; ++igpu) {
      CUDACHECK(cudaSetDevice(gpus[igpu]));
      if (n + 1 == nn) {
        float localgosa;
        int threadwidth = ibounds[igpu + 1] - ibounds[igpu];
        CUDACHECK(cudaMemcpy(h_gosa[ibounds[igpu]],
                             d_gosa[igpu],
                             threadwidth * NBLOCKSJ *
                             sizeof(float),
                             cudaMemcpyDeviceToHost));
        localgosa = 0.0f;
        for (i = ibounds[igpu]; i < ibounds[igpu + 1]; ++i) {
          for (j = 0; j < NBLOCKSJ; ++j) {
            localgosa += h_gosa[i][j];
          }
        }
        gosa += localgosa;
      } else {
        CUDACHECK(cudaDeviceSynchronize());
      }
    }

  } /* end n loop */

  return(gosa);
}

double
fflop(int mx,int my, int mz)
{
  return((double)(mz-2)*(double)(my-2)*(double)(mx-2)*34.0);
}

double
mflops(int nn,double cpu,double flop)
{
  return(flop/cpu*1.e-6*(double)nn);
}

double
second()
{
  struct timeval tm;
  double t ;

  static int base_sec = 0,base_usec = 0;

  gettimeofday(&tm, NULL);
  
  if(base_sec == 0 && base_usec == 0)
    {
      base_sec = tm.tv_sec;
      base_usec = tm.tv_usec;
      t = 0.0;
  } else {
    t = (double) (tm.tv_sec-base_sec) + 
      ((double) (tm.tv_usec-base_usec))/1.0e6 ;
  }

  return t ;
}
