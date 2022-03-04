/* Schoenhage's fast multiplication modulo 2^N+1.

   Contributed by Paul Zimmermann.

   THE FUNCTIONS IN THIS FILE ARE INTERNAL WITH MUTABLE INTERFACES.  IT IS ONLY
   SAFE TO REACH THEM THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT THEY WILL CHANGE OR DISAPPEAR IN A FUTURE GNU MP RELEASE.

Copyright 1998-2010, 2012, 2013, 2018, 2020 Free Software Foundation, Inc.

This file is part of the GNU MP Library.

The GNU MP Library is free software; you can redistribute it and/or modify
it under the terms of either:

  * the GNU Lesser General Public License as published by the Free
    Software Foundation; either version 3 of the License, or (at your
    option) any later version.

or

  * the GNU General Public License as published by the Free Software
    Foundation; either version 2 of the License, or (at your option) any
    later version.

or both in parallel, as here.

The GNU MP Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
for more details.

You should have received copies of the GNU General Public License and the
GNU Lesser General Public License along with the GNU MP Library.  If not,
see https://www.gnu.org/licenses/.  */


/* References:

   Schnelle Multiplikation grosser Zahlen, by Arnold Schoenhage and Volker
   Strassen, Computing 7, p. 281-292, 1971.

   Asymptotically fast algorithms for the numerical multiplication and division
   of polynomials with complex coefficients, by Arnold Schoenhage, Computer
   Algebra, EUROCAM'82, LNCS 144, p. 3-15, 1982.

   Tapes versus Pointers, a study in implementing fast algorithms, by Arnold
   Schoenhage, Bulletin of the EATCS, 30, p. 23-32, 1986.

   TODO:

   Implement some of the tricks published at ISSAC'2007 by Gaudry, Kruppa, and
   Zimmermann.

   It might be possible to avoid a small number of MPN_COPYs by using a
   rotating temporary or two.

   Cleanup and simplify the code!
*/

#define _GNU_SOURCE

#define TRACE_TIME
//#define CHRONOGRAM

#define DISP_CEIL 8
#define DISP_CEIL_FFT 16

//#define UNSAFE_THREAD_ALLOCATION
#ifdef UNSAFE_THREAD_ALLOCATION
#define ALLOC_LOCK(x) pthread_mutex_lock(x)
#define ALLOC_UNLOCK(x) pthread_mutex_unlock(x)
#else
#define ALLOC_LOCK(x) 
#define ALLOC_UNLOCK(x)
#endif

#if defined(TRACE_TIME)
#include <stdio.h>
#include <stdlib.h>
#endif

#ifdef TRACE_TIME
#include <time.h>
#endif

#ifdef TRACE
#undef TRACE
#define TRACE(x) x
#include <stdio.h>
#else
#define TRACE(x)
#endif

#include "gmp-impl.h"

#include <sched.h>
#include <hwloc.h>
#include <pthread.h>

#ifdef WANT_ADDSUB
#include "generic/add_n_sub_n.c"
#define HAVE_NATIVE_mpn_add_n_sub_n 1
#endif

int number_of_threads = 4;
#define NB_THREAD number_of_threads


#ifdef CHRONOGRAM
double **timers1 = NULL;
double **timers2 = NULL;
double **timers3 = NULL;
#endif

typedef struct
{
  pthread_mutex_t *mutex;
  mp_size_t total;
  pthread_barrier_t barrier;
  int initialized;
} mp_lock_t;


void
gmp_set_num_threads(int not) {
  NB_THREAD = not;
}

void inline
mpn_fft_sync(mp_lock_t *lock)
{
  pthread_barrier_wait(&lock->barrier);
}

hwloc_topology_t mpn_mul_fft_topology = NULL;


static mp_limb_t mpn_mul_fft_internal (mp_ptr, mp_size_t, int, mp_ptr *,
				       mp_ptr *, mp_ptr, mp_ptr, mp_size_t,
				       mp_size_t, mp_size_t, int **, mp_ptr, int, mp_lock_t*);

static void mpn_mul_fft_decompose (mp_ptr, mp_ptr *, mp_size_t, mp_size_t, mp_srcptr,
				   mp_size_t, mp_size_t, mp_size_t, mp_ptr, int, int, mp_lock_t*);

static void mpn_fft_div_2exp_modF (mp_ptr r, mp_srcptr a, mp_bitcnt_t k, mp_size_t n);


/* Find the best k to use for a mod 2^(m*GMP_NUMB_BITS)+1 FFT for m >= n.
   We have sqr=0 if for a multiply, sqr=1 for a square.
   There are three generations of this code; we keep the old ones as long as
   some gmp-mparam.h is not updated.  */


/*****************************************************************************/

#if TUNE_PROGRAM_BUILD || (defined (MUL_FFT_TABLE3) && defined (SQR_FFT_TABLE3))

#ifndef FFT_TABLE3_SIZE		/* When tuning this is defined in gmp-impl.h */
#if defined (MUL_FFT_TABLE3_SIZE) && defined (SQR_FFT_TABLE3_SIZE)
#if MUL_FFT_TABLE3_SIZE > SQR_FFT_TABLE3_SIZE
#define FFT_TABLE3_SIZE MUL_FFT_TABLE3_SIZE
#else
#define FFT_TABLE3_SIZE SQR_FFT_TABLE3_SIZE
#endif
#endif
#endif

#ifndef FFT_TABLE3_SIZE
#define FFT_TABLE3_SIZE 200
#endif

FFT_TABLE_ATTRS struct fft_table_nk mpn_fft_table3[2][FFT_TABLE3_SIZE] =
{
  MUL_FFT_TABLE3,
  SQR_FFT_TABLE3
};

int
mpn_fft_best_k (mp_size_t n, int sqr)
{
  const struct fft_table_nk *fft_tab, *tab;
  mp_size_t tab_n, thres;
  int last_k;

  fft_tab = mpn_fft_table3[sqr];
  last_k = fft_tab->k;
  for (tab = fft_tab + 1; ; tab++)
    {
      tab_n = tab->n;
      thres = tab_n << last_k;
      if (n <= thres)
	break;
      last_k = tab->k;
    }
  return last_k;
}

#define MPN_FFT_BEST_READY 1
#endif

#ifdef TRACE_TIME
/* borrowed from CADO-NFS (utils/timing.cpp) */
#include <sys/time.h>
static double
wct_seconds (void)
{
    struct timeval tv[1];
    gettimeofday (tv, NULL);
    return (double)tv->tv_sec + (double)tv->tv_usec*1.0e-6;
}
#endif

/*****************************************************************************/

#if ! defined (MPN_FFT_BEST_READY)
FFT_TABLE_ATTRS mp_size_t mpn_fft_table[2][MPN_FFT_TABLE_SIZE] =
{
  MUL_FFT_TABLE,
  SQR_FFT_TABLE
};

int
mpn_fft_best_k (mp_size_t n, int sqr)
{
  int i;

  for (i = 0; mpn_fft_table[sqr][i] != 0; i++)
    if (n < mpn_fft_table[sqr][i])
      return i + FFT_FIRST_K;

  /* treat 4*last as one further entry */
  if (i == 0 || n < 4 * mpn_fft_table[sqr][i - 1])
    return i + FFT_FIRST_K;
  else
    return i + FFT_FIRST_K + 1;
}
#endif

/*****************************************************************************/


/* Returns smallest possible number of limbs >= pl for a fft of size 2^k,
   i.e. smallest multiple of 2^k >= pl.

   Don't declare static: needed by tuneup.
*/

static void
mpn_fft_add_n_para (mp_ptr p, mp_srcptr n1, mp_srcptr n2, mp_size_t nl,
    mp_limb_t* out, int id, mp_lock_t* lock
#ifdef CHRONOGRAM
    , double *chrono
#endif
    )
{
  mp_size_t r, size;
  mp_limb_t cc;
  r = (nl + lock->total - 1) / lock->total;
  if (r * (id + 1) > nl)
  {
    if (r * id < nl) {
      size = nl - r * id;
    } else {
      size = 0;
    }
  } else {
    size = r;
  }

  cc = mpn_add_n(p + r * id, n1 + r * id, n2 + r * id, size);

  if (id == 0)
  {
    (*out) = 0;
  }

#ifdef CHRONOGRAM
  chrono[0] = wct_seconds ();
#endif
  mpn_fft_sync(lock);
#ifdef CHRONOGRAM
  chrono[1] = wct_seconds ();
#endif

  if (cc != 0)
  {
    pthread_mutex_lock(lock->mutex);
    size += r * id;
    while (cc && size < nl)
    {
      cc = ++(p[size++]) == 0;
    }
    if (cc != 0)
    {
      (*out) += cc;
    }
    pthread_mutex_unlock(lock->mutex);
  }
}

static void
mpn_fft_sub_n_para (mp_ptr p, mp_srcptr n1, mp_srcptr n2, mp_size_t nl,
    mp_limb_t* out, int id, mp_lock_t* lock
#ifdef CHRONOGRAM
    , double *chrono
#endif
    )
{
  double t0, t1, t2, t3, t4;
  mp_size_t r, pos, size;
  mp_limb_t cc, c;
  r = (nl + lock->total - 1) / lock->total;
  size = r * (id + 1);
  r *= id;
  if (size > nl) {size = nl;}
  size -= r;

  pos = r;
  cc = 2;
  while (--pos >= 0 && cc == 2)
  {
    if (n1[pos] != n2[pos])
    {
      cc = n1[pos] < n2[pos];
    }
  }
  if (cc == 2) {
    cc = 0;
  }
#ifdef CHRONOGRAM
  chrono[0] = wct_seconds ();
#endif
  mpn_fft_sync(lock);
#ifdef CHRONOGRAM
  chrono[1] = wct_seconds ();
#endif
  c = mpn_sub_nc(p + r, n1 + r, n2 + r, size, cc);

  if (size && r + size == nl)
  {
    *out = c;
  }

}

static void
mpn_fft_rshift_para (mp_ptr p, mp_srcptr n, mp_size_t nl, unsigned int count, 
    mp_limb_t* out, mp_limb_t in, int id, mp_lock_t* lock
#ifdef CHRONOGRAM
    , double *chrono
#endif
    )
{
  mp_size_t r, size;
  mp_limb_t saved = (1 << count) - 1;
  
  r = (nl + lock->total - 1) / lock->total;
  if (r * (id + 1) > nl)
  {
    if (r * id < nl)
    {
      size = nl - r * id;
    } else {
      size = 0;
    }
  } else {
    size = r;
  }

  if (id == 0 && out != NULL) {
    (*out) = saved & n[0];
  }

  if (r * (id + 1) < nl) {
    saved &= n[r * (id + 1)];
  } else if (r * id + size == nl)
  {
    saved = in;
  } else {
    saved = 0;
  }
  
#ifdef CHRONOGRAM
    chrono[0] = wct_seconds ();
#endif
  mpn_fft_sync(lock);
#ifdef CHRONOGRAM
    chrono[1] = wct_seconds ();
#endif

  mpn_rshift(p + r * id, n + r * id, size, count);
  p[r * id + size - 1] |= saved << (GMP_LIMB_BITS - count);
}

static void
mpn_fft_copy_para (mp_ptr p, mp_srcptr s, mp_size_t nl, int id, mp_lock_t* lock)
{
  mp_size_t r, size;
  r = (nl + lock->total - 1) / lock->total;
  if (r * (id + 1) < nl) {
    size = r;
  } else {
    size = (r * id < nl) ? nl - r * id : 0;
  }
  MPN_COPY (p + r * id, s + r * id, size);
}

mp_size_t
mpn_fft_next_size (mp_size_t pl, int k)
{
  pl = 1 + ((pl - 1) >> k); /* ceil (pl/2^k) */
  return pl << k;
}


/* Initialize l[i][j] with bitrev(j) */
static void
mpn_fft_initl (int **l, int k)
{
  int i, j, K;
  int *li;

  l[0][0] = 0;
  for (i = 1, K = 1; i <= k; i++, K *= 2)
    {
      li = l[i];
      for (j = 0; j < K; j++)
	{
	  li[j] = 2 * l[i - 1][j];
	  li[K + j] = 1 + li[j];
	}
    }
}


/* r <- a*2^d mod 2^(n*GMP_NUMB_BITS)+1 with a = {a, n+1}
   Assumes a is semi-normalized, i.e. a[n] <= 1.
   r and a must have n+1 limbs, and not overlap.
*/
static void
mpn_fft_mul_2exp_modF (mp_ptr r, mp_srcptr a, mp_bitcnt_t d, mp_size_t n)
{
  unsigned int sh;
  mp_size_t m;
  mp_limb_t cc, rd;

  sh = d % GMP_NUMB_BITS;
  m = d / GMP_NUMB_BITS;

  if (m >= n)			/* negate */
    {
      /* r[0..m-1]  <-- lshift(a[n-m]..a[n-1], sh)
	 r[m..n-1]  <-- -lshift(a[0]..a[n-m-1],  sh) */

      m -= n;
      if (sh != 0)
	{
	  /* no out shift below since a[n] <= 1 */
	  mpn_lshift (r, a + n - m, m + 1, sh);
	  rd = r[m];
	  cc = mpn_lshiftc (r + m, a, n - m, sh);
	}
      else
	{
	  MPN_COPY (r, a + n - m, m);
	  rd = a[n];
	  mpn_com (r + m, a, n - m);
	  cc = 0;
	}

      /* add cc to r[0], and add rd to r[m] */

      /* now add 1 in r[m], subtract 1 in r[n], i.e. add 1 in r[0] */

      r[n] = 0;
      /* cc < 2^sh <= 2^(GMP_NUMB_BITS-1) thus no overflow here */
      cc++;
      mpn_incr_u (r, cc);

      rd++;
      /* rd might overflow when sh=GMP_NUMB_BITS-1 */
      cc = (rd == 0) ? 1 : rd;
      r = r + m + (rd == 0);
      mpn_incr_u (r, cc);
    }
  else
    {
      /* r[0..m-1]  <-- -lshift(a[n-m]..a[n-1], sh)
	 r[m..n-1]  <-- lshift(a[0]..a[n-m-1],  sh)  */
      if (sh != 0)
	{
	  /* no out bits below since a[n] <= 1 */
	  mpn_lshiftc (r, a + n - m, m + 1, sh);
	  rd = ~r[m];
	  /* {r, m+1} = {a+n-m, m+1} << sh */
	  cc = mpn_lshift (r + m, a, n - m, sh); /* {r+m, n-m} = {a, n-m}<<sh */
	}
      else
	{
	  /* r[m] is not used below, but we save a test for m=0 */
	  mpn_com (r, a + n - m, m + 1);
	  rd = a[n];
	  MPN_COPY (r + m, a, n - m);
	  cc = 0;
	}

      /* now complement {r, m}, subtract cc from r[0], subtract rd from r[m] */

      /* if m=0 we just have r[0]=a[n] << sh */
      if (m != 0)
	{
	  /* now add 1 in r[0], subtract 1 in r[m] */
	  if (cc-- == 0) /* then add 1 to r[0] */
	    cc = mpn_add_1 (r, r, n, CNST_LIMB(1));
	  cc = mpn_sub_1 (r, r, m, cc) + 1;
	  /* add 1 to cc instead of rd since rd might overflow */
	}

      /* now subtract cc and rd from r[m..n] */

      r[n] = -mpn_sub_1 (r + m, r + m, n - m, cc);
      r[n] -= mpn_sub_1 (r + m, r + m, n - m, rd);
      if (r[n] & GMP_LIMB_HIGHBIT)
	r[n] = mpn_add_1 (r, r, n, CNST_LIMB(1));
    }
}

#if HAVE_NATIVE_mpn_add_n_sub_n
static inline void
mpn_fft_add_sub_modF (mp_ptr A0, mp_ptr Ai, mp_srcptr tp, mp_size_t n)
{
  mp_limb_t cyas, c, x;

  cyas = mpn_add_n_sub_n (A0, Ai, A0, tp, n);

  c = A0[n] - tp[n] - (cyas & 1);
  x = (-c) & -((c & GMP_LIMB_HIGHBIT) != 0);
  Ai[n] = x + c;
  MPN_INCR_U (Ai, n + 1, x);

  c = A0[n] + tp[n] + (cyas >> 1);
  x = (c - 1) & -(c != 0);
  A0[n] = c - x;
  MPN_DECR_U (A0, n + 1, x);
}

#else /* ! HAVE_NATIVE_mpn_add_n_sub_n  */

/* r <- a+b mod 2^(n*GMP_NUMB_BITS)+1.
   Assumes a and b are semi-normalized.
*/
static inline void
mpn_fft_add_modF (mp_ptr r, mp_srcptr a, mp_srcptr b, mp_size_t n)
{
  mp_limb_t c, x;

  c = a[n] + b[n] + mpn_add_n (r, a, b, n);
  /* 0 <= c <= 3 */

#if 1
  /* GCC 4.1 outsmarts most expressions here, and generates a 50% branch.  The
     result is slower code, of course.  But the following outsmarts GCC.  */
  x = (c - 1) & -(c != 0);
  r[n] = c - x;
  MPN_DECR_U (r, n + 1, x);
#endif
#if 0
  if (c > 1)
    {
      r[n] = 1;                       /* r[n] - c = 1 */
      MPN_DECR_U (r, n + 1, c - 1);
    }
  else
    {
      r[n] = c;
    }
#endif
}

/* r <- a-b mod 2^(n*GMP_NUMB_BITS)+1.
   Assumes a and b are semi-normalized.
*/
static inline void
mpn_fft_sub_modF (mp_ptr r, mp_srcptr a, mp_srcptr b, mp_size_t n)
{
  mp_limb_t c, x;

  c = a[n] - b[n] - mpn_sub_n (r, a, b, n);
  /* -2 <= c <= 1 */

#if 1
  /* GCC 4.1 outsmarts most expressions here, and generates a 50% branch.  The
     result is slower code, of course.  But the following outsmarts GCC.  */
  x = (-c) & -((c & GMP_LIMB_HIGHBIT) != 0);
  r[n] = x + c;
  MPN_INCR_U (r, n + 1, x);
#endif
#if 0
  if ((c & GMP_LIMB_HIGHBIT) != 0)
    {
      r[n] = 0;
      MPN_INCR_U (r, n + 1, -c);
    }
  else
    {
      r[n] = c;
    }
#endif
}

static inline void
mpn_fft_add_sub_modF (mp_ptr A0, mp_ptr Ai, mp_srcptr tp, mp_size_t n)
{
  mpn_fft_sub_modF (Ai, A0, tp, n);
	mpn_fft_add_modF (A0, A0, tp, n);
}
#endif /* HAVE_NATIVE_mpn_add_n_sub_n */

/* input: A[0] ... A[inc*(K-1)] are residues mod 2^N+1 where
	  N=n*GMP_NUMB_BITS, and 2^omega is a primitive root mod 2^N+1
   output: A[inc*l[k][i]] <- \sum (2^omega)^(ij) A[inc*j] mod 2^N+1 */

static void
mpn_fft_fft (mp_ptr *Ap, mp_size_t K, int **ll,
	     mp_size_t omega, mp_size_t n, mp_size_t inc, mp_ptr tp)
{
  if (K == 2)
    {
      mp_limb_t cy;
#if HAVE_NATIVE_mpn_add_n_sub_n
      cy = mpn_add_n_sub_n (Ap[0], Ap[inc], Ap[0], Ap[inc], n + 1) & 1;
#else
      MPN_COPY (tp, Ap[0], n + 1);
      mpn_add_n (Ap[0], Ap[0], Ap[inc], n + 1);
      cy = mpn_sub_n (Ap[inc], tp, Ap[inc], n + 1);
#endif
      if (Ap[0][n] > 1) /* can be 2 or 3 */
	Ap[0][n] = 1 - mpn_sub_1 (Ap[0], Ap[0], n, Ap[0][n] - 1);
      if (cy) /* Ap[inc][n] can be -1 or -2 */
	Ap[inc][n] = mpn_add_1 (Ap[inc], Ap[inc], n, ~Ap[inc][n] + 1);
    }
  else
    {
      mp_size_t j, K2 = K >> 1;
      int *lk = *ll;

      mpn_fft_fft (Ap,     K2, ll-1, 2 * omega, n, inc * 2, tp);
      mpn_fft_fft (Ap+inc, K2, ll-1, 2 * omega, n, inc * 2, tp);
      /* A[2*j*inc]   <- A[2*j*inc] + omega^l[k][2*j*inc] A[(2j+1)inc]
	 A[(2j+1)inc] <- A[2*j*inc] + omega^l[k][(2j+1)inc] A[(2j+1)inc] */
      for (j = 0; j < K2; j++, lk += 2, Ap += 2 * inc)
	{
	  /* Ap[inc] <- Ap[0] + Ap[inc] * 2^(lk[1] * omega)
	     Ap[0]   <- Ap[0] + Ap[inc] * 2^(lk[0] * omega) */
	  mpn_fft_mul_2exp_modF (tp, Ap[inc], lk[0] * omega, n);
#if HAVE_NATIVE_mpn_add_n_sub_n
	  mpn_fft_add_sub_modF (Ap[0], Ap[inc], tp, n);
#else
	  mpn_fft_sub_modF (Ap[inc], Ap[0], tp, n);
	  mpn_fft_add_modF (Ap[0],   Ap[0], tp, n);
#endif
	}
    }
}

static void
mpn_fft_bailey (
  mp_ptr *Ap, mp_size_t k, int **fft_l,
  mp_size_t omega, mp_size_t n, mp_ptr tp_local, mp_lock_t* lock, int id, int nt_raw
#ifdef CHRONOGRAM
  , double* chrono
#endif
  )
{
#ifdef FFT_TIMER
  double t1, t2, t3, t4;
#endif
  mp_size_t k1, K1, k2, K2, i, j, root, r, maxi;
  mp_size_t nt = (mp_size_t)nt_raw;
  k1 = k >> 1;
  k2 = k - k1;
  K1 = 1 << k1;
  K2 = 1 << k2;

  r = (K2 + nt - 1)/nt;
  maxi = r * (id + 1);
  if (maxi > K2) {maxi = K2;};

  for (j = id * r; j < maxi; j++)
  {
    mpn_fft_fft(Ap + j, K1, fft_l + k1, omega * K2, n, K2, tp_local);
  }

#ifdef CHRONOGRAM
  chrono[11] = wct_seconds ();
#endif
  mpn_fft_sync(lock);
#ifdef CHRONOGRAM
  chrono[12] = wct_seconds ();
#endif

  r = (K1 + nt - 1)/nt;
  maxi = r * (id + 1);
  if (maxi > K1) {maxi = K1;};

  for (i = id; i < K1; i+=nt)
  {
    root = fft_l[k1][i] * omega;
    for (j = 1; j < K2; j++)
    {
      mpn_fft_mul_2exp_modF(tp_local, Ap[i * K2 + j], root * j, n);
      mpn_copyi(Ap[i * K2 + j], tp_local, n + 1);
    }

    mpn_fft_fft(Ap + i * K2, K2, fft_l + k2, omega * K1, n, 1, tp_local);
  }
}


/* Given ap[0..n] with ap[n]<=1, reduce it modulo 2^(n*GMP_NUMB_BITS)+1,
   by subtracting that modulus if necessary.

   If ap[0..n] is exactly 2^(n*GMP_NUMB_BITS) then mpn_sub_1 produces a
   borrow and the limbs must be zeroed out again.  This will occur very
   infrequently.  */

static inline void
mpn_fft_normalize (mp_ptr ap, mp_size_t n)
{
  if (ap[n] != 0)
    {
      MPN_DECR_U (ap, n + 1, CNST_LIMB(1));
      if (ap[n] == 0)
	{
	  /* This happens with very low probability; we have yet to trigger it,
	     and thereby make sure this code is correct.  */
	  MPN_ZERO (ap, n);
	  ap[n] = 1;
	}
      else
	ap[n] = 0;
    }
}

/* a[i] <- a[i]*b[i] mod 2^(n*GMP_NUMB_BITS)+1 for 0 <= i < K */
static void
mpn_fft_mul_modF_K (mp_ptr *ap, mp_ptr *bp, int **fft_l, mp_size_t n, mp_size_t K, mp_lock_t* lock)
{
  int i;
  int sqr = (ap == bp);
  //TMP_DECL;
  //TMP_MARK;

  if (n >= (sqr ? SQR_FFT_MODF_THRESHOLD : MUL_FFT_MODF_THRESHOLD))
    {
      mp_size_t K2, nprime2, Nprime2, M2, maxLK, l, Mp2;
      int k;
      int **fft_l, *tmp;
      mp_ptr *Ap, *Bp, A, B, T;

      k = mpn_fft_best_k (n, sqr);
      K2 = (mp_size_t) 1 << k;
      ASSERT_ALWAYS((n & (K2 - 1)) == 0);
      maxLK = (K2 > GMP_NUMB_BITS) ? K2 : GMP_NUMB_BITS;
      M2 = n * GMP_NUMB_BITS >> k;
      l = n >> k;
      Nprime2 = ((2 * M2 + k + 2 + maxLK) / maxLK) * maxLK;
      /* Nprime2 = ceil((2*M2+k+3)/maxLK)*maxLK*/
      nprime2 = Nprime2 / GMP_NUMB_BITS;

      /* we should ensure that nprime2 is a multiple of the next K */
      if (nprime2 >= (sqr ? SQR_FFT_MODF_THRESHOLD : MUL_FFT_MODF_THRESHOLD))
	{
	  mp_size_t K3;
	  for (;;)
	    {
	      K3 = (mp_size_t) 1 << mpn_fft_best_k (nprime2, sqr);
	      if ((nprime2 & (K3 - 1)) == 0)
		break;
	      nprime2 = (nprime2 + K3 - 1) & -K3;
	      Nprime2 = nprime2 * GMP_LIMB_BITS;
	      /* warning: since nprime2 changed, K3 may change too! */
	    }
	}
      ASSERT_ALWAYS(nprime2 < n); /* otherwise we'll loop */
      ASSERT_ALWAYS(K2 <= K); /* otherwise fft_l is not valid */

      Mp2 = Nprime2 >> k;
      ALLOC_LOCK (lock->mutex);
      Ap = __GMP_ALLOCATE_FUNC_TYPE (K2, mp_ptr);
      Bp = __GMP_ALLOCATE_FUNC_TYPE (K2, mp_ptr);
      A = __GMP_ALLOCATE_FUNC_LIMBS (2 * (nprime2 + 1) << k);
      T = __GMP_ALLOCATE_FUNC_LIMBS (2 * (nprime2 + 1));
      ALLOC_UNLOCK (lock->mutex);
      B = A + ((nprime2 + 1) << k);
      TRACE (printf ("recurse: %ldx%ld limbs -> %ld times %ldx%ld (%1.2f)\n", n,
		    n, K2, nprime2, nprime2, 2.0*(double)n/nprime2/K2));
      for (i = 0; i < K; i++, ap++, bp++)
	{
	  mp_limb_t cy;
	  mpn_fft_normalize (*ap, n);
	  if (!sqr)
	    mpn_fft_normalize (*bp, n);

	  mpn_mul_fft_decompose (A, Ap, K2, nprime2, *ap, (l << k) + 1, l, Mp2, T, 0, 0, lock);
	  if (!sqr)
	    mpn_mul_fft_decompose (B, Bp, K2, nprime2, *bp, (l << k) + 1, l, Mp2, T, 0, 0, lock);

	  cy = mpn_mul_fft_internal (*ap, n, k, Ap, Bp, A, B, nprime2,
				     l, Mp2, fft_l, T, sqr, lock);
	  (*ap)[n] = cy;
	}
      ALLOC_LOCK (lock->mutex);
    __GMP_FREE_FUNC_TYPE (Ap, K2, mp_ptr);
    __GMP_FREE_FUNC_TYPE (Bp, K2, mp_ptr);
    __GMP_FREE_FUNC_LIMBS(T, 2 * (nprime2 + 1) << k);
    __GMP_FREE_FUNC_LIMBS(A, 2 * (nprime2 + 1));
      ALLOC_UNLOCK (lock->mutex);
    }
  else
    {
      mp_ptr a, b, tp, tpn;
      mp_limb_t cc;
      mp_size_t n2 = 2 * n;
      ALLOC_LOCK (lock->mutex);
      tp = __GMP_ALLOCATE_FUNC_LIMBS (n2);
      ALLOC_UNLOCK (lock->mutex);
      tpn = tp + n;
      TRACE (printf ("  mpn_mul_n %ld of %ld limbs\n", K, n));
      for (i = 0; i < K; i++)
	{
	  a = *ap++;
	  b = *bp++;
	  if (sqr)
	    mpn_sqr (tp, a, n);
	  else
	    mpn_mul_n (tp, b, a, n);
	  if (a[n] != 0)
	    cc = mpn_add_n (tpn, tpn, b, n);
	  else
	    cc = 0;
	  if (b[n] != 0)
	    cc += mpn_add_n (tpn, tpn, a, n) + a[n];
	  if (cc != 0)
	    {
	      cc = mpn_add_1 (tp, tp, n2, cc);
	      /* If mpn_add_1 give a carry (cc != 0),
		 the result (tp) is at most GMP_NUMB_MAX - 1,
		 so the following addition can't overflow.
	      */
	      tp[0] += cc;
	    }
	  a[n] = mpn_sub_n (a, tp, tpn, n) && mpn_add_1 (a, a, n, CNST_LIMB(1));
	}

      ALLOC_LOCK (lock->mutex);
    __GMP_FREE_FUNC_LIMBS(tp, n2);
      ALLOC_UNLOCK (lock->mutex);
    }
  
  //TMP_FREE;
}


/* a[i] <- a[i]*b[i] mod 2^(n*GMP_NUMB_BITS)+1 for 0 <= i < K */
static void
mpn_fft_mul_modF_K_para (mp_ptr *ap, mp_ptr *bp, int **fft_l, mp_size_t n, mp_size_t K, mp_lock_t* lock, int id)
{
  int i;
  int sqr = (ap == bp);
  //TMP_DECL;

  //TMP_MARK;
  if (n >= (sqr ? SQR_FFT_MODF_THRESHOLD : MUL_FFT_MODF_THRESHOLD))
    {
      mp_size_t K2, nprime2, Nprime2, M2, maxLK, l, Mp2;
      int k;

      k = mpn_fft_best_k (n, sqr);
      K2 = (mp_size_t) 1 << k;
      ASSERT_ALWAYS((n & (K2 - 1)) == 0);
      maxLK = (K2 > GMP_NUMB_BITS) ? K2 : GMP_NUMB_BITS;
      M2 = n * GMP_NUMB_BITS >> k;
      l = n >> k;
      Nprime2 = ((2 * M2 + k + 2 + maxLK) / maxLK) * maxLK;
      /* Nprime2 = ceil((2*M2+k+3)/maxLK)*maxLK*/
      nprime2 = Nprime2 / GMP_NUMB_BITS;

      /* we should ensure that nprime2 is a multiple of the next K */
      if (nprime2 >= (sqr ? SQR_FFT_MODF_THRESHOLD : MUL_FFT_MODF_THRESHOLD))
	{
	  mp_size_t K3;
	  for (;;)
	    {
	      K3 = (mp_size_t) 1 << mpn_fft_best_k (nprime2, sqr);
	      if ((nprime2 & (K3 - 1)) == 0)
		break;
	      nprime2 = (nprime2 + K3 - 1) & -K3;
	      Nprime2 = nprime2 * GMP_LIMB_BITS;
	      /* warning: since nprime2 changed, K3 may change too! */
	    }
	}
      ASSERT_ALWAYS(nprime2 < n); /* otherwise we'll loop */

      Mp2 = Nprime2 >> k;

      mp_ptr *Ap, *Bp, A, B, T;
      int i;

      ALLOC_LOCK (lock->mutex);
      Ap = __GMP_ALLOCATE_FUNC_TYPE(K2, mp_ptr);
      Bp = __GMP_ALLOCATE_FUNC_TYPE(K2, mp_ptr);
      A =  __GMP_ALLOCATE_FUNC_LIMBS((nprime2 + 1) << k);
      B =  __GMP_ALLOCATE_FUNC_LIMBS((nprime2 + 1) << k);
      T =  __GMP_ALLOCATE_FUNC_LIMBS(2 * (nprime2 + 1));
      ALLOC_UNLOCK (lock->mutex);

      TRACE (printf ("recurse: %ldx%ld limbs -> %ld times %ldx%ld (%1.2f)\n", n,
        n, K2, nprime2, nprime2, 2.0*(double)n/nprime2/K2));

      mp_size_t r = (K + lock->total - 1) / lock->total;
      mp_size_t maxi = r * (id + 1);
      if (maxi > K) {maxi = K;};
      ap += r * id;
      bp += r * id;
      for (i = r * id; i < maxi; i++, ap++, bp++)
    {
      mp_limb_t cy;
      mpn_fft_normalize (*ap, n);
      if (!sqr)
        mpn_fft_normalize (*bp, n);
      mpn_mul_fft_decompose (A, Ap, K2, nprime2, *ap, (l << k) + (*ap)[n], l, Mp2, T, i, id, lock);
      if (!sqr)
        mpn_mul_fft_decompose (B, Bp, K2, nprime2, *bp, (l << k) + (*bp)[n], l, Mp2, T, i, id, lock);
      cy = mpn_mul_fft_internal (*ap, n, k, Ap, Bp, A, B, nprime2,
              l, Mp2, fft_l, T, sqr, lock);
      (*ap)[n] = cy;
    }

      ALLOC_LOCK (lock->mutex);
    __GMP_FREE_FUNC_TYPE(Ap, K2, mp_ptr);
    __GMP_FREE_FUNC_TYPE(Bp, K2, mp_ptr);
    __GMP_FREE_FUNC_LIMBS(A, (nprime2 + 1) << k);
    __GMP_FREE_FUNC_LIMBS(B, (nprime2 + 1) << k);
    __GMP_FREE_FUNC_LIMBS(T, 2 * (nprime2 + 1));
      ALLOC_UNLOCK (lock->mutex);
    }
  else
    {
      mp_ptr a, b, tp, tpn;
      mp_limb_t cc;
      mp_size_t n2 = 2 * n;

      ALLOC_LOCK (lock->mutex);
      tp = __GMP_ALLOCATE_FUNC_LIMBS (n2);
      ALLOC_UNLOCK (lock->mutex);

      tpn = tp + n;
      TRACE (printf ("  mpn_mul_n %ld of %ld limbs\n", K, n));

      mp_size_t r = (K + lock->total - 1) / lock->total;
      mp_size_t maxi = r * (id + 1);
      if (maxi > K) {maxi = K;};
      for (int i = r * id; i < maxi; i++)
    {
      a = *(ap+i);
      b = *(bp+i);
      if (sqr)
        mpn_sqr (tp, a, n);
      else
        mpn_mul_n (tp, b, a, n);
      if (a[n] != 0)
        cc = mpn_add_n (tpn, tpn, b, n);
      else
        cc = 0;
      if (b[n] != 0)
        cc += mpn_add_n (tpn, tpn, a, n) + a[n];
      if (cc != 0)
        {
          cc = mpn_add_1 (tp, tp, n2, cc);
          /* If mpn_add_1 give a carry (cc != 0),
      the result (tp) is at most GMP_NUMB_MAX - 1,
      so the following addition can't overflow.
          */
          tp[0] += cc;
        }
      a[n] = mpn_sub_n (a, tp, tpn, n) && mpn_add_1 (a, a, n, CNST_LIMB(1));
    }
      ALLOC_LOCK (lock->mutex);
      __GMP_FREE_FUNC_LIMBS(tp, n2);
      ALLOC_UNLOCK (lock->mutex);
  }


}


/* R <- A/2^k mod 2^(n*GMP_NUMB_BITS)+1 */
static void
mpn_fft_div_2exp_modF (mp_ptr r, mp_srcptr a, mp_bitcnt_t k, mp_size_t n)
{
  mp_bitcnt_t i;

  ASSERT (r != a);
  i = (mp_bitcnt_t) 2 * n * GMP_NUMB_BITS - k;
  mpn_fft_mul_2exp_modF (r, a, i, n);
  /* 1/2^k = 2^(2nL-k) mod 2^(n*GMP_NUMB_BITS)+1 */
  /* normalize so that R < 2^(n*GMP_NUMB_BITS)+1 */
  mpn_fft_normalize (r, n);
}

/* input: A^[l[k][0]] A^[l[k][1]] ... A^[l[k][K-1]]
   output: K*A[0] K*A[K-1] ... K*A[1].
   Assumes the Ap[] are pseudo-normalized, i.e. 0 <= Ap[][n] <= 1.
   This condition is also fulfilled at exit.
*/
static void
mpn_fft_fftinv (mp_ptr *Ap, mp_size_t K, mp_size_t omega, mp_size_t n, mp_ptr tp)
{
  if (K == 2)
    {
      mp_limb_t cy;
#if HAVE_NATIVE_mpn_add_n_sub_n
      cy = mpn_add_n_sub_n (Ap[0], Ap[1], Ap[0], Ap[1], n + 1) & 1;
#else
      MPN_COPY (tp, Ap[0], n + 1);
      mpn_add_n (Ap[0], Ap[0], Ap[1], n + 1);
      cy = mpn_sub_n (Ap[1], tp, Ap[1], n + 1);
#endif
      if (Ap[0][n] > 1) /* can be 2 or 3 */
	Ap[0][n] = 1 - mpn_sub_1 (Ap[0], Ap[0], n, Ap[0][n] - 1);
      if (cy) /* Ap[1][n] can be -1 or -2 */
	Ap[1][n] = mpn_add_1 (Ap[1], Ap[1], n, ~Ap[1][n] + 1);
    }
  else
    {
      mp_size_t j, K2 = K >> 1;

      mpn_fft_fftinv (Ap,      K2, 2 * omega, n, tp);
      mpn_fft_fftinv (Ap + K2, K2, 2 * omega, n, tp);
      /* A[j]     <- A[j] + omega^j A[j+K/2]
	 A[j+K/2] <- A[j] + omega^(j+K/2) A[j+K/2] */
      for (j = 0; j < K2; j++, Ap++)
	{
	  /* Ap[K2] <- Ap[0] + Ap[K2] * 2^((j + K2) * omega)
	     Ap[0]  <- Ap[0] + Ap[K2] * 2^(j * omega) */
	  mpn_fft_mul_2exp_modF (tp, Ap[K2], j * omega, n);
	  mpn_fft_add_sub_modF (Ap[0], Ap[K2], tp, n);
	}
    }
}

/*
 inverse FFT above but only using one number out of inc.
*/

static void
mpn_fft_fftinv_inc (mp_ptr *Ap, mp_size_t K, mp_size_t omega, mp_size_t n, mp_size_t inc, mp_ptr tp)
{
  if (K == 2)
    {
      mp_limb_t cy;
#if HAVE_NATIVE_mpn_add_n_sub_n
      cy = mpn_add_n_sub_n (Ap[0], Ap[inc], Ap[0], Ap[inc], n + 1) & 1;
#else
      MPN_COPY (tp, Ap[0], n + 1);
      mpn_add_n (Ap[0], Ap[0], Ap[1], n + 1);
      cy = mpn_sub_n (Ap[1], tp, Ap[1], n + 1);
#endif
      if (Ap[0][n] > 1) /* can be 2 or 3 */
	Ap[0][n] = 1 - mpn_sub_1 (Ap[0], Ap[0], n, Ap[0][n] - 1);
      if (cy) /* Ap[1][n] can be -1 or -2 */
	Ap[inc][n] = mpn_add_1 (Ap[inc], Ap[inc], n, ~Ap[inc][n] + 1);
    }
  else
    {
      mp_size_t j, K2 = K >> 1;

      mpn_fft_fftinv_inc (Ap,      K2, 2 * omega, n, inc, tp);
      mpn_fft_fftinv_inc (Ap + inc * K2, K2, 2 * omega, n, inc, tp);
      /* A[j]     <- A[j] + omega^j A[j+K/2]
	 A[j+K/2] <- A[j] + omega^(j+K/2) A[j+K/2] */
      for (j = 0; j < K2; j++, Ap+= inc)
	{
	  /* Ap[K2] <- Ap[0] + Ap[K2] * 2^((j + K2) * omega)
	     Ap[0]  <- Ap[0] + Ap[K2] * 2^(j * omega) */
	  mpn_fft_mul_2exp_modF (tp, Ap[inc * K2], j * omega, n);
	  mpn_fft_add_sub_modF (Ap[0], Ap[inc * K2], tp, n);
	}
    }
}


static void
mpn_fft_baileyinv (
  mp_ptr *Ap, mp_size_t k, int**ll,
  mp_size_t omega, mp_size_t n, mp_ptr tp_local, mp_lock_t *lock, int id
#ifdef CHRONOGRAM
  , double* chrono
#endif
  )
{
  mp_size_t k1, K1, k2, K2, i, j, r, maxi, nt;
  nt = (mp_size_t)lock->total;

  k1 = k >> 1;
  k2 = k - k1;
  K1 = 1 << k1;
  K2 = 1 << k2;

  /*r = (K1 + nt - 1) / nt;
  maxi = r * (id + 1);
  if (maxi > K1) {maxi = K1;};*/

  for (i = id; i < K1; i+= nt)
  {
    mpn_fft_fftinv(Ap + i * K2, K2, omega * K1, n, tp_local);

    for (j = 0; j < K2; j++)
    {
      mpn_fft_mul_2exp_modF(tp_local, Ap[i * K2 + j], ll[k1][i] * j * omega, n);
      mpn_copyi(Ap[i * K2 + j], tp_local, n + 1);
    }
  }

#ifdef CHRONOGRAM
  chrono[17] = wct_seconds ();
#endif
  mpn_fft_sync(lock);
#ifdef CHRONOGRAM
  chrono[18] = wct_seconds ();
#endif
  r = (K2 + nt - 1) / nt;
  maxi = r * (id + 1);
  if (maxi > K2) {maxi = K2;};

  for (i = r * id; i < maxi; i++)
  {
    mpn_fft_fftinv_inc(Ap + i, K1, omega * K2, n, K2, tp_local);
  }

}

/* {rp,n} <- {ap,an} mod 2^(n*GMP_NUMB_BITS)+1, n <= an <= 3*n.
   Returns carry out, i.e. 1 iff {ap,an} = -1 mod 2^(n*GMP_NUMB_BITS)+1,
   then {rp,n}=0.
*/
static mp_size_t
mpn_fft_norm_modF (mp_ptr rp, mp_size_t n, mp_ptr ap, mp_size_t an)
{
  mp_size_t l, m, rpn;
  mp_limb_t cc;

  ASSERT ((n <= an) && (an <= 3 * n));
  m = an - 2 * n;
  if (m > 0)
    {
      l = n;
      /* add {ap, m} and {ap+2n, m} in {rp, m} */
      cc = mpn_add_n (rp, ap, ap + 2 * n, m);
      /* copy {ap+m, n-m} to {rp+m, n-m} */
      rpn = mpn_add_1 (rp + m, ap + m, n - m, cc);
    }
  else
    {
      l = an - n; /* l <= n */
      MPN_COPY (rp, ap, n);
      rpn = 0;
    }

  /* remains to subtract {ap+n, l} from {rp, n+1} */
  cc = mpn_sub_n (rp, rp, ap + n, l);
  rpn -= mpn_sub_1 (rp + l, rp + l, n - l, cc);
  if (rpn < 0) /* necessarily rpn = -1 */
    rpn = mpn_add_1 (rp, rp, n, CNST_LIMB(1));
  return rpn;
}

static void
mpn_fft_norm_modF_para (mp_ptr rp, mp_size_t n, mp_ptr ap, mp_size_t an, mp_limb_t* out, int id, mp_lock_t* lock
#ifdef CHRONOGRAM
  , double *chrono
#endif
  )
{

  mp_size_t l, m, rpn, r, size;
  mp_limb_t cc;

  ASSERT ((n <= an) && (an <= 3 * n));
  m = an - 2 * n;
  if (m > 0)
    {
      if (id == 0)
      {
      l = n;
      /* add {ap, m} and {ap+2n, m} in {rp, m} */
      cc = mpn_add_n (rp, ap, ap + 2 * n, m);
      /* copy {ap+m, n-m} to {rp+m, n-m} */
      rpn = mpn_add_1 (rp + m, ap + m, n - m, cc);
      }
    }
  else
    {
      l = an - n;
      mpn_fft_copy_para(rp, ap, n, id, lock);
      rpn = 0;
    }
#ifdef CHRONOGRAM
  chrono[32] = wct_seconds ();
#endif
  mpn_fft_sync(lock);
#ifdef CHRONOGRAM
  chrono[33] = wct_seconds ();
#endif
  if (id == 0)
  {
  /* remains to subtract {ap+n, l} from {rp, n+1} */
  cc = mpn_sub_n (rp, rp, ap + n, l);
  rpn -= mpn_sub_1 (rp + l, rp + l, n - l, cc);
  if (rpn < 0) /* necessarily rpn = -1 */
    rpn = mpn_add_1 (rp, rp, n, CNST_LIMB(1));
  (*out) = rpn;
  }
}

/* store in A[0..nprime] the first M bits from {n, nl},
   in A[nprime+1..] the following M bits, ...
   Assumes M is a multiple of GMP_NUMB_BITS (M = l * GMP_NUMB_BITS).
   T must have space for at least (nprime + 1) limbs.
   We must have nl <= 2*K*l.
*/
static void
mpn_mul_fft_decompose (mp_ptr A, mp_ptr *Ap, mp_size_t K, mp_size_t nprime,
		       mp_srcptr n, mp_size_t nl, mp_size_t l, mp_size_t Mp,
		       mp_ptr T, int p, int id, mp_lock_t* lock)
{
  mp_size_t i, j;
  mp_ptr tmp;
  mp_size_t Kl = K * l;
  int dif1 = nl > Kl;


  if (dif1) /* normalize {n, nl} mod 2^(Kl*GMP_NUMB_BITS)+1 */
    {
      mp_size_t dif = nl - Kl;
      mp_limb_signed_t cy;
      ALLOC_LOCK (lock->mutex);
      tmp = __GMP_ALLOCATE_FUNC_LIMBS (Kl + 1);
      ALLOC_UNLOCK (lock->mutex);
      if (dif > Kl)
	{
	  int subp = 0;

	  cy = mpn_sub_n (tmp, n, n + Kl, Kl);
	  n += 2 * Kl;
	  dif -= Kl;

	  /* now dif > 0 */
	  while (dif > Kl)
	    {
	      if (subp)
		cy += mpn_sub_n (tmp, tmp, n, Kl);
	      else
		cy -= mpn_add_n (tmp, tmp, n, Kl);
	      subp ^= 1;
	      n += Kl;
	      dif -= Kl;
	    }
	  /* now dif <= Kl */
	  if (subp)
	    cy += mpn_sub (tmp, tmp, Kl, n, dif);
	  else
	    cy -= mpn_add (tmp, tmp, Kl, n, dif);
	  if (cy >= 0)
	    cy = mpn_add_1 (tmp, tmp, Kl, cy);
	  else
	    cy = mpn_sub_1 (tmp, tmp, Kl, -cy);
	}
      else /* dif <= Kl, i.e. nl <= 2 * Kl */
	{
	  cy = mpn_sub (tmp, n, Kl, n + Kl, dif);
	  cy = mpn_add_1 (tmp, tmp, Kl, cy);
	}
      tmp[Kl] = cy;
      nl = Kl + 1;
      n = tmp;
    }
  for (i = 0; i < K; i++)
    {
      Ap[i] = A;
      /* store the next M bits of n into A[0..nprime] */
      if (nl > 0) /* nl is the number of remaining limbs */
	{
	  j = (l <= nl && i < K - 1) ? l : nl; /* store j next limbs */
	  nl -= j;
	  MPN_COPY (T, n, j);
	  MPN_ZERO (T + j, nprime + 1 - j);
	  n += l;
	  mpn_fft_mul_2exp_modF (A, T, i * Mp, nprime);
	}
      else
	MPN_ZERO (A, nprime + 1);
      A += nprime + 1;
    }
  ASSERT_ALWAYS (nl == 0);
  if (dif1)
  {
      ALLOC_LOCK (lock->mutex);
    __GMP_FREE_FUNC_LIMBS (tmp, Kl + 1);
      ALLOC_UNLOCK (lock->mutex);
  }
}

static void
mpn_mul_fft_decompose_para (mp_ptr A, mp_ptr *Ap, mp_size_t K, mp_size_t nprime,
		       mp_srcptr *pn, mp_size_t *pnl, mp_size_t l, mp_size_t Mp, mp_ptr T_local, mp_lock_t *lock, int id
#ifdef CHRONOGRAM
           , double *chrono, int iteration
#endif
           )
{
  mp_ptr tmp;
  mp_size_t Kl = K * l;
  //TMP_DECL;
  //TMP_MARK;

  mp_size_t nl = *pnl;
  mp_srcptr n = *pn;
  int diff1 = nl > Kl;

  if (diff1) /* normalize {n, nl} mod 2^(Kl*GMP_NUMB_BITS)+1 */
  {
    if ((lock->total == 1 && id == 0) || id == 1)
    {
      mp_size_t dif = nl - Kl;
      mp_limb_signed_t cy;

      tmp = __GMP_ALLOCATE_FUNC_LIMBS (Kl + 1);
      if (dif > Kl)
	{
	  int subp = 0;

	  cy = mpn_sub_n (tmp, n, n + Kl, Kl);
	  n += 2 * Kl;
	  dif -= Kl;

	  /* now dif > 0 */
	  while (dif > Kl)
	    {
	      if (subp)
		cy += mpn_sub_n (tmp, tmp, n, Kl);
	      else
		cy -= mpn_add_n (tmp, tmp, n, Kl);
	      subp ^= 1;
	      n += Kl;
	      dif -= Kl;
	    }
	  /* now dif <= Kl */
	  if (subp)
	    cy += mpn_sub (tmp, tmp, Kl, n, dif);
	  else
	    cy -= mpn_add (tmp, tmp, Kl, n, dif);
	  if (cy >= 0)
	    cy = mpn_add_1 (tmp, tmp, Kl, cy);
	  else
	    cy = mpn_sub_1 (tmp, tmp, Kl, -cy);

    tmp[Kl] = cy;
	}
      else /* dif <= Kl, i.e. nl <= 2 * Kl */
	{
	  cy = mpn_sub (tmp, n, Kl, n + Kl, dif);
    tmp[Kl] = 0;
    MPN_INCR_U(tmp, Kl, cy);
	}
      *pn = tmp;
    }
#ifdef CHRONOGRAM
  chrono[1 + 4 * iteration] = wct_seconds ();
#endif
    mpn_fft_sync(lock);
#ifdef CHRONOGRAM
  chrono[2 + 4 * iteration] = wct_seconds ();
#endif
    n = *pn;
    nl = Kl + 1;
  }
#ifdef CHRONOGRAM
  else
  {
    chrono[1 + 4 * iteration] = wct_seconds ();
    chrono[2 + 4 * iteration] = wct_seconds ();
  }
#endif

  mp_size_t i, j, r, maxi;

  r = (K + lock->total - 1) / lock->total;
  maxi = r * (id + 1);
  if (maxi > K)
  {
    maxi = K;
  }

  MPN_ZERO (T_local + l, nprime + 1 - l);
  
  for (i = r * id; i < maxi; i++)
    {
      mp_ptr currentA = A + i * (nprime + 1);
      Ap[i] = currentA;
      /* store the next M bits of n into A[0..nprime] */
      if (nl > l * i) /* nl is the number of remaining limbs */
  {
    j = (l + l * i <= nl && i < K - 1) ? l : (nl - l * i); /* store j next limbs */
    MPN_COPY (T_local, n + i * l, j);
    MPN_ZERO (T_local + j, l + 1 - j);
    mpn_fft_mul_2exp_modF (currentA, T_local, i * Mp, nprime);
  }
      else
  MPN_ZERO (currentA, nprime + 1);
    }

  if (diff1)
  {
#ifdef CHRONOGRAM
    chrono[3 + 4 * iteration] = wct_seconds ();
#endif
    mpn_fft_sync(lock);
#ifdef CHRONOGRAM
    chrono[4 + 4 * iteration] = wct_seconds ();
#endif

    if ((lock->total == 1 && id == 0) || id == 1)
    {
      __GMP_FREE_FUNC_LIMBS (tmp, Kl + 1);
    }
  } else {
#ifdef CHRONOGRAM
    chrono[3 + 4 * iteration] = wct_seconds ();
    chrono[4 + 4 * iteration] = chrono[3 + 4 * iteration];
#endif
  }
  //TMP_FREE;
}

static void
mpn_mul_fft_decompose_para2 (mp_ptr A, mp_ptr *Ap, mp_size_t K, mp_size_t nprime,
		       mp_srcptr *pn, mp_size_t *pnl, mp_size_t l, mp_size_t Mp, mp_ptr T_local, mp_lock_t *lock, int id
#ifdef CHRONOGRAM
           , double *chrono, int iteration
#endif
           )
{
  mp_size_t Kl = K * l;
  if (*pnl <= Kl || *pnl > (Kl << 1))
  {
    mpn_mul_fft_decompose_para(A, Ap, K, nprime, pn, pnl, l, Mp, T_local, lock, id
#ifdef CHRONOGRAM
           , chrono, iteration
#endif
    );
  }
  else
  {
    mp_size_t r, pos, maxi, diff, j;
    mp_limb_t comma = 2;

    r = (K + lock->total - 1) / lock->total;
    maxi = r * (id + 1);
    if (maxi > K) {
      maxi = K;
    }

    pos = r * id * l - 1;
    diff = *pnl - Kl;
    while (pos >= 0 && comma == 2) {
      if (((pos < diff) ? (*pn)[Kl + pos] : 0)  == (*pn)[pos])
      {
        pos -= 1;
      }
      else
      {
        comma = ((pos < diff) ? (*pn)[Kl + pos] : 0) > (*pn)[pos];
      }
    }

    if (comma == 2) {
      comma = 0;
    }

#ifdef CHRONOGRAM
  chrono[1 + 4 * iteration] = wct_seconds ();
  chrono[2 + 4 * iteration] = wct_seconds ();
#endif

    for (pos = r * id; pos < maxi; pos++)
    {
      Ap[pos] = A + (nprime + 1) * pos;
      if (pos * l < diff)
      {
        j = (l + l * pos <= diff && pos < K - 1) ? l : (diff - l * pos);
        mp_size_t c = mpn_sub(T_local, *pn + pos * l, l, *pn + pos * l + Kl, j);
        comma = c + mpn_sub_1(T_local, T_local, l, comma);
        MPN_ZERO (T_local + l, nprime + 1 - l);
        mpn_fft_mul_2exp_modF (Ap[pos], T_local, pos * Mp, nprime);
      }
      else if (comma == 0)
      {
        MPN_COPY (T_local, *pn + pos * l, l);
        MPN_ZERO (T_local + l, nprime + 1 - l);
        mpn_fft_mul_2exp_modF (Ap[pos], T_local, pos * Mp, nprime);
      }
      else if (comma == 1)
      {
        comma = mpn_sub_1 (T_local, *pn + pos * l, l, comma);
        MPN_ZERO (T_local + l, nprime + 1 - l);
        mpn_fft_mul_2exp_modF (Ap[pos], T_local, pos * Mp, nprime);
      }
    }
#ifdef CHRONOGRAM
  chrono[3 + 4 * iteration] = wct_seconds ();
#endif
    mpn_fft_sync(lock);
#ifdef CHRONOGRAM
  chrono[4 + 4 * iteration] = wct_seconds ();
#endif

    if (pos == K && r * id != K)
    {
      pos = 0;
      while (comma == 1) {
        mpn_fft_div_2exp_modF(T_local, Ap[pos], pos * Mp, nprime);
        comma = mpn_add_1(T_local, T_local, l + (pos == K - 1), comma);
        mpn_fft_mul_2exp_modF (Ap[pos], T_local, pos * Mp, nprime);
        pos += 1;
      }
    }
  }
}

/* op <- n*m mod 2^N+1 with fft of size 2^k where N=pl*GMP_NUMB_BITS
   op is pl limbs, its high bit is returned.
   One must have pl = mpn_fft_next_size (pl, k).
   T must have space for 2 * (nprime + 1) limbs.
*/

static mp_limb_t
mpn_mul_fft_internal (mp_ptr op, mp_size_t pl, int k,
		      mp_ptr *Ap, mp_ptr *Bp, mp_ptr A, mp_ptr B,
		      mp_size_t nprime, mp_size_t l, mp_size_t Mp,
		      int **fft_l, mp_ptr T, int sqr, mp_lock_t* lock)
{
  mp_size_t K, i, pla, lo, sh, j;
  mp_ptr p;
  mp_limb_t cc;

  K = (mp_size_t) 1 << k;

  /* direct fft's */
  mpn_fft_fft (Ap, K, fft_l + k, 2 * Mp, nprime, 1, T);
  if (!sqr)
    mpn_fft_fft (Bp, K, fft_l + k, 2 * Mp, nprime, 1, T);

  /* term to term multiplications */
  mpn_fft_mul_modF_K (Ap, sqr ? Ap : Bp, fft_l, nprime, K, lock);

  /* inverse fft's */
  mpn_fft_fftinv (Ap, K, 2 * Mp, nprime, T);

  /* division of terms after inverse fft */
  Bp[0] = T + nprime + 1;
  mpn_fft_div_2exp_modF (Bp[0], Ap[0], k, nprime);
  for (i = 1; i < K; i++)
    {
      Bp[i] = Ap[i - 1];
      mpn_fft_div_2exp_modF (Bp[i], Ap[i], k + (K - i) * Mp, nprime);
    }

  /* addition of terms in result p */
  MPN_ZERO (T, nprime + 1);
  pla = l * (K - 1) + nprime + 1; /* number of required limbs for p */
  p = B; /* B has K*(n' + 1) limbs, which is >= pla, i.e. enough */
  MPN_ZERO (p, pla);
  cc = 0; /* will accumulate the (signed) carry at p[pla] */
  for (i = K - 1, lo = l * i + nprime,sh = l * i; i >= 0; i--,lo -= l,sh -= l)
    {
      mp_ptr n = p + sh;

      j = (K - i) & (K - 1);

      if (mpn_add_n (n, n, Bp[j], nprime + 1))
	cc += mpn_add_1 (n + nprime + 1, n + nprime + 1,
			  pla - sh - nprime - 1, CNST_LIMB(1));
      T[2 * l] = i + 1; /* T = (i + 1)*2^(2*M) */
      if (mpn_cmp (Bp[j], T, nprime + 1) > 0)
	{ /* subtract 2^N'+1 */
	  cc -= mpn_sub_1 (n, n, pla - sh, CNST_LIMB(1));
	  cc -= mpn_sub_1 (p + lo, p + lo, pla - lo, CNST_LIMB(1));
	}
    }
  if (cc == -CNST_LIMB(1))
    {
      if ((cc = mpn_add_1 (p + pla - pl, p + pla - pl, pl, CNST_LIMB(1))))
	{
	  /* p[pla-pl]...p[pla-1] are all zero */
	  mpn_sub_1 (p + pla - pl - 1, p + pla - pl - 1, pl + 1, CNST_LIMB(1));
	  mpn_sub_1 (p + pla - 1, p + pla - 1, 1, CNST_LIMB(1));
	}
    }
  else if (cc == 1)
    {
      if (pla >= 2 * pl)
	{
	  while ((cc = mpn_add_1 (p + pla - 2 * pl, p + pla - 2 * pl, 2 * pl, cc)))
	    ;
	}
      else
	{
	  cc = mpn_sub_1 (p + pla - pl, p + pla - pl, pl, cc);
	  ASSERT (cc == 0);
	}
    }
  else
    ASSERT (cc == 0);

  /* here p < 2^(2M) [K 2^(M(K-1)) + (K-1) 2^(M(K-2)) + ... ]
     < K 2^(2M) [2^(M(K-1)) + 2^(M(K-2)) + ... ]
     < K 2^(2M) 2^(M(K-1))*2 = 2^(M*K+M+k+1) */
  return mpn_fft_norm_modF (op, pl, p, pla);
}

static void
mpn_mul_fft_internal_para (mp_ptr op, mp_size_t pl, int k,
		      mp_ptr *Ap, mp_ptr *Bp, mp_ptr A, mp_ptr B,
		      mp_size_t nprime, mp_size_t l, mp_size_t Mp,
		      int **fft_l, mp_ptr tp_local, int sqr,
          mp_lock_t *lock, int id, mp_limb_t *out
#ifdef CHRONOGRAM
          , double *chrono
#endif          
          )
{
  int nt = lock->total;
  mp_size_t K, i, pla, lo, sh, j;
  mp_ptr p;
  mp_limb_t cc;
#ifdef TRACE_TIME
  char *trace_time = getenv("TRACE_TIME");
  char *trace_clock = getenv("TRACE_CLOCK");
  char *trace = getenv("TRACE_POSITION");
  double wct1, wct2, wct3, wct4, wct5, wct6, wct_mid;
  clock_t t1, t2, t3, t4, t5, t6, t_mid;
  wct1 = trace_time == NULL ? 0 : wct_seconds ();
  t1  = trace_clock == NULL ? 0 : clock ();

  if (id == 0 && trace != NULL) {
    printf("ffts\n");
  }
#endif


  K = (mp_size_t) 1 << k;

#ifdef TRACE_TIME
  wct_mid = trace_time == NULL ? 0 : wct_seconds ();
  t_mid  = trace_clock == NULL ? 0 : clock ();
#endif


 if (!sqr) {
    if (number_of_threads > DISP_CEIL_FFT)
    {
      if (id & 1) {
        mpn_fft_bailey (Ap, k, fft_l, 2 * Mp, nprime, tp_local, lock, id >> 1, nt/2
#ifdef CHRONOGRAM
  , chrono
#endif
        );
      } else {
        mpn_fft_bailey (Bp, k, fft_l, 2 * Mp, nprime, tp_local, lock, id >> 1, (nt + 1)/2
#ifdef CHRONOGRAM
  , chrono
#endif
        );
      }
    }
    else
    {
      mpn_fft_bailey (Bp, k, fft_l, 2 * Mp, nprime, tp_local, lock, id, nt
#ifdef CHRONOGRAM
  , chrono
#endif
        );
      mpn_fft_bailey (Ap, k, fft_l, 2 * Mp, nprime, tp_local, lock, id, nt
#ifdef CHRONOGRAM
  , chrono
#endif
        );
    }
 } else {
  mpn_fft_bailey (Ap, k, fft_l, 2 * Mp, nprime, tp_local, lock, id, nt
#ifdef CHRONOGRAM
  , chrono
#endif
        );
 }



#ifdef CHRONOGRAM
  chrono[13] = wct_seconds ();
#endif
    mpn_fft_sync(lock);
#ifdef CHRONOGRAM
  chrono[14] = wct_seconds ();
#endif

#ifdef TRACE_TIME
  wct2 = trace_time == NULL ? 0 : wct_seconds ();
  t2  = trace_clock == NULL ? 0 : clock ();
  if (id == 0 && trace != NULL) {
    printf("conv\n");
  }
#endif
  /* term to term multiplications */
  mpn_fft_mul_modF_K_para (Ap, sqr ? Ap : Bp, fft_l, nprime, K, lock, id);
#ifdef CHRONOGRAM
  chrono[15] = wct_seconds ();
#endif
  mpn_fft_sync(lock);
#ifdef CHRONOGRAM
  chrono[16] = wct_seconds ();
#endif


#ifdef TRACE_TIME
  wct3 = trace_time == NULL ? 0 : wct_seconds ();
  t3  = trace_clock == NULL ? 0 : clock ();
  if (id == 0 && trace != NULL) {
    printf("fft_inv\n");
  }
#endif

  /* inverse fft's */
  mpn_fft_baileyinv (Ap, k, fft_l, 2 * Mp, nprime, tp_local, lock, id

#ifdef CHRONOGRAM
  , chrono
#endif
        );

#ifdef CHRONOGRAM
  chrono[19] = wct_seconds ();
#endif
  mpn_fft_sync(lock);
#ifdef CHRONOGRAM
  chrono[20] = wct_seconds ();
#endif
  if (id == 0)
  {
    (*out) = 0;
  }

#ifdef TRACE_TIME
  wct4 = trace_time == NULL ? 0 : wct_seconds ();
  t4  = trace_clock == NULL ? 0 : clock ();
  if (id == 0 && trace != NULL) {
    printf("div and add\n");
  }
#endif


  /* division of terms after inverse fft */
  double timer, timer1, timer2, timer3, timer4;
  /* addition of terms in result p */
  pla = l * (K - 1) + nprime + 1; /* number of required limbs for p */
  p = B; /* B has K*(n' + 1) limbs, which is >= pla, i.e. enough */
  cc = 0; /* will accumulate the (signed) carry at p[pla] */


  mp_size_t j_1, r, maxi;
  mp_ptr n;
  mp_size_t K2 = K >> 1;

  if (id == 0)
  {
    Bp[0] = Ap[0];
    mpn_fft_div_2exp_modF(tp_local, Ap[0], k, nprime);
    mpn_copyi(Bp[0], tp_local, nprime + 1);
  }

  r = (K + lock->total - 1) / lock->total;
  maxi = r * (id + 1);
  if (maxi > K) {maxi = K;};

  for (i = r * id + !(r * id); i < maxi; i++)
  {
    mpn_fft_div_2exp_modF (tp_local, Ap[i], k + (K - i) * Mp, nprime);
    mpn_copyi(Ap[i], tp_local, nprime + 1);
    Bp[i] = Ap[i];
  }

#ifdef CHRONOGRAM
  chrono[21] = wct_seconds ();
#endif
  mpn_fft_sync(lock);
#ifdef CHRONOGRAM
  chrono[22] = wct_seconds ();
#endif

#ifdef TRACE_TIME
    wct5 = trace_time == NULL ? 0 : wct_seconds ();
    t5  = trace_clock == NULL ? 0 : clock ();
#endif
  if (id == 0)
  {
    MPN_ZERO(p, l);
  }

  K2 = K >> 1;
  MPN_ZERO (tp_local, nprime + 1);
  timer1 = wct_seconds ();
  timer = timer1;

  for (j = r * id; j < maxi; j++)
  {
      i = (K - j) & (K - 1);
      tp_local[2 * l] = i + 1;
      if (mpn_cmp (Bp[j], tp_local, nprime + 1) > 0)
        {
          mpn_neg(Bp[j], Bp[j], nprime + 1);
          Bp[j][nprime] += 1;
          MPN_INCR_U(Bp[j], nprime + 1, 1);
          Ap[j] = NULL;
        }
  }
  tp_local[2 * l] = 0;

#ifdef CHRONOGRAM
  chrono[23] = wct_seconds ();
#endif
  mpn_fft_sync(lock);
#ifdef CHRONOGRAM
  chrono[24] = wct_seconds ();
#endif

  timer2 = wct_seconds ();

  r = (K2 + lock->total - 1) / lock->total;

  maxi = r * (id + 1);
  if (maxi > K2) {
    maxi = K2;
  }

  for (j = r * id * 2 + 1; j <= maxi << 1; j+=2)
    {
      // cc is not used in the loop, only the sum is important and at the end

      i = (K - j) & (K - 1);
      j_1 = j - 1; //(K - i - 1) & (K - 1);
      mp_ptr n = p + l * i;
      if (Ap[j] == NULL)
      {
        if (mpn_neg (n, Bp[j], 2 * l))
          MPN_INCR_U (Bp[j] + 2 * l, nprime + 1 - 2 * l, CNST_LIMB(1));
      }
      else
      {
        mpn_copyi(n, Bp[j], 2 * l);
      }

      if (i < K - 1)
      {
        if ((Ap[j] == NULL) == (Ap[j_1] == NULL))
        {
          mp_size_t t = mpn_add_n(Bp[j_1] + l, Bp[j_1] + l, Bp[j] + 2 * l, nprime + 1 - 2 * l);
          MPN_INCR_U(Bp[j_1] + nprime + 1 - l, l, t);
        }
        else
        {
          mpn_copyi(tp_local + l, Bp[j] + 2 * l, nprime + 1 - 2 * l);
          if (mpn_cmp(Bp[j_1] + l, tp_local + l, nprime + 1 - l) >= 0)
          {
            mp_size_t c = mpn_sub_n(Bp[j_1] + l, Bp[j_1] + l, tp_local + l, nprime + 1 - l);
            ASSERT_ALWAYS(c == 0);
          }
          else
          {
            mpn_sub_n(Bp[j_1], tp_local, Bp[j_1], nprime + 1);
            Ap[j_1] = Ap[j];
          }
        }

      }
      else
      {
        if (Ap[j] == NULL)
        {
          if (mpn_sub_n (n + 2 * l, n + 2 * l, Bp[j] + 2 * l, nprime + 1 - 2 * l))
            cc = -mpn_sub_1 (n + nprime + 1, n + nprime + 1,
                pla - l * i - nprime - 1, CNST_LIMB(1));
        }
        else
        {
          mpn_copyi (n + 2 * l, Bp[j] + 2 * l, nprime + 1 - 2 * l);
        }
      }
    }
#ifdef CHRONOGRAM
  chrono[25] = wct_seconds ();
#endif
  mpn_fft_sync(lock);
#ifdef CHRONOGRAM
  chrono[26] = wct_seconds ();
#endif

  timer3 = wct_seconds ();
  for (j_1 = r * id; j_1 < maxi; j_1++)
    { 
      j = j_1 << 1;
      i = (K - j) & (K - 1);
      //lo = l * i + nprime; // = sh + nprime
      //sh = l * i;
      // cc is not used in the loop, only the sum is important and at the end
      mp_ptr n = p + l * i;
      if (Ap[j] == NULL)
      {
        if (mpn_sub_n (n, n, Bp[j], 2 * l))
          MPN_INCR_U(Bp[j] + 2 * l, nprime + 1 - 2 * l, CNST_LIMB(1));
      }
      else
      {
          if (mpn_add_n (n, n, Bp[j], 2 * l))
            MPN_INCR_U(Bp[j] + 2 * l, nprime + 1 - 2 * l, CNST_LIMB(1));
      }
    }

#ifdef CHRONOGRAM
  chrono[27] = wct_seconds ();
#endif
  mpn_fft_sync(lock);
#ifdef CHRONOGRAM
  chrono[28] = wct_seconds ();
#endif

  timer4 = wct_seconds ();

#ifdef TRACE_TIME
  clock_t tmid2  = trace_clock == NULL ? 0 : clock ();
  double wctmid2 = wct_seconds ();
#endif
  if (id == 0)
  {
    for (i = K - 2; i >= 0; i-=2)
    {
      j = (K - i) & (K - 1);
      lo = l * i + nprime; // = sh + nprime
      sh = l * i;
      // cc is not used in the loop, only the sum is important and at the end
      mp_ptr n = p + sh;

      if (Ap[j] == NULL)
      {
        if (mpn_sub_n (n + 2 * l, n + 2 * l, Bp[j] + 2 * l, nprime + 1 - 2 * l))
          cc -= mpn_sub_1 (n + nprime + 1, n + nprime + 1,
              pla - sh - nprime - 1, CNST_LIMB(1));
      }
      else
      {
        if (mpn_add_n (n + 2 * l, n + 2 * l, Bp[j] + 2 * l, nprime + 1 - 2 * l))
          cc += mpn_add_1 (n + nprime + 1, n + nprime + 1,
              pla - sh - nprime - 1, CNST_LIMB(1));
      }
    }

#ifdef CHRONOGRAM
  chrono[29] = wct_seconds ();
#endif

    if (cc == -CNST_LIMB(1))
      {
        if ((cc = mpn_add_1 (p + pla - pl, p + pla - pl, pl, CNST_LIMB(1))))
    {
      /* p[pla-pl]...p[pla-1] are all zero */
      mpn_sub_1 (p + pla - pl - 1, p + pla - pl - 1, pl + 1, CNST_LIMB(1));
      mpn_sub_1 (p + pla - 1, p + pla - 1, 1, CNST_LIMB(1));
    }
      }
    else if (cc == 1)
      {
        if (pla >= 2 * pl)
    {
      while ((cc = mpn_add_1 (p + pla - 2 * pl, p + pla - 2 * pl, 2 * pl, cc)))
        ;
    }
        else
    {
      cc = mpn_sub_1 (p + pla - pl, p + pla - pl, pl, cc);
      ASSERT (cc == 0);
    }
      }
    else
      ASSERT (cc == 0);
  }
#ifdef CHRONOGRAM
  else {
    chrono[29] = wct_seconds ();
  }

  chrono[30] = wct_seconds ();

#endif


  mpn_fft_sync(lock);

#ifdef CHRONOGRAM
  chrono[31] = wct_seconds ();
#endif

#ifdef TRACE_TIME
  t6  = trace_clock == NULL ? 0 : clock ();
  if (id == 0 && trace != NULL) {
    printf("finished mul\n");
  }
  if (id == 0 && trace_time != NULL) {
    wct6 = wct_seconds ();
    printf("fft1 : %.3fs, fft2 : %.3fs\n", wct_mid - wct1, wct2 - wct_mid);
    printf("ffts : %.5fs, conv : %.5fs, fft_inv : %.5fs, division : %.5fs, add : %.5f\n", wct2 - wct1, wct3 - wct2, wct4 - wct3, wct5 - wct4, wct6 - wct5);
    printf("add1 : %.3fs, add2 : %.3fs\n", wctmid2 - wct5, wct6 - wctmid2);
    printf("%.3fs, %.3fs, %.3fs\n", timer2 - timer1, timer3 - timer2, timer4 - timer3);
    printf("%.3fs, %.3fs, %.3fs\n", timer - wct5, timer1 - timer, wctmid2 - timer4);
  }
  if (id == 0 && trace_clock != NULL) {
    printf("fft1 : %.3fs, fft2 : %.3fs\n", (double)(t_mid - t1)/ CLOCKS_PER_SEC, (double)(t2 - t_mid) / CLOCKS_PER_SEC);
    printf("ffts : %.3fs, conv : %.3fs, fft_inv : %.3fs, division : %.3fs, add : %.3f\n", (double)(t2 - t1) / CLOCKS_PER_SEC, (double)(t3 - t2) / CLOCKS_PER_SEC, (double)(t4 - t3) / CLOCKS_PER_SEC, (double)(t5 - t4) / CLOCKS_PER_SEC, (double)(t6 - t5) / CLOCKS_PER_SEC);
  }
#endif

  /* here p < 2^(2M) [K 2^(M(K-1)) + (K-1) 2^(M(K-2)) + ... ]
     < K 2^(2M) [2^(M(K-1)) + 2^(M(K-2)) + ... ]
     < K 2^(2M) 2^(M(K-1))*2 = 2^(M*K+M+k+1) */
  mpn_fft_norm_modF_para (op, pl, p, pla, out, id, lock
#ifdef CHRONOGRAM
  , chrono
#endif
  );


#ifdef CHRONOGRAM
  chrono[34] = wct_seconds ();
#endif

  return;
}


/* return the lcm of a and 2^k */
static mp_bitcnt_t
mpn_mul_fft_lcm (mp_bitcnt_t a, int k)
{
  mp_bitcnt_t l = k;

  while (a % 2 == 0 && k > 0)
    {
      a >>= 1;
      k --;
    }
  return a << l;
}

struct common_args
{
  mp_ptr op;
  mp_size_t pl;
  int k;
	mp_ptr *Ap;
  mp_ptr *Bp;
  mp_ptr A;
  mp_ptr B;
  mp_srcptr *n;
  mp_size_t *nl;
  mp_srcptr *m;
  mp_size_t *ml;
	mp_size_t nprime;
  mp_size_t l;
  mp_size_t Mp;
	int **fft_l;
  mp_ptr T;
  int sqr;
  mp_lock_t *lock;
  mp_limb_t *out;
#ifdef CHRONOGRAM
  double **chrono;
#endif
};

struct internal_args 
{
  struct common_args* ca;
  int id;
};

void *
mpn_mul_fft_internal_outer(void *ia)
{
  struct internal_args *args = (struct internal_args*)ia;
  mpn_fft_sync(args->ca->lock);
  double wct0 = wct_seconds ();
#ifdef CHRONOGRAM
  args->ca->chrono[args->id][0] = wct0;
#endif
  mp_size_t mul = args->id == 0 ? 2 : 1;
  mp_ptr T = malloc(mul * (args->ca->nprime + 1) * sizeof(mp_size_t));//args->ca->T + (args->ca->nprime + 1) * args->id;
  double wct1, wct2;
  wct1 = wct_seconds ();
  mpn_mul_fft_decompose_para2(args->ca->A, args->ca->Ap,
    1 << args->ca->k, args->ca->nprime, args->ca->n,
    args->ca->nl, args->ca->l, args->ca->Mp, T,
    args->ca->lock, args->id
#ifdef CHRONOGRAM
  , args->ca->chrono[args->id], 0
#endif
  );

  if (!args->ca->sqr)
  {
    mpn_mul_fft_decompose_para2(args->ca->B, args->ca->Bp,
      1 << args->ca->k, args->ca->nprime, args->ca->m,
      args->ca->ml, args->ca->l, args->ca->Mp, T,
      args->ca->lock, args->id
#ifdef CHRONOGRAM
      , args->ca->chrono[args->id], 1
#endif
      );
  }

#ifdef CHRONOGRAM
  args->ca->chrono[args->id][9] = wct_seconds ();
#endif
  mpn_fft_sync(args->ca->lock);
#ifdef CHRONOGRAM
  args->ca->chrono[args->id][10] = wct_seconds ();
#endif

  if (args->id == 0 && NULL != getenv ("TRACE_TIME"))
  {
    wct2 = wct_seconds ();
    printf("decomp : %.5fs %.3fs\n", wct2 - wct1, wct1 - wct0);
  }


  mpn_mul_fft_internal_para(args->ca->op, args->ca->pl,
    args->ca->k, args->ca->Ap, args->ca->Bp, args->ca->A, args->ca->B,
    args->ca->nprime, args->ca->l, args->ca->Mp, args->ca->fft_l,
    T, args->ca->sqr, args->ca->lock, args->id, args->ca->out
#ifdef CHRONOGRAM
    , args->ca->chrono[args->id]
#endif
    );
  free(T);
  return NULL;
}



mp_limb_t
mpn_mul_fft (mp_ptr op, mp_size_t pl,
	     mp_srcptr n, mp_size_t nl,
	     mp_srcptr m, mp_size_t ml,
	     int k)
{
  int i;
  mp_size_t K, maxLK;
  mp_size_t N, Nprime, nprime, M, Mp, l;
  mp_ptr *Ap, *Bp, A, T, B;
  int **fft_l, *tmp;
  int sqr = (n == m && nl == ml);
  mp_limb_t h;
  
  if (mpn_mul_fft_topology == NULL)
  {
    hwloc_topology_init(&mpn_mul_fft_topology);
    hwloc_topology_load(mpn_mul_fft_topology);
  }

#ifdef TRACE_TIME
  char *trace_time;
  trace_time = getenv ("TRACE_TIME");
  double wct1, wct2, wct3;
  wct1 = wct_seconds ();
#endif
  TMP_DECL;

  TRACE (printf ("\nmpn_mul_fft pl=%ld nl=%ld ml=%ld k=%d\n", pl, nl, ml, k));
  ASSERT_ALWAYS (mpn_fft_next_size (pl, k) == pl);

  TMP_MARK;
  N = pl * GMP_NUMB_BITS;
  fft_l = TMP_BALLOC_TYPE (k + 1, int *);
  tmp = TMP_BALLOC_TYPE ((size_t) 2 << k, int);
  for (i = 0; i <= k; i++)
    {
      fft_l[i] = tmp;
      tmp += (mp_size_t) 1 << i;
    }

  mpn_fft_initl (fft_l, k);
  K = (mp_size_t) 1 << k;
  M = N >> k;	/* N = 2^k M */
  l = 1 + (M - 1) / GMP_NUMB_BITS;
  maxLK = mpn_mul_fft_lcm (GMP_NUMB_BITS, k); /* lcm (GMP_NUMB_BITS, 2^k) */

  Nprime = (1 + (2 * M + k + 2) / maxLK) * maxLK;
  /* Nprime = ceil((2*M+k+3)/maxLK)*maxLK; */
  nprime = Nprime / GMP_NUMB_BITS;
  TRACE (printf ("N=%ld K=%ld, M=%ld, l=%ld, maxLK=%ld, Np=%ld, np=%ld\n",
		 N, K, M, l, maxLK, Nprime, nprime));
  /* we should ensure that recursively, nprime is a multiple of the next K */
  if (nprime >= (sqr ? SQR_FFT_MODF_THRESHOLD : MUL_FFT_MODF_THRESHOLD))
    {
      mp_size_t K2;
      for (;;)
	{
	  K2 = (mp_size_t) 1 << mpn_fft_best_k (nprime, sqr);
	  if ((nprime & (K2 - 1)) == 0)
	    break;
	  nprime = (nprime + K2 - 1) & -K2;
	  Nprime = nprime * GMP_LIMB_BITS;
	  /* warning: since nprime changed, K2 may change too! */
	}
      TRACE (printf ("new maxLK=%ld, Np=%ld, np=%ld\n", maxLK, Nprime, nprime));
    }
  ASSERT_ALWAYS (nprime < pl); /* otherwise we'll loop */

  T = TMP_BALLOC_LIMBS ((nprime + 1) * (1 + NB_THREAD));// (1 + omp_get_max_threads()));
  Mp = Nprime >> k;

  TRACE (printf ("%ldx%ld limbs -> %ld times %ldx%ld limbs (%1.2f)\n",
		pl, pl, K, nprime, nprime, 2.0 * (double) N / Nprime / K);
	 printf ("   temp space %ld\n", 2 * K * (nprime + 1)));

  A = TMP_BALLOC_LIMBS (K * (nprime + 1));
  Ap = TMP_BALLOC_MP_PTRS (K);
  if (sqr)
    {
      mp_size_t pla;
      pla = l * (K - 1) + nprime + 1; /* number of required limbs for p */
      B = TMP_BALLOC_LIMBS (pla);
      Bp = TMP_BALLOC_MP_PTRS (K);
    }
  else
    {
      B = TMP_BALLOC_LIMBS (K * (nprime + 1));
      Bp = TMP_BALLOC_MP_PTRS (K);
    }


  pthread_t threads[NB_THREAD];
  pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

#ifdef CHRONOGRAM
  double wct4 = wct_seconds ();
#define NB_STEPS 35
  double** timers = malloc(NB_THREAD * sizeof(double*));
  
  for (mp_size_t i = 0; i < NB_THREAD; i++) {
    timers[i] = malloc(NB_STEPS * sizeof(double));
  }
#endif

  mp_lock_t lock = {
    .mutex = &mutex,
    .total = NB_THREAD,
  };

  struct common_args com_arg  = {
    .op = op,
    .pl = pl,
    .k = k,
    .Ap = Ap,
    .Bp = Bp,
    .A = A,
    .B = B,
    .n = &n,
    .nl = &nl,
    .m = &m,
    .ml = &ml,
    .nprime = nprime,
    .l = l,
    .Mp = Mp,
    .fft_l = fft_l,
    .T = T,
    .sqr = sqr,
    .lock = &lock,
    .out = &h,
#ifdef CHRONOGRAM
    .chrono = timers,
#endif
  };

  pthread_barrier_init(&lock.barrier, NULL, lock.total);

  struct internal_args args_list[NB_THREAD];
  for (int i = 0; i < NB_THREAD; i++)
  {
    args_list[i].id = i;
    args_list[i].ca = &com_arg;
  }

  hwloc_cpuset_t set = hwloc_bitmap_alloc();

  wct2 = wct_seconds ();
  if (trace_time != NULL) {
    printf ("\nmpn_mul_fft pl=%ld nl=%ld ml=%ld k=%d\n", pl, nl, ml, k);
    printf("init : %.5fs\n", wct2 - wct1);
  }

  double t_pthreads[NB_THREAD];
  double t_thread0 = wct_seconds ();

  for (int i = 1; i < NB_THREAD; i++)
  {
    pthread_create(&threads[i], NULL, mpn_mul_fft_internal_outer, (void *)(args_list + i));
    t_pthreads[i] = wct_seconds ();

    if (NB_THREAD > DISP_CEIL)
    {
      hwloc_bitmap_only (set, i);
    } else {
      hwloc_bitmap_only (set, i << 1);
    }
    hwloc_set_thread_cpubind(mpn_mul_fft_topology, threads[i], set, HWLOC_CPUBIND_THREAD);

  }

  t_pthreads[0] = wct_seconds ();

  hwloc_bitmap_only (set, 0);
  hwloc_set_cpubind(mpn_mul_fft_topology, set, HWLOC_CPUBIND_THREAD);
  hwloc_bitmap_free(set);

  mpn_mul_fft_internal_outer((void *)args_list);

#ifdef CHRONOGRAM
  wct3 = wct_seconds ();
#endif

  for (int i = 1; i < NB_THREAD; i++)
  {
    pthread_join (threads[i], NULL);
  }

#ifdef CHRONOGRAM
  double wct5 = wct_seconds ();
#endif

#ifdef CHRONOGRAM
  if (timers1 == NULL)
  {
    timers1 = timers;
  } else {
    timers2 = timers;
  }
  printf("%.5f %.5f %.5f %.5f %.5f\n", wct1, wct4, wct2, wct3, wct5);
#endif

  TMP_FREE;
  return h;
}

#if WANT_OLD_FFT_FULL

void
mpn_fft_full_chinese(mp_ptr op,
      mp_size_t pl, mp_ptr pad_op,
      mp_size_t pl2, mp_size_t pl3, mp_size_t cc,
      mp_limb_t *out, int id, mp_lock_t* lock)
{
  mp_size_t l;
  mp_size_t c2, oldcc;

#ifdef CHRONOGRAM
  timers3[id][0] = wct_seconds ();
#endif
  mpn_fft_sync(lock);

  mpn_fft_sub_n_para (pad_op, pad_op, op, pl2, out, id, lock
#ifdef CHRONOGRAM
  , timers3[id] + 1
#endif
  );

#ifdef CHRONOGRAM
  timers3[id][3] = wct_seconds ();
#endif
  mpn_fft_sync(lock);
#ifdef CHRONOGRAM
  timers3[id][4] = wct_seconds ();
#endif
  cc = -cc + out[0];    /* lambda - low(mu) */
  /* 0 <= cc <= 1 */
  ASSERT(0 <= cc && cc <= 1);
  l = pl3 - pl2; /* l = pl2 / 2 since pl3 = 3/2 * pl2 */

#ifdef CHRONOGRAM
  timers3[id][5] = wct_seconds ();
#endif
  mpn_fft_sync(lock);
#ifdef CHRONOGRAM
  timers3[id][6] = wct_seconds ();
#endif

  mpn_fft_add_n_para (pad_op, pad_op, op + pl2, l, out + 1, id, lock
#ifdef CHRONOGRAM
  , timers3[id] + 7
#endif
    );

#ifdef CHRONOGRAM
  timers3[id][9] = wct_seconds ();
#endif
  mpn_fft_sync(lock);
#ifdef CHRONOGRAM
  timers3[id][10] = wct_seconds ();
#endif

  c2 = out[1];
  mp_ptr tmp;
  if (id == 0)
  {
    cc = mpn_add_1 (pad_op + l, pad_op + l, l, (mp_limb_t) c2) - cc;
    ASSERT(-1 <= cc && cc <= 1);
    if (cc < 0)
      cc = mpn_add_1 (pad_op, pad_op, pl2, (mp_limb_t) -cc);
    ASSERT(0 <= cc && cc <= 1);
    /* now lambda-mu = {pad_op, pl2} - cc mod 2^(pl2*GMP_NUMB_BITS)+1 */
    oldcc = cc;

#if HAVE_NATIVE_mpn_add_n_sub_n && 0
    c2 = mpn_add_n_sub_n (pad_op + l, pad_op, pad_op, pad_op + l, l);
    cc += c2 >> 1; /* carry out from high <- low + high */
    c2 = c2 & 1; /* borrow out from low <- low - high */
#else
      tmp = __GMP_ALLOCATE_FUNC_LIMBS (l);
      out[0] = (mp_limb_t)tmp;
  }
    
  mpn_fft_sync(lock);
  tmp = (mp_ptr)out[0];
  mpn_fft_copy_para (tmp, pad_op, l, id, lock);


  mpn_fft_sub_n_para (pad_op,      pad_op, pad_op + l, l, out, id, lock
  #ifdef CHRONOGRAM
      , timers3[id] + 11
  #endif
      );

  mpn_fft_add_n_para (pad_op + l, tmp,    pad_op + l, l, out + 1, id, lock
  #ifdef CHRONOGRAM
      , timers3[id] + 11
  #endif
    );

  mpn_fft_sync(lock);
        
    if (id == 0) {
        c2 = out[0];
        cc += out[1];
        __GMP_FREE_FUNC_LIMBS (tmp, l);
#endif


    /* first normalize {pad_op, pl2} before dividing by 2: c2 is the borrow
      at pad_op + l, cc is the carry at pad_op + pl2 */
    /* 0 <= cc <= 2 */
    cc -= mpn_sub_1 (pad_op + l, pad_op + l, l, (mp_limb_t) c2);
    /* -1 <= cc <= 2 */
    if (cc > 0)
      cc = -mpn_sub_1 (pad_op, pad_op, pl2, (mp_limb_t) cc);
    /* now -1 <= cc <= 0 */
    if (cc < 0)
      cc = mpn_add_1 (pad_op, pad_op, pl2, (mp_limb_t) -cc);
    /* now {pad_op, pl2} is normalized, with 0 <= cc <= 1 */
    if (pad_op[0] & 1) /* if odd, add 2^(pl2*GMP_NUMB_BITS)+1 */
      cc += 1 + mpn_add_1 (pad_op, pad_op, pl2, CNST_LIMB(1));
    /* now 0 <= cc <= 2, but cc=2 cannot occur since it would give a carry
      out below */
    out[0] = cc;
  }

#ifdef CHRONOGRAM
  timers3[id][11] = wct_seconds ();
#endif
  mpn_fft_sync(lock);
#ifdef CHRONOGRAM
  timers3[id][12] = wct_seconds ();
#endif


  mpn_fft_rshift_para (pad_op, pad_op, pl2, 1, NULL, out[0], id, lock
#ifdef CHRONOGRAM
  , timers3[id] + 13
#endif
  );

#ifdef CHRONOGRAM
  timers3[id][15] = wct_seconds ();
#endif

#ifdef CHRONOGRAM
  timers3[id][16] = wct_seconds ();
#endif

#if 0 
if (id == 0)
  {
    if (cc) /* then cc=1 */
      pad_op [pl2 - 1] |= (mp_limb_t) 1 << (GMP_NUMB_BITS - 1);
  }
#endif

#ifdef CHRONOGRAM
  timers3[id][17] = wct_seconds ();
#endif
  mpn_fft_sync(lock);
#ifdef CHRONOGRAM
  timers3[id][18] = wct_seconds ();
#endif
  mpn_fft_add_n_para (op, op, pad_op, pl2, out + 1, id, lock
#ifdef CHRONOGRAM
  , timers3[id] + 19
#endif
  );

#ifdef CHRONOGRAM
  timers3[id][21] = wct_seconds ();
#endif
  mpn_fft_sync(lock);
#ifdef CHRONOGRAM
  timers3[id][22] = wct_seconds ();
#endif
  c2 = out[1];

    /* since pl2+pl3 >= pl, necessary the extra limbs (including cc) are zero */

  mpn_fft_copy_para (op + pl3, pad_op, pl - pl3, id, lock);

#ifdef CHRONOGRAM
  timers3[id][23] = wct_seconds ();
#endif
  mpn_fft_sync(lock);
#ifdef CHRONOGRAM
  timers3[id][24] = wct_seconds ();
#endif

  if (id == 0)
  {
    ASSERT_MPN_ZERO_P (pad_op + pl - pl3, pl2 + pl3 - pl);
    __GMP_FREE_FUNC_LIMBS (pad_op, pl2);
    /* since the final result has at most pl limbs, no carry out below */
    mpn_add_1 (op + pl2, op + pl2, pl - pl2, (mp_limb_t) c2);
  }
#ifdef CHRONOGRAM
  timers3[id][25] = wct_seconds ();
#endif
}

typedef struct {
  mp_ptr op;
  mp_size_t pl;
  mp_ptr pad_op;
  mp_size_t pl2;
  mp_size_t pl3;
  mp_size_t cc;
  mp_limb_t *out;
  mp_lock_t *lock;
} mpn_fft_chinese_common_args;

typedef struct {
  mpn_fft_chinese_common_args* ca;
  int id;
} mpn_fft_chinese_private_args;

void * mpn_fft_full_chinese_out(void* arg)
{
  mpn_fft_chinese_private_args* pa = arg;
  mpn_fft_full_chinese(pa->ca->op, pa->ca->pl,
    pa->ca->pad_op, pa->ca->pl2, pa->ca->pl3, pa->ca->cc,
    pa->ca->out, pa->id, pa->ca->lock);
}

/* multiply {n, nl} by {m, ml}, and put the result in {op, nl+ml} */
void
mpn_mul_fft_full (mp_ptr op,
		  mp_srcptr n, mp_size_t nl,
		  mp_srcptr m, mp_size_t ml)
{
#ifdef CHRONOGRAM
  double t0, t1, t2, t3;
  t0 = wct_seconds ();
#endif
  mp_ptr pad_op;
  mp_size_t pl, pl2, pl3, l;
  mp_size_t cc, c2, oldcc;
  int k2, k3;
  int sqr = (n == m && nl == ml);

  pl = nl + ml; /* total number of limbs of the result */

  /* perform a fft mod 2^(2N)+1 and one mod 2^(3N)+1.
     We must have pl3 = 3/2 * pl2, with pl2 a multiple of 2^k2, and
     pl3 a multiple of 2^k3. Since k3 >= k2, both are multiples of 2^k2,
     and pl2 must be an even multiple of 2^k2. Thus (pl2,pl3) =
     (2*j*2^k2,3*j*2^k2), which works for 3*j <= pl/2^k2 <= 5*j.
     We need that consecutive intervals overlap, i.e. 5*j >= 3*(j+1),
     which requires j>=2. Thus this scheme requires pl >= 6 * 2^FFT_FIRST_K. */

  /*  ASSERT_ALWAYS(pl >= 6 * (1 << FFT_FIRST_K)); */

  pl2 = (2 * pl - 1) / 5; /* ceil (2pl/5) - 1 */
  do
    {
      pl2++;
      k2 = mpn_fft_best_k (pl2, sqr); /* best fft size for pl2 limbs */
      pl2 = mpn_fft_next_size (pl2, k2);
      pl3 = 3 * pl2 / 2; /* since k>=FFT_FIRST_K=4, pl2 is a multiple of 2^4,
			    thus pl2 / 2 is exact */
      k3 = mpn_fft_best_k (pl3, sqr);
    }
  while (mpn_fft_next_size (pl3, k3) != pl3);

  TRACE (printf ("mpn_mul_fft_full nl=%ld ml=%ld -> pl2=%ld pl3=%ld k=%d\n",
		 nl, ml, pl2, pl3, k2));

  ASSERT_ALWAYS(pl3 <= pl);
#ifdef CHRONOGRAM
  t1 = wct_seconds ();
#endif
  cc = mpn_mul_fft (op, pl3, n, nl, m, ml, k3);     /* mu */

  ASSERT(cc == 0);
  pad_op = __GMP_ALLOCATE_FUNC_LIMBS (pl2);

  cc = mpn_mul_fft (pad_op, pl2, n, nl, m, ml, k2); /* lambda */
#ifdef CHRONOGRAM
  t2 = wct_seconds ();
#endif

  mp_limb_t shared_info[3];
  pthread_mutex_t shared_mutex = PTHREAD_MUTEX_INITIALIZER;
  mp_lock_t lock = {
    .mutex = &shared_mutex,
    .total = NB_THREAD,
  };
  double wct0 = wct_seconds ();
  pthread_barrier_init(&lock.barrier, NULL, lock.total);

#ifdef CHRONOGRAM
#define NB_STEPS_CHINESE 26
  timers3 = malloc(NB_THREAD * sizeof(double*));
  for (int i = 0; i < NB_THREAD; i++)
  {
    timers3[i] = malloc(NB_STEPS_CHINESE * sizeof(double));
  }
#endif

  mpn_fft_chinese_common_args c_args = {
    .op = op,
    .pl = pl,
    .pad_op = pad_op,
    .pl2 = pl2,
    .pl3 = pl3,
    .cc = cc,
    .out = shared_info,
    .lock = &lock,
  };

  pthread_t threads[NB_THREAD];

  mpn_fft_chinese_private_args args[NB_THREAD];
  for (int i = 0; i < NB_THREAD; i++) {
    args[i].id = i;
    args[i].ca = &c_args;
  }

  hwloc_cpuset_t set = hwloc_bitmap_alloc();
  for (int i = 1; i < NB_THREAD; i++) {
    pthread_create(threads + i, NULL, mpn_fft_full_chinese_out, args + i);
    if (NB_THREAD < 16)
    {
      hwloc_bitmap_only (set, i << 1);
    } else {
      hwloc_bitmap_only (set, i);
    }
    hwloc_set_thread_cpubind(mpn_mul_fft_topology, threads[i], set, HWLOC_CPUBIND_THREAD);

  }

  hwloc_bitmap_only (set, 0);
  hwloc_set_cpubind(mpn_mul_fft_topology, set, HWLOC_CPUBIND_THREAD);
  hwloc_bitmap_free(set);
  mpn_fft_full_chinese_out((void*)args);

  for (int i = 1; i < NB_THREAD; i++) {
    pthread_join(threads[i], NULL);
  }
  double wct1 = wct_seconds ();


  printf("end needs : %.5f\n", wct1 - wct0);

#ifdef CHRONOGRAM
  t3 = wct_seconds ();
  printf("%.5f %.5f %.5f %.5f\n", t0, t1, t2, t3);
  for (mp_size_t i = 0; i < NB_THREAD; i++)
  {
    for (mp_size_t j = 0; j < NB_STEPS; j++)
    {
      printf("%.5f ", timers1[i][j]);
    }
    printf("\n");
    free(timers1[i]);
  }
  free(timers1);
  timers1 = NULL;

  for (mp_size_t i = 0; i < NB_THREAD; i++)
  {
    for (mp_size_t j = 0; j < NB_STEPS; j++)
    {
      printf("%.5f ", timers2[i][j]);
    }
    printf("\n");
    free(timers2[i]);
  }
  free(timers2);
  timers2 = NULL;

  for (mp_size_t i = 0; i < NB_THREAD; i++)
  {
    for (mp_size_t j = 0; j < NB_STEPS_CHINESE; j++)
    {
      printf("%.5f ", timers3[i][j]);
    }
    printf("\n");
    free(timers3[i]);
  }
  free(timers3);
  timers3 = NULL;
#endif
}
#endif

#if WANT_OLD_FFT_FULL_NON_PARA
/* multiply {n, nl} by {m, ml}, and put the result in {op, nl+ml} */
void
mpn_mul_fft_full (mp_ptr op,
		  mp_srcptr n, mp_size_t nl,
		  mp_srcptr m, mp_size_t ml)
{
#ifdef CHRONOGRAM
  double t0, t1, t2, t3;
  t0 = wct_seconds ();
#endif
  mp_ptr pad_op;
  mp_size_t pl, pl2, pl3, l;
  mp_size_t cc, c2, oldcc;
  int k2, k3;
  int sqr = (n == m && nl == ml);

  pl = nl + ml; /* total number of limbs of the result */

  /* perform a fft mod 2^(2N)+1 and one mod 2^(3N)+1.
     We must have pl3 = 3/2 * pl2, with pl2 a multiple of 2^k2, and
     pl3 a multiple of 2^k3. Since k3 >= k2, both are multiples of 2^k2,
     and pl2 must be an even multiple of 2^k2. Thus (pl2,pl3) =
     (2*j*2^k2,3*j*2^k2), which works for 3*j <= pl/2^k2 <= 5*j.
     We need that consecutive intervals overlap, i.e. 5*j >= 3*(j+1),
     which requires j>=2. Thus this scheme requires pl >= 6 * 2^FFT_FIRST_K. */

  /*  ASSERT_ALWAYS(pl >= 6 * (1 << FFT_FIRST_K)); */

  pl2 = (2 * pl - 1) / 5; /* ceil (2pl/5) - 1 */
  do
    {
      pl2++;
      k2 = mpn_fft_best_k (pl2, sqr); /* best fft size for pl2 limbs */
      pl2 = mpn_fft_next_size (pl2, k2);
      pl3 = 3 * pl2 / 2; /* since k>=FFT_FIRST_K=4, pl2 is a multiple of 2^4,
			    thus pl2 / 2 is exact */
      k3 = mpn_fft_best_k (pl3, sqr);
    }
  while (mpn_fft_next_size (pl3, k3) != pl3);

  TRACE (printf ("mpn_mul_fft_full nl=%ld ml=%ld -> pl2=%ld pl3=%ld k=%d\n",
		 nl, ml, pl2, pl3, k2));

  ASSERT_ALWAYS(pl3 <= pl);
#ifdef CHRONOGRAM
  t1 = wct_seconds ();
#endif
  cc = mpn_mul_fft (op, pl3, n, nl, m, ml, k3);     /* mu */
  //double wct0 = wct_seconds ();
  ASSERT(cc == 0);
  pad_op = __GMP_ALLOCATE_FUNC_LIMBS (pl2);
  //double wct1 = wct_seconds ();
  cc = mpn_mul_fft (pad_op, pl2, n, nl, m, ml, k2); /* lambda */
#ifdef CHRONOGRAM
  t2 = wct_seconds ();
#endif
  //double wct2 = wct_seconds ();
// Costs a lot 0.013
  cc = -cc + mpn_sub_n (pad_op, pad_op, op, pl2);    /* lambda - low(mu) */
  /* 0 <= cc <= 1 */
  //double wct2bis = wct_seconds ();
  ASSERT(0 <= cc && cc <= 1);
  l = pl3 - pl2; /* l = pl2 / 2 since pl3 = 3/2 * pl2 */
// Costs a lot 0.008
  c2 = mpn_add_n (pad_op, pad_op, op + pl2, l);
  //double wct3 = wct_seconds ();
  cc = mpn_add_1 (pad_op + l, pad_op + l, l, (mp_limb_t) c2) - cc;
  ASSERT(-1 <= cc && cc <= 1);
  if (cc < 0)
    cc = mpn_add_1 (pad_op, pad_op, pl2, (mp_limb_t) -cc);
  ASSERT(0 <= cc && cc <= 1);
  /* now lambda-mu = {pad_op, pl2} - cc mod 2^(pl2*GMP_NUMB_BITS)+1 */
  oldcc = cc;
  //double wct4 = wct_seconds ();
// Costs a lot 0.007
#if HAVE_NATIVE_mpn_add_n_sub_n
  c2 = mpn_add_n_sub_n (pad_op + l, pad_op, pad_op, pad_op + l, l);
  cc += c2 >> 1; /* carry out from high <- low + high */
  c2 = c2 & 1; /* borrow out from low <- low - high */
#else
  {
    mp_ptr tmp;
    TMP_DECL;

    TMP_MARK;
    tmp = TMP_BALLOC_LIMBS (l);
    MPN_COPY (tmp, pad_op, l);
    c2 = mpn_sub_n (pad_op,      pad_op, pad_op + l, l);
    cc += mpn_add_n (pad_op + l, tmp,    pad_op + l, l);
    TMP_FREE;
  }
#endif
  //double wct5 = wct_seconds ();
  c2 += oldcc;
  /* first normalize {pad_op, pl2} before dividing by 2: c2 is the borrow
     at pad_op + l, cc is the carry at pad_op + pl2 */
  /* 0 <= cc <= 2 */
  cc -= mpn_sub_1 (pad_op + l, pad_op + l, l, (mp_limb_t) c2);
  /* -1 <= cc <= 2 */
  if (cc > 0)
    cc = -mpn_sub_1 (pad_op, pad_op, pl2, (mp_limb_t) cc);
  /* now -1 <= cc <= 0 */
  if (cc < 0)
    cc = mpn_add_1 (pad_op, pad_op, pl2, (mp_limb_t) -cc);
  /* now {pad_op, pl2} is normalized, with 0 <= cc <= 1 */
  if (pad_op[0] & 1) /* if odd, add 2^(pl2*GMP_NUMB_BITS)+1 */
    cc += 1 + mpn_add_1 (pad_op, pad_op, pl2, CNST_LIMB(1));
  /* now 0 <= cc <= 2, but cc=2 cannot occur since it would give a carry
     out below */
  //double wct7 = wct_seconds ();
// Cost a lot 0.007
  mpn_rshift (pad_op, pad_op, pl2, 1); /* divide by two */
  //double wct8 = wct_seconds ();
  if (cc) /* then cc=1 */
    pad_op [pl2 - 1] |= (mp_limb_t) 1 << (GMP_NUMB_BITS - 1);
  /* now {pad_op,pl2}-cc = (lambda-mu)/(1-2^(l*GMP_NUMB_BITS))
     mod 2^(pl2*GMP_NUMB_BITS) + 1 */
// Cost a lot 0.013
  c2 = mpn_add_n (op, op, pad_op, pl2); /* no need to add cc (is 0) */
  //double wct8_2 = wct_seconds ();
// Cost a lot 0.020
  /* since pl2+pl3 >= pl, necessary the extra limbs (including cc) are zero */
  MPN_COPY (op + pl3, pad_op, pl - pl3);
  ASSERT_MPN_ZERO_P (pad_op + pl - pl3, pl2 + pl3 - pl);
  //double wct8_3 = wct_seconds ();
  __GMP_FREE_FUNC_LIMBS (pad_op, pl2);
  /* since the final result has at most pl limbs, no carry out below */
  //double wct8_4 = wct_seconds ();
  mpn_add_1 (op + pl2, op + pl2, pl - pl2, (mp_limb_t) c2);
  /*double wct9 = wct_seconds ();
  printf("%.3f %.3f %.3f - %.3f %.3f %.3f - %.3f %.3f %.3f - %.3f %.3f\n",
    wct1 - wct0, wct2bis - wct2, wct3 - wct2bis,
    wct4 - wct3, wct5 - wct4, wct7 - wct5,
    wct8 - wct7, wct8_2 - wct8, wct8_3 - wct8_2,
    wct8_4 - wct8_3, wct9 - wct8_4);*/
#ifdef CHRONOGRAM
  t3 = wct_seconds ();
  printf("%.5f %.5f %.5f %.5f\n", t0, t1, t2, t3);
  for (mp_size_t i = 0; i < NB_THREAD; i++)
  {
    for (mp_size_t j = 0; j < NB_STEPS; j++)
    {
      printf("%.5f ", timers1[i][j]);
    }
    printf("\n");
    free(timers1[i]);
  }
  free(timers1);
  timers1 = NULL;

  for (mp_size_t i = 0; i < NB_THREAD; i++)
  {
    for (mp_size_t j = 0; j < NB_STEPS; j++)
    {
      printf("%.5f ", timers2[i][j]);
    }
    printf("\n");
    free(timers2[i]);
  }
  free(timers2);
  timers2 = NULL;
#endif
}
#endif
