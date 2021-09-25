/* 
gcc -I. $DIR/test-gmp-para.c .libs/libgmp.a -O3 -fopenmp
gcc -I. $DIR/test-gmp-para.c .libs/libgmp.a -O3 -lpthread -lhwloc
*/


#include "gmp.h"
#include <time.h>  /* for clock() */
#include <sys/time.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int nt_opt = 0;
int nb_opt = 0;
int help_opt = 0;
int inc_opt = 0;
int mul_opt = 0;
static struct option longopts[] =
  {
    { "t",    required_argument,  &nt_opt, 1},
    { "n",    required_argument,  &nb_opt, 1},
    { "help", no_argument,        &help_opt,    1},
    { "inc", required_argument,   &inc_opt, 1},
    { "mul", required_argument,   &mul_opt, 1},
    { NULL,   0,                  NULL,          0}

  };

double
seconds ()
{
  return (double) clock () / CLOCKS_PER_SEC;
}

/* borrowed from CADO-NFS (utils/timing.cpp) */
double
wct_seconds (void)
{
    struct timeval tv[1];
    gettimeofday (tv, NULL);
    return (double)tv->tv_sec + (double)tv->tv_usec*1.0e-6;
}

int
main (int argc, char *argv[])
{
  mp_limb_t *a, *b, *d;
  double t, wct;
  int nb_thread = 1;
  int nb_iter = 0;
  unsigned int ratio = 0;
  unsigned int add = 0;
  while (getopt_long_only (argc, argv, "", longopts, NULL) != -1)
    {
      if (nt_opt) {
          if (optarg != NULL) {
            nb_thread = strtol (optarg, 0, 0);
          } else {
            nb_thread = nt_opt;
          }
          nt_opt = 0;
      } else if (nb_opt) {
          if (optarg != NULL) {
            nb_iter = strtol (optarg, 0, 0);
          } else {
            nb_iter = mul_opt;
          }
          nb_opt = 0;
      } else if (mul_opt) {
          if (optarg != NULL) {
            ratio = strtol (optarg, 0, 0);
          } else {
            ratio = mul_opt;
          }
          mul_opt = 0;
      } else if (inc_opt) {
          if (optarg != NULL) {
            add = strtol (optarg, 0, 0);
          } else {
            add = inc_opt;
          }
          inc_opt = 0;
      } else if (help_opt) {
        printf ("usage: %s [-t=VAL] [-mul=VAL] [-inc=VAL]\n", argv[0]);
        return 0;
      }
    }

  int n = atoi (argv[argc - 1]);



  a = malloc (n * sizeof (mp_limb_t));
  b = malloc (n * sizeof (mp_limb_t));
  d = malloc (2 * n * sizeof (mp_limb_t));

  mpn_random (a, n);
  mpn_random (b, n);

  gmp_set_num_threads(nb_thread);
  mpn_mul_n (d, a, b, n / 10);

  if (ratio == 0 && add == 0)
  {
    gmp_set_num_threads(nb_thread);
    t = seconds ();
    wct =	wct_seconds ();
    mpn_mul_n (d, a, b, n);
    t = seconds () - t;
    wct =	wct_seconds () - wct;
    printf ("mpn_mul_n took %.3fs ( wct : %.3fs ) with %d thread(s)\n", t, wct, nb_thread);
  }
  else
  {
    double wct1;
    clock_t t1;
    int nb_t = 1;
    while (nb_t <= nb_thread)
    {
      gmp_set_num_threads(nb_t);
      t = seconds ();
      wct =	wct_seconds ();
      mpn_mul_n (d, a, b, n);
      t = seconds () - t;
      wct =	wct_seconds () - wct;
      if (nb_t == 1)
      {
        wct1 = wct;
        t1 = t;
      }
      printf ("clock %.3fs (%.3f), wct %.3fs (%.3f) with %d thread(s)\n", t, t/t1, wct, wct1/wct, nb_t);
      if (add == 0 || (nb_t + add > nb_t * ratio && ratio != 0))
      {
        nb_t *= ratio;
      }
      else
      {
        nb_t += add;
      }
    }
  }

  free (a);
  free (b);
  free (d);

  return 0;
}
