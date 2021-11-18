// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Tim Bernhard (ETHZ)
   [ based on angle_fourier_simple_omp.cpp Axel Kohlmeyer (Temple U)]
------------------------------------------------------------------------- */

#include "angle_fourier_simple_intel.h"
#include <cmath>
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "neighbor.h"
#include <immintrin.h>
#include "math_const.h"

using namespace LAMMPS_NS;

#define SMALL 0.001


typedef struct { double x,y,z; } dbl3_t;
typedef struct { int a,b,c,t;  } int4_t;

/* ---------------------------------------------------------------------- */

AngleFourierSimpleIntel::AngleFourierSimpleIntel(class LAMMPS *lmp)
  : AngleFourierSimple(lmp)
{
  
}

/* ---------------------------------------------------------------------- */

void AngleFourierSimpleIntel::compute(int eflag, int vflag)
{
  ev_init(eflag,vflag);

  if (evflag) {
    if (eflag) {
      if (force->newton_bond) eval<1,1,1>();
      else eval<1,1,0>();
    } else {
      if (force->newton_bond) eval<1,0,1>();
      else eval<1,0,0>();
    }
  } else {
    if (force->newton_bond) eval<0,0,1>();
    else eval<0,0,0>();
  }
}

template <int EVFLAG, int EFLAG, int NEWTON_BOND>
void AngleFourierSimpleIntel::eval()
{
  //int i1,i2,i3,n,type;
  double delx1,dely1,delz1,delx2,dely2,delz2;
  double eangle;
  double term,sgn;
  double rsq1,rsq2,r1,r2,c,cn,a,a11,a12,a22;
  double theta,ntheta;
  int nanglelist = neighbor->nanglelist;

  __m256d ones = _mm256_set1_pd(1.0);
  __m256d minusOnes = _mm256_set1_pd(-1.0);

  const dbl3_t * _noalias const x = (dbl3_t *) atom->x[0];
  dbl3_t * _noalias const f = (dbl3_t *) atom->f[0];
  const int4_t * _noalias const anglelist = (int4_t *) neighbor->anglelist[0];
  const int nlocal = atom->nlocal;
  eangle = 0.0;

  const int nanglelistMod4 = nanglelist % 4;
  const int nanglelistMinMod4 = nanglelist - nanglelistMod4;
  for (int n = 0; n < nanglelistMinMod4; n++) {
    // TODO: memory access patterns are shit like this
    int i1s[4] = {anglelist[n].a, anglelist[n+1].a, anglelist[n+2].a, anglelist[n+3].a};
    int i2s[4] = {anglelist[n].b, anglelist[n+1].b, anglelist[n+2].b, anglelist[n+3].b};
    int i3s[4] = {anglelist[n].c, anglelist[n+1].c, anglelist[n+2].c, anglelist[n+3].c};
    int types[4] = {anglelist[n].t, anglelist[n+1].t, anglelist[n+2].t, anglelist[n+3].t};
    double Ns[4] = {(double)N[types[0]], (double)N[types[1]], (double)N[types[2]], (double)N[types[3]]};
    __m256d m_Ns = _mm256_loadu_pd(Ns);
    double Cs[4] = {(double)C[types[0]], (double)C[types[1]], (double)C[types[2]], (double)C[types[3]]};
    __m256d m_Cs = _mm256_loadu_pd(Cs);
    double ks[4] = {(double)k[types[0]], (double)k[types[1]], (double)k[types[2]], (double)k[types[3]]};
    __m256d m_ks = _mm256_loadu_pd(ks);

    double x1s[4] = {x[i1s[0]].x, x[i1s[1]].x, x[i1s[2]].x, x[i1s[3]].x};
    __m256d m_x1s = _mm256_loadu_pd(x1s);
    double x2s[4] = {x[i2s[0]].x, x[i2s[1]].x, x[i2s[2]].x, x[i2s[3]].x};
    __m256d m_x2s = _mm256_loadu_pd(x2s);
    double x3s[4] = {x[i3s[0]].x, x[i3s[1]].x, x[i3s[2]].x, x[i3s[3]].x};
    __m256d m_x3s = _mm256_loadu_pd(x3s);

    double y1s[4] = {x[i1s[0]].y, x[i1s[1]].y, x[i1s[2]].y, x[i1s[3]].y};
    __m256d m_y1s = _mm256_loadu_pd(y1s);
    double y2s[4] = {x[i2s[0]].y, x[i2s[1]].y, x[i2s[2]].y, x[i2s[3]].y};
    __m256d m_y2s = _mm256_loadu_pd(y2s);
    double y3s[4] = {x[i3s[0]].y, x[i3s[1]].y, x[i3s[2]].y, x[i3s[3]].y};
    __m256d m_y3s = _mm256_loadu_pd(y3s);

    double z1s[4] = {x[i1s[0]].z, x[i1s[1]].z, x[i1s[2]].z, x[i1s[3]].z};
    __m256d m_z1s = _mm256_loadu_pd(z1s);
    double z2s[4] = {x[i2s[0]].z, x[i2s[1]].z, x[i2s[2]].z, x[i2s[3]].z};
    __m256d m_z2s = _mm256_loadu_pd(z2s);
    double z3s[4] = {x[i3s[0]].z, x[i3s[1]].z, x[i3s[2]].z, x[i3s[3]].z};
    __m256d m_z3s = _mm256_loadu_pd(z3s);

    // const int i1 = anglelist[n].a;
    // const int i2 = anglelist[n].b;
    // const int i3 = anglelist[n].c;
    // const int type = anglelist[n].t;

    // 1st bond
    // delx1 = x[i1].x - x[i2].x;
    __m256d m_delx1 = _mm256_sub_pd(m_x1s, m_x2s);
    // dely1 = x[i1].y - x[i2].y;
    __m256d m_dely1 = _mm256_sub_pd(m_y1s, m_y2s);
    // delz1 = x[i1].z - x[i2].z;
    __m256d m_delz1 = _mm256_sub_pd(m_z1s, m_z2s);

    // rsq1 = delx1*delx1 + dely1*dely1 + delz1*delz1;
    // r1 = sqrt(rsq1);
    __m256d m_rsq1_c1 = _mm256_mul_pd(m_delx1, m_delx1);
    __m256d m_rsq1_c2 = _mm256_fmadd_pd(m_dely1, m_dely1, m_rsq1_c1);
    __m256d m_rsq1 = _mm256_fmadd_pd(m_delz1, m_delz1, m_rsq1_c2);
    __m256d m_r1 = _mm256_sqrt_pd(m_rsq1);

    // 2nd bond

    // delx2 = x[i3].x - x[i2].x;
    // dely2 = x[i3].y - x[i2].y;
    // delz2 = x[i3].z - x[i2].z;
    __m256d m_delx2 = _mm256_sub_pd(m_x3s, m_x2s);
    __m256d m_dely2 = _mm256_sub_pd(m_y3s, m_y2s);
    __m256d m_delz2 = _mm256_sub_pd(m_z3s, m_z2s);

    // rsq2 = delx2*delx2 + dely2*dely2 + delz2*delz2;
    // r2 = sqrt(rsq2);
    __m256d m_rsq2_c1 = _mm256_mul_pd(m_delx2, m_delx2);
    __m256d m_rsq2_c2 = _mm256_fmadd_pd(m_dely2, m_dely2, m_rsq2_c1);
    __m256d m_rsq2 = _mm256_fmadd_pd(m_delz2, m_delz2, m_rsq2_c2);
    __m256d m_r2 = _mm256_sqrt_pd(m_rsq2); // sqrt has terrible latency & throuput :(

    // angle (cos and sin)
    // c = delx1*delx2 + dely1*dely2 + delz1*delz2;
    __m256d m_c_init_c1 = _mm256_mul_pd(m_delx1, m_delx2);
    __m256d m_c_init_c2 = _mm256_fmadd_pd(m_dely1, m_dely2, m_c_init_c1);
    __m256d m_c_init = _mm256_fmadd_pd(m_delz1, m_delz2, m_c_init_c2);

    // c /= r1*r2;
    __m256d m_r1r2 = _mm256_mul_pd(m_r1, m_r2);
    __m256d m_c = _mm256_div_pd(m_c_init, m_r1r2);

    // if (c > 1.0) c = 1.0;
    // else if (c < -1.0) c = -1.0;
    m_c = _mm256_max_pd(minusOnes, m_c);
    m_c = _mm256_min_pd(m_c, ones);

    // force & energy
    // theta = fastAcos(c);
    __m256d m_thetas = _mm256_acos_pd(m_c);
    // ntheta = (double)N[type]*theta;
    __m256d m_Nthetas = _mm256_mul_pd(m_thetas, m_Ns);
    // cn = std::cos(ntheta);
    __m256d m_cns = _mm256_cos_pd(m_Nthetas);
    // term = k[type]*(1.0+C[type]*cn);
    __m256d m_1p_ccn = _mm256_fmadd_pd(m_Cs, m_cns, ones);

    if (EFLAG) eangle = term;

    // TODO: handle sin(n th)/sin(th) singulatiries

    // if (fabs(c)-1.0 > 0.0001) {
    //   a = k[type]*C[type]*N[type]*(std::sin(ntheta))/(std::sin(theta));
    __m256d m_sin_nthetas = _mm256_sin_pd(m_Nthetas);
    __m256d m_sin_thetas = _mm256_sin_pd(m_thetas);
    __m256d m_a_c1 = _mm256_mul_pd(m_ks, m_Cs);
    __m256d m_sin_ntheta_divs = _mm256_div_pd(m_sin_nthetas, m_sin_thetas);
    __m256d m_a_c2 = _mm256_mul_pd(m_a_c1, m_Ns);
    __m256d m_a = _mm256_mul_pd(m_a_c2, m_sin_ntheta_divs);
    // } else {
    //   if (c >= 0.0) {
    //     term = 1.0 - c;
    //     sgn = 1.0;
    //   } else {
    //     term = 1.0 + c;
    //     sgn = ( fmod((double)(N[type]),2.0) == 0.0 )?-1.0:1.0;
    //   }
    //   a = N[type]+N[type]*(1.0-N[type]*N[type])*term/3.0;
    //   a = k[type]*C[type]*N[type]*(sgn)*a;
    // }

    // a11 = a*c / rsq1;
    __m256d m_a11s_c1 = _mm256_mul_pd(m_a, m_c);
    __m256d m_a11s = _mm256_div_pd(m_a11s_c1, m_rsq1);
    // a12 = -a / (r1*r2);
    __m256d m_a12s_c1 = _mm256_mul_pd(m_a, minusOnes);
    __m256d m_a12s = _mm256_div_pd(m_a12s_c1, m_r1r2);
    // a22 = a*c / rsq2;
    __m256d m_a22s = _mm256_div_pd(m_a11s_c1, m_rsq2);

    // f1[0] = a11*delx1 + a12*delx2;
    __m256d m_f1_0_c1 = _mm256_mul_pd(m_a11s, m_delx1);
    __m256d m_f1_0 = _mm256_fmadd_pd(m_a12s, m_delx2, m_f1_0_c1);
    // f1[1] = a11*dely1 + a12*dely2;
    __m256d m_f1_1_c1 = _mm256_mul_pd(m_a11s, m_dely1);
    __m256d m_f1_1 = _mm256_fmadd_pd(m_a12s, m_dely2, m_f1_1_c1);
    // f1[2] = a11*delz1 + a12*delz2;
    __m256d m_f1_2_c1 = _mm256_mul_pd(m_a11s, m_delz1);
    __m256d m_f1_2 = _mm256_fmadd_pd(m_a12s, m_delz2, m_f1_2_c1);
    // f3[0] = a22*delx2 + a12*delx1;
    __m256d m_f3_0_c1 = _mm256_mul_pd(m_a22s, m_delx2);
    __m256d m_f3_0 = _mm256_fmadd_pd(m_a12s, m_delx1, m_f3_0_c1);
    // f3[1] = a22*dely2 + a12*dely1;
    __m256d m_f3_1_c1 = _mm256_mul_pd(m_a22s, m_dely2);
    __m256d m_f3_1 = _mm256_fmadd_pd(m_a12s, m_dely1, m_f3_1_c1);
    // f3[2] = a22*delz2 + a12*delz1;
    __m256d m_f3_2_c1 = _mm256_mul_pd(m_a22s, m_delx2);
    __m256d m_f3_2 = _mm256_fmadd_pd(m_a12s, m_delx1, m_f3_2_c1);

    double f1xs[4];
    _mm256_store_pd(f1xs, m_f1_0);
    double f1ys[4];
    _mm256_store_pd(f1ys, m_f1_1);
    double f1zs[4];
    _mm256_store_pd(f1zs, m_f1_2);
    
    double f3xs[4];
    _mm256_store_pd(f3xs, m_f3_0);
    double f3ys[4];
    _mm256_store_pd(f3ys, m_f3_1);
    double f3zs[4];
    _mm256_store_pd(f3zs, m_f3_2);


    // apply force to each of 3 atoms
    for (int i = 0; i < 4; ++i){
      int i1 = i1s[i];
      int i2 = i2s[i];
      int i3 = i3s[i];

      double f1[3] = {f1xs[i], f1ys[i], f1zs[i]};
      double f3[3] = {f3xs[i], f3ys[i], f3zs[i]};

      if (NEWTON_BOND || i1 < nlocal) {
        f[i1].x += f1xs[i];
        f[i1].y += f1ys[i];
        f[i1].z += f1zs[i];
      }

      if (NEWTON_BOND || i2 < nlocal) {
        f[i2].x -= f1xs[i] + f3xs[i];
        f[i2].y -= f1ys[i] + f3ys[i];
        f[i2].z -= f1zs[i] + f3zs[i];
      }

      if (NEWTON_BOND || i3 < nlocal) {
        f[i3].x += f3xs[i];
        f[i3].y += f3ys[i];
        f[i3].z += f3zs[i];
      }

      if (EVFLAG) ev_tally(i1,i2,i3,nlocal,NEWTON_BOND,eangle,f1,f3,
                              delx1,dely1,delz1,delx2,dely2,delz2);
    }
  }

  double f1[3],f3[3];

  for (int n = nanglelistMinMod4; n < nanglelist; n++) {
    const int i1 = anglelist[n].a;
    const int i2 = anglelist[n].b;
    const int i3 = anglelist[n].c;
    const int type = anglelist[n].t;

    // 1st bond

    delx1 = x[i1].x - x[i2].x;
    dely1 = x[i1].y - x[i2].y;
    delz1 = x[i1].z - x[i2].z;

    rsq1 = delx1*delx1 + dely1*dely1 + delz1*delz1;
    r1 = sqrt(rsq1);

    // 2nd bond

    delx2 = x[i3].x - x[i2].x;
    dely2 = x[i3].y - x[i2].y;
    delz2 = x[i3].z - x[i2].z;

    rsq2 = delx2*delx2 + dely2*dely2 + delz2*delz2;
    r2 = sqrt(rsq2);

    // angle (cos and sin)

    c = delx1*delx2 + dely1*dely2 + delz1*delz2;
    c /= r1*r2;

    if (c > 1.0) c = 1.0;
    else if (c < -1.0) c = -1.0;

    // force & energy

    theta = std::acos(c);
    ntheta = (double)N[type]*theta;
    cn = std::cos(ntheta);
    term = k[type]*(1.0+C[type]*cn);

    if (EFLAG) eangle = term;

    // handle sin(n th)/sin(th) singulatiries

    if (fabs(c)-1.0 > 0.0001) {
      a = k[type]*C[type]*N[type]*(std::sin(ntheta))/(std::sin(theta));
    } else {
      if (c >= 0.0) {
        term = 1.0 - c;
        sgn = 1.0;
      } else {
        term = 1.0 + c;
        sgn = ( fmod((double)(N[type]),2.0) == 0.0 )?-1.0:1.0;
      }
      a = N[type]+N[type]*(1.0-N[type]*N[type])*term/3.0;
      a = k[type]*C[type]*N[type]*(sgn)*a;
    }

    a11 = a*c / rsq1;
    a12 = -a / (r1*r2);
    a22 = a*c / rsq2;

    f1[0] = a11*delx1 + a12*delx2;
    f1[1] = a11*dely1 + a12*dely2;
    f1[2] = a11*delz1 + a12*delz2;
    f3[0] = a22*delx2 + a12*delx1;
    f3[1] = a22*dely2 + a12*dely1;
    f3[2] = a22*delz2 + a12*delz1;

    // apply force to each of 3 atoms

    if (NEWTON_BOND || i1 < nlocal) {
      f[i1].x += f1[0];
      f[i1].y += f1[1];
      f[i1].z += f1[2];
    }

    if (NEWTON_BOND || i2 < nlocal) {
      f[i2].x -= f1[0] + f3[0];
      f[i2].y -= f1[1] + f3[1];
      f[i2].z -= f1[2] + f3[2];
    }

    if (NEWTON_BOND || i3 < nlocal) {
      f[i3].x += f3[0];
      f[i3].y += f3[1];
      f[i3].z += f3[2];
    }

    if (EVFLAG) ev_tally(i1,i2,i3,nlocal,NEWTON_BOND,eangle,f1,f3,
                             delx1,dely1,delz1,delx2,dely2,delz2);
  }
}
