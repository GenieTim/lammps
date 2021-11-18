/* -*- c++ -*- ----------------------------------------------------------
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

#ifdef ANGLE_CLASS
// clang-format off
AngleStyle(fourier/simple/intel,AngleFourierSimpleIntel);
// clang-format on
#else

#ifndef LMP_ANGLE_FOURIER_SIMPLE_AVX2_H
#define LMP_ANGLE_FOURIER_SIMPLE_AVX2_H

#include "angle_fourier_simple.h"
#include "math_const.h"

namespace LAMMPS_NS {

/* ---------------------------------------------------------------------- */

// #define ONE_OVER_PI 0.318309886183790671538035357
// #define ONE_OVER_PI_SQUARE 0.1013211836423377714440499
// #define ONE_OVER_PI_CUBED 0.03225153443319948918450346
// #define TWO_PI_INVERSE 0.1591549430918953357690176

/* ---------------------------------------------------------------------- */

/**
 * Approximation of acos 
 * Returns the arccosine of x in the range [0,pi], expecting x to be in the range [-1,+1]. 
 * Absolute error <= 6.7e-5
 * Range mapping not necessary as C++ standard library would throw domain error
 * Source: https://developer.download.nvidia.com/cg/acos.html
 */
// static __m256d fastAcos(__m256d x)
// {
//   double negate = double(x < 0);
//   x = _mm256_abs_pd(x);
//   // fmas are used here only if I use -ffast-math
//   // tested compiler: clang version 12.0.0, Target: x86_64-apple-darwin20.5.0
//   double ret = -0.0187293 * x + 0.0742610;
//   double ret2 = ret * x - 0.2121144;
//   double ret3 = ret2 * x + 1.5707288;
//   double ret4 = ret3 * sqrt(1.0 - x);
//   double ret5 = ret4 * (1 - 2 * negate);
//   return negate * 3.14159265358979 + ret5;
// }

// static double fastCos(double);
// static double fastAcos(double);

class AngleFourierSimpleIntel : public AngleFourierSimple {

 public:
  AngleFourierSimpleIntel(class LAMMPS *lmp);
  virtual void compute(int, int);

 private:
  template <int EVFLAG, int EFLAG, int NEWTON_BOND> void eval();
};

}    // namespace LAMMPS_NS

#endif
#endif
