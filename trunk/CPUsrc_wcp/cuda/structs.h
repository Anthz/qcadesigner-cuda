#ifndef _STRUCTS_H_
#define _STRUCTS_H_

// MODIFIED coherence_OP struct (from coherence_vector.h)
typedef struct
{
   float T;
   float relaxation;
   float time_step;
   float duration;
   float clock_high;
   float clock_low;
   float clock_shift;
   float clock_amplitude_factor;
   float radius_of_effect;
   float epsilonR;
   float layer_separation;
   int algorithm;
} CUDA_coherence_OP;

// MODIFIED coherence_optimizations struct (from coherence_vector.c)
typedef struct
{
   float clock_prefactor;
   float clock_shift;
   float four_pi_over_number_samples;
   float two_pi_over_number_samples;
   float hbar_over_kBT;
} CUDA_coherence_optimizations;

#endif /* _STRUCTS_H_ */