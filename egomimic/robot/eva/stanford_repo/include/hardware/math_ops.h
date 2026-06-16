#ifndef __MATH_OPS_H
#define __MATH_OPS_H

#include <math.h>

float fmaxf3(float x, float y, float z);
float fminf3(float x, float y, float z);
void limit_norm(float *x, float *y, float limit);
int float_to_uint(float x, float x_min, float x_max, int bits);
float uint_to_float(int x_int, float x_min, float x_max, int bits);
float int_to_float(int val, float max_, float min_, int bits);

void float32_to_float16(float *float32, unsigned short int *float16);
void float16_to_float32(unsigned short int *float16, float *float32);

#endif