#include <iostream>
#include <immintrin.h>
/* #include "ml_module.h" */
#include "common.h"

#define abs(x) (x >= 0 ? x : -x)
#define MAX(x, y) (x > y? x : y)

void matmul(REAL *c, REAL *a, REAL *b, int M, int N, int K) {

  int i, j, k;
  for(i = 0; i < M; i++) {
    for(k = 0; k < K; k++) {
      for(j = 0; j < N; j++) {
        c[i * M + j] += a[i * K + k] * b[k * N + j];
      }
    }
  }
}

/* void mlp_layer(REAL * __restrict__ c, const REAL * __restrict__ a, const REAL * __restrict__ B_T, const REAL * __restrict__ bias) { */
/*   // a -> input */
/*   // B_T -> weights transposed */
/*   const int M = 1; */
/*   const int N = 16; */
/*   const int K = NR_FEAT; */
/*   __m512 acc[16]; */
/*   __m512 res; */
/*   __m512 Avec; */
/*   for(int i = 0; i < 16; ++i) { */
/*     acc[i] = _mm512_setzero_ps(); */
/*   } */
/*   int i, j, k; */
/*   for(k = 0; k < K / 16; k++) { */
/*     Avec = _mm512_loadu_ps(&a[k * 16]); */
/*     for(i = 0; i < 16; i++) { */
/*       // Changed N in B_T to K for it to work */
/*       acc[i] = _mm512_fmadd_ps(acc[i], Avec, _mm512_loadu_ps(&B_T[k * 16 + i * K])); */
/*     } */
/*   } */
/*   // Perform reduction and add in bias */
/*   for(j = 0; j < N; j++) { */
/*     c[j] = _mm512_reduce_add_ps(acc[j]) + bias[j]; */
/*   } */
/*   /\* _mm512_storeu_ps(&c, res); *\/ */
/* } */
void mlp_layer(REAL * __restrict__ c, const REAL * __restrict__ a, const REAL * __restrict__ B_T, const REAL * __restrict__ bias) {
  // a -> input
  // B_T -> weights transposed
  const int M = 1;
  const int N = 16;
  const int K = NR_FEAT;
  __m512 acc[16];
  __m512 res;
  __m512 Avec;
  /* __m512 Avec_0; */
  /* __m512 Avec_1; */
  /* __m512 Avec_2; */
  /* __m512 Avec_3; */
  __m512 one = _mm512_set1_ps(1.0f);
  for(int i = 0; i < 16; ++i) {
    acc[i] = _mm512_set1_ps(0.0f);
  }
  int i, j, k;
  for(k = 0; k < K / 16; k++) {
    /* Avec = _mm512_loadu_ps(&a[k * 16]); */
    Avec = _mm512_loadu_ps(&a[k * 16]);
    for(i = 0; i < 16; i+=4) {
      // Changed N in B_T to K for it to work
      /* acc[i] = _mm512_fmadd_ps(Avec, _mm512_loadu_ps(&B_T[k * 16 + i * K]), acc[i]); */
      acc[i] = _mm512_fmadd_ps(Avec, _mm512_loadu_ps(&B_T[k * 16 + i * K]), acc[i]);
      acc[i + 1] = _mm512_fmadd_ps(Avec, _mm512_loadu_ps(&B_T[(k) * 16 + (i+1) * K]), acc[i + 1]);
      acc[i + 2] = _mm512_fmadd_ps(Avec, _mm512_loadu_ps(&B_T[(k) * 16 + (i+2) * K]), acc[i + 2]);
      acc[i + 3] = _mm512_fmadd_ps(Avec, _mm512_loadu_ps(&B_T[(k) * 16 + (i+3) * K]), acc[i + 3]);
    }
  }
  // Perform reduction and add in bias
  for(j = 0; j < N; j++) {
  /* for(j = 0; j < N; j++) { */
    c[j] = _mm512_reduce_add_ps(acc[j]) + bias[j];
    /* res = _mm512_dp_ps(acc[j], one, 0b1111111111111111); */
    /* c[j] = res[0] + res[8]; */
    /* c[j + 1] = _mm512_reduce_add_ps(acc[j + 1]) + bias[j + 1]; */
    /* c[j + 2] = _mm512_reduce_add_ps(acc[j + 2]) + bias[j + 2]; */
    /* c[j + 3] = _mm512_reduce_add_ps(acc[j + 3]) + bias[j + 3]; */
  }
  /* _mm512_storeu_ps(&c, res); */
}

static inline float _mm256_reduce_add_ps(__m256 x) {
    /* ( x3+x7, x2+x6, x1+x5, x0+x4 ) */
    const __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    /* ( -, -, x1+x3+x5+x7, x0+x2+x4+x6 ) */
    const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    /* ( -, -, -, x0+x1+x2+x3+x4+x5+x6+x7 ) */
    const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    /* Conversion to float is a no-op on x86-64 */
    return _mm_cvtss_f32(x32);
}

void mlp_layer_avx2_v2(REAL * __restrict__ c, const REAL * __restrict__ a, const REAL * __restrict__ b, const REAL * __restrict__ bias) {
  // a -> input
  // B_T -> weights transposed
  const int M = 1;
  const int N = 16;
  const int K = 64;
  const int lda = M;
  const int ldb = N;
  const int ldc = N;

  /* _mm256_storeu_ps((__m256*)c, ()) */

  /* __m512 Avec_0; */
  /* __m512 Avec_1; */
  /* __m512 Avec_2; */
  /* __m512 Avec_3; */

  int i, j, k, l;
  /* __m256 acc[8]; */
  /* __m256 res; */
  /* __m256 Avec; */
  /* __m256 one = _mm256_set1_ps(1.0f); */
  for (i = 0; i < M; i++) {
    for (j = 0; j < N; j += 8) {
      auto sumA = _mm256_setzero_ps();
      auto sumB = _mm256_setzero_ps();
      for (k = 0; k < K; k++) {
        auto bc_a = _mm256_set1_ps(a[i * lda + k]);
        auto vecA_b = _mm256_loadu_ps(&b[k * ldb + j]);
        auto vecB_b = _mm256_loadu_ps(&b[k * ldb + j + 8]);
        /* auto prodA = _mm256_mullo_epi32(bc_a, vecA_b); */
        /* auto prodB = _mm256_mullo_epi32(bc_a, vecB_b); */
        /* sumA = _mm256_add_epi32(sumA, prodA); */
        /* sumB = _mm256_add_epi32(sumB, prodB); */
        sumA = _mm256_fmadd_ps(bc_a, vecA_b, sumA);
        sumB = _mm256_fmadd_ps(bc_a, vecB_b, sumB);
      }
      _mm256_storeu_ps(&c[i * ldc + j], sumA);
      _mm256_storeu_ps(&c[i * ldc + j + 8], sumB);
    }
  }
}

void mlp_layer_avx2_v3(REAL * __restrict__ c, const REAL * __restrict__ a, const REAL * __restrict__ b, const REAL * __restrict__ bias) {
  // a -> input
  // B_T -> weights transposed
  int n = 0;
  const int M = 1;
  const int N = 16;
  const int K = 64;
  const int block_width = 8;
  const int block_height = 8;
  for (int row_offset = 0; row_offset < K; row_offset += block_height) {
    for (int column_offset = 0; column_offset < N; column_offset += block_width) {
      for (int i = 0; i < M; ++i) {
        for (int j = column_offset; j < column_offset + block_width && j < N; j += 8) {
          __m256 sum = _mm256_load_ps(&c[i * N + j]);
          for (int k = row_offset; k < row_offset + block_height && k < K; ++k) {
            sum = _mm256_fmadd_ps(_mm256_set1_ps(a[i * M + k]), _mm256_load_ps(&b[k * N + j]), sum);
          }
          _mm256_store_ps(&c[i * N + j], sum);
        }
      }
    }
  }
}

void mlp_layer_avx2_v4(REAL * __restrict__ c, const REAL * __restrict__ a, const REAL * __restrict__ b, const REAL * __restrict__ bias) {
  // a -> input
  // B_T -> weights transposed
  const int M = 1;
  const int N = 16;
  const int K = 64;
  const int block_width = 8;
  const int block_height = 8;
  for (int column_offset = 0; column_offset < N; column_offset += block_width) {
    for (int row_offset = 0; row_offset < K; row_offset += block_height) {
      for (int i = 0; i < M; ++i) {
        for (int j = column_offset; j < column_offset + block_width && j < N; j += 16) {
          __m256 sum1 = _mm256_load_ps(c + i * N + j);
          __m256 sum2 = _mm256_load_ps(c + i * N + j + 8);
          for (int k = row_offset; k < row_offset + block_height && k < K; ++k) {
            __m256 a_broadcast = _mm256_set1_ps(a[i * M + k]);
            sum1 = _mm256_fmadd_ps(a_broadcast, _mm256_load_ps(b + k * N + j), sum1);
            /* sum2 = _mm256_fmadd_ps(a_broadcast, _mm256_load_ps(b + k * M + j + 8), sum2); */
          }
          _mm256_store_ps(c + i * N + j, sum1);
          _mm256_store_ps(c + i * N + j + 8, sum2);
        }
      }
    }
  }
}

void mlp_layer_avx2(REAL * __restrict__ c, const REAL * __restrict__ a, const REAL * __restrict__ B_T, const REAL * __restrict__ bias) {
  // a -> input
  // B_T -> weights transposed
  const int M = 1;
  const int N = 16;
  const int K = 64;

  /* _mm256_storeu_ps((__m256*)c, ()) */

  /* __m512 Avec_0; */
  /* __m512 Avec_1; */
  /* __m512 Avec_2; */
  /* __m512 Avec_3; */

  int i, j, k, l;
  __m256 acc[8];
  __m256 res;
  __m256 Avec;
  __m256 one = _mm256_set1_ps(1.0f);
  for(l = 0; l < 2; ++l) {
    for(i = 0; i < 8; ++i) {
      acc[i] = _mm256_setzero_ps();
    }
    for(k = 0; k < K / 8; k++) {
      /* Avec = _mm512_loadu_ps(&a[k * 16]); */
      Avec = _mm256_loadu_ps(&a[k * 8]);
      for(i = 0; i < 8; i++) {
        // Changed N in B_T to K for it to work
        /* acc[i] = _mm512_fmadd_ps(Avec, _mm512_loadu_ps(&B_T[k * 16 + i * K]), acc[i]); */
        /* acc[i] = _mm256_fmadd_ps(Avec, _mm256_loadu_ps(&B_T[k * 8 + i * K]), acc[i]); */
        acc[i] = _mm256_fmadd_ps(Avec, _mm256_loadu_ps(&B_T[k * 8 + i * K]), acc[i]);
        /* acc[i + 1] = _mm256_fmadd_ps(Avec, _mm256_loadu_ps(&B_T[(k) * 8 + (i+1) * K]), acc[i + 1]); */
        /* acc[i + 2] = _mm256_fmadd_ps(Avec, _mm256_loadu_ps(&B_T[(k) * 8 + (i+2) * K]), acc[i + 2]); */
        /* acc[i + 3] = _mm256_fmadd_ps(Avec, _mm256_loadu_ps(&B_T[(k) * 8 + (i+3) * K]), acc[i + 3]); */
      }
    }
    // Perform reduction and add in bias
    for(j = 0; j < N / 2; j++) {
      /* for(j = 0; j < N; j++) { */
      /* c[j] = _mm256_reduce_add_ps(acc[j]) + bias[j]; */
      res = _mm256_dp_ps(acc[j], one, 0b11111111);
      c[j] = res[0] + res[4] + bias[j];
      /* c[j + 1] = _mm256_reduce_add_ps(acc[j + 1]) + bias[j + 1]; */
      /* c[j + 2] = _mm256_reduce_add_ps(acc[j + 2]) + bias[j + 2]; */
      /* c[j + 3] = _mm256_reduce_add_ps(acc[j + 3]) + bias[j + 3]; */
    }
    B_T += N * K / 2;
    c += 8;
    bias += 8;
  }
  /* _mm512_storeu_ps(&c, res); */
}

void mlp_layer_v2(REAL * __restrict__ c, const REAL * __restrict__ a, const REAL * __restrict__ b, const REAL * __restrict__ bias) {
__m512 vec_multi_res = _mm512_setzero_ps(); //Initialize vector to zero
__m512 vec_a = _mm512_setzero_ps(); //Initialize vector to zero
__m512 vec_b = _mm512_setzero_ps(); //Initialize vector to zero
 int M = 1;
 int N = 16;
 int K = NR_FEAT;

 int i, j, k;
 for (i = 0; i < M; i++)
   {
     for (j = 0; j < N; ++j)
       {
         //Stores one element in a and use it in all computations needed before proceeding
         //Stores as vector to increase computations per cycle
         /* vec_a = _mm256_set1_epi32(a[i][j]); */
         vec_a = _mm512_set1_ps(a[i * K + j]);

         for (k = 0; k < K; k += 16)
           {
             vec_b = _mm512_loadu_ps((__m256i*)&b[j * N + k]); //Stores row of second matrix (eight in each iteration)
             vec_multi_res = _mm512_loadu_ps((__m256i*)&c[i * N + k]); //Loads the c matrix row as a vector
             /* vec_multi_res = _mm256_add_epi32(vec_multi_res ,_mm256_mullo_epi32(vec_a, vec_b));//Multiplies the vectors and adds to th the c vector */
             vec_multi_res = _mm512_fmadd_ps(vec_a, vec_b, vec_multi_res);

             _mm512_storeu_ps((__m512*)&c[i * N + k], vec_multi_res); //Stores the c vector into the c array
           }
       }
   }
  for(j = 0; j < N; j++) {
    c[j] = c[j] + bias[j];
  }
}

/* void matadd_layer(REAL * __restrict__ c, const REAL * __restrict__ a, const REAL * __restrict__ b) { */
/*   __m512 avec = _mm512_loadu_ps(a); */
/*   __m512 bvec = _mm512_loadu_ps(b); */
/*   __m512 res = _mm512_add_ps(avec, bvec); */
/*   _mm512_storeu_ps(&c, res); */
/* } */


void mlp_layer2(REAL * __restrict__ c, const REAL * __restrict__ a, const REAL * __restrict__ B_T, const REAL * __restrict__ bias) {
  // a -> input
  // B_T -> weights transposed
  const int M = 1;
  const int N = 1;
  const int K = 16;
  __m512 res;
  res = _mm512_mul_ps(_mm512_loadu_ps(a), _mm512_loadu_ps(B_T));
  c[0] = _mm512_reduce_add_ps(res) + bias[0];
}

void mlp_layer2_avx2(REAL * __restrict__ c, const REAL * __restrict__ a, const REAL * __restrict__ B_T, const REAL * __restrict__ bias) {
  // a -> input
  // B_T -> weights transposed
  const int M = 1;
  const int N = 1;
  const int K = 16;
  __m256 res, res2;
  res = _mm256_mul_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(B_T));
  /* c[0] = _mm512_reduce_add_ps(res) + bias[0]; */
  res2 = _mm256_mul_ps(_mm256_loadu_ps(a + 8), _mm256_loadu_ps(B_T + 8));
  c[0] = _mm256_reduce_add_ps(res) + _mm256_reduce_add_ps(res2) + bias[0];
}

/* __attribute__((target("sse2"))) int matadd(matrix *C, matrix *A, matrix *B) */
int matadd(REAL *C, REAL *A, REAL *B, int nrows, int ncols)
{
  int i;
  for (i = 0; i < nrows * ncols; i++) {
    C[i] = A[i] + B[i];
  }
  return 0;
}

/* __attribute__((target("sse2"))) void ReLU(matrix *C) */
void ReLU_avx(REAL *c)
{
  __m512 temp_c = _mm512_loadu_ps(c);
  __m512 res = _mm512_max_ps(_mm512_setzero_ps(), temp_c);
  _mm512_storeu_ps(c, res);
}

void ReLU_avx2(REAL *c)
{
  __m256 temp_c = _mm256_loadu_ps(c);
  __m256 res = _mm256_max_ps(_mm256_setzero_ps(), temp_c);
  temp_c = _mm256_loadu_ps(c + 8);
  __m256 res2 = _mm256_max_ps(_mm256_setzero_ps(), temp_c);
  _mm256_storeu_ps(c, res);
  _mm256_storeu_ps(c + 8, res2);
}

void ReLU(REAL *c, int nrows, int ncols)
{
  /* __m512 res = _mm512_loadu_ps(c); */
  /* res = _mm512_max_ps(res, _mm512_setzero_ps()); */
  /* _mm512_storeu_ps(&c, res); */
  int i;
  for (i = 0; i < nrows * ncols; i++) {
    c[i] = MAX(0, c[i]);
  }
}

static inline REAL Fabs(REAL f) {
    /* optimizer will optimize away the `if` statement and the library call */
  if (sizeof(REAL) == sizeof(uint32_t)) {
    union {
            REAL f;
                  uint32_t i;
                      
    } u;
        u.f = f;
            u.i &= 0x7fffffff;
                return u.f;
                  
  }
    return (f);
    
}


int forward_pass_avx(REAL *input){
  REAL output;

  REAL out1_d[HIDDEN_SIZE] = {0};
  REAL out2_d[NR_OUT] = {0};
  /* for(int i=0; i < NR_FEAT * HIDDEN_SIZE; i++) { */
  /*   W1_d[i] = 1; */
  /* } */
  /* for(int i = 0; i < HIDDEN_SIZE; i++) */
  /*   out1_d[i] = 0; */

  /* matmul(out1_d, input, W1_d, 1, HIDDEN_SIZE, NR_FEAT); */
    /* std::cout << out1_d << ", "; */
  /* mlp_layer(out1_d, input, W1_d_T, b1_d); */
  /* mlp_layer_avx2(out1_d, input, W1_d, b1_d); */
  mlp_layer_avx2_v3(out1_d, input, W1_d, b1_d);
    /* std::cout << out1_d << "\n"; */
  /* mlp_layer_v2(out1_d, input, W1_d_T, b1_d); */
  std::cout << "AVX results: " << std::endl;
  for(int i = 0; i < 16; i++) {
    std::cout << out1_d[i] << ", ";
  }
  std::cout << std::endl;
  ReLU_avx2(out1_d);

  /* matmul(out2_d, out1_d, W2_d, 1, NR_OUT, HIDDEN_SIZE); */
  /* mlp_layer2(out2_d, out1_d, W2_d, b2_d); */
  mlp_layer2_avx2(out2_d, out1_d, W2_d, b2_d);
  /* matadd(out2_d, out2_d, b2_d, 1, NR_OUT); */
  output = out2_d[0];
  /* std::cout << output << "\n"; */

  /* output = 0.5 * (output / (1 + abs(output))); */
  /* output = output / (1 + abs(output)); */
  output = 0.5 * (output / (1 + Fabs(output)) + 1);
 
  /* std::cout << "Out (avx): " << output << std::endl; */

  return output >= 0.5 ? 1 : 0;
}

int forward_pass(REAL *input){
  REAL output;

  REAL out1_d[HIDDEN_SIZE] = {0};
  REAL out2_d[NR_OUT] = {0};
  /* REAL test[] = {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,10}; */
  /* REAL test_ = -1.0f; */
  /* std::cout << Fabs(test_) << std::endl; */
  /* return 0; */
  /* ReLU_avx(test, 1, 1); */
  /* for(int i = 0; i < 16;i++) */
  /*   std::cout << test[i] << ", "; */
  /* std::cout << std::endl; */
  /* for(int i=0; i < NR_FEAT * HIDDEN_SIZE; i++) { */
  /*   W1_d[i] = 1; */
  /* } */
  /* for(int i = 0; i < HIDDEN_SIZE; i++) */
  /*   out1_d[i] = 0; */

  matmul(out1_d, input, W1_d, 1, HIDDEN_SIZE, NR_FEAT);
  /* matadd(out1_d, out1_d, b1_d, 1, HIDDEN_SIZE); */
  std::cout << "Reference results: " << std::endl;
  for(int i = 0; i < 16; i++) {
    std::cout << out1_d[i] << ", ";
  }
  std::cout << std::endl;
  /* mlp_layer(out1_d, input, W1_d); */
  ReLU(out1_d, 1, HIDDEN_SIZE);

  matmul(out2_d, out1_d, W2_d, 1, NR_OUT, HIDDEN_SIZE);
  /* mlp_layer2(out2_d, out1_d, W2_d); */
  matadd(out2_d, out2_d, b2_d, 1, NR_OUT);
  output = out2_d[0];
  /* std::cout << output << "\n"; */

  /* output = 0.5 * (output / (1 + abs(output))); */
  output = output / (1 + abs(output));
  /* output = 0.5 * (output / (1 + Fabs(output)) + 1); */

  /* std::cout << "Out: " << output << std::endl; */

  return output >= 0.5 ? 1 : 0;
}

