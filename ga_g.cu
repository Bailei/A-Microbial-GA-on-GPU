#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <unistd.h>
#include <stdint.h>

#define POP 300
#define LEN 30
#define MUT 0.1
#define REC 0.5
#define END 10000
#define SUMTAG 150
#define PRODTAG 3600

int gene[POP][LEN];
int value[POP][LEN];
int seed[POP][LEN];
void init_pop();
double evaluate(int n);
void run();
void display(int tournaments, int n);
void get_result(int idx);
double score[POP];

double random_double(){
  double d;
  d = (double)(rand() % 10) / 10;
  return d; 
}

__device__ unsigned int get_rand(int range, int* seed){
  *seed ^= (*seed << 13);
  *seed ^= (*seed >> 17);
  *seed ^= (*seed << 5);
  return *seed % range;
}
         
__global__ void comput_kernel(double* score, int* gene, int* value){
  int offset = blockIdx.x * LEN;
  //evaluate
  uint64_t prod = 1;
  uint32_t sum = 0;
  for(int i = 0; i < LEN; ++i)
    if(gene[offset + i] == 0)
      sum += value[offset + i];
    else
      prod *= value[offset + i];

  double scaled_sum_error = (double)(sum - (double)SUMTAG) / (double)SUMTAG;
  double scaled_prod_error = (double)(prod - (double)PRODTAG) / (double)PRODTAG;
  
  if(scaled_sum_error < 0.0) 
    scaled_sum_error *= -1;
  
  if(scaled_prod_error < 0.0)
    scaled_prod_error *= -1;
  
  score[blockIdx.x] = scaled_sum_error + scaled_prod_error;
  
}

__global__ void find_min_score(double *score, double* min_score, int *min_idx){
  int low_idx = -1;
  double low = 0;
  
  __shared__ double shared_score[POP];

  for(int i = threadIdx.x; i < POP; i += blockDim.x)
    shared_score[i] = score[i];
  
  __syncthreads();
  if(threadIdx.x != 0)
    return;

  for(int i = 0; i < POP; ++i)
    if(shared_score[i] < low || low_idx == -1){
      low = score[i];
      low_idx = i;
    }

  *min_idx = low_idx;
  *min_score = low;
}


__global__ void mutate_kernel(int* gene, int *min_idx, int* seed){
  if(blockIdx.x == *min_idx)
    return;

  int offset = blockIdx.x * LEN;
  int min_offset = *min_idx * LEN;
  int reg_seed = seed[blockIdx.x * LEN + threadIdx.x];

  if(get_rand(100, &reg_seed) < (REC / 100))
    gene[offset + threadIdx.x] = gene[min_offset + threadIdx.x];
  if(get_rand(100, &reg_seed) < (MUT / 100))
    gene[offset + threadIdx.x] = 1 - gene[offset + threadIdx.x];

  seed[blockIdx.x * LEN + threadIdx.x] = reg_seed;
}

void run(){
  init_pop();
  int low_idx = -1;
  double low = 0;   
  int tournamentNo;
  
  int* gene_d;
  int* value_d;
  double* score_d;
  double* min_score;
  int* min_idx;
  int* seed_d;

  cudaMalloc((void**)&gene_d, sizeof(int) * POP * LEN); 
  cudaMalloc((void**)&value_d, sizeof(int) * POP * LEN); 
  cudaMalloc((void**)&score_d, sizeof(double) * POP); 
  cudaMalloc((void**)&seed_d, sizeof(int) * POP * LEN); 
  cudaMalloc((void**)&min_score, sizeof(double)); 
  cudaMalloc((void**)&min_idx, sizeof(int)); 
  

  cudaMemcpy(gene_d, gene, sizeof(int) * POP * LEN, cudaMemcpyHostToDevice);
  cudaMemcpy(value_d, value, sizeof(int) * POP * LEN, cudaMemcpyHostToDevice);
  cudaMemcpy(score_d, score, sizeof(double) * POP, cudaMemcpyHostToDevice);
  cudaMemcpy(seed_d, seed, sizeof(int) * POP * LEN, cudaMemcpyHostToDevice);

  dim3 dimGrid(POP, 1);
  dim3 dimBlock(1, 1);

  for(tournamentNo = 0; tournamentNo < END; tournamentNo++){
    comput_kernel<<<dimGrid, dimBlock>>>(score_d, gene_d, value_d);
    find_min_score<<<1, 128>>>(score_d, min_score, min_idx); 
    mutate_kernel<<<dimGrid, LEN>>>(gene_d, min_idx, seed_d);
  }
  
  cudaMemcpy(gene, gene_d, sizeof(int) * POP * LEN, cudaMemcpyDeviceToHost);
  cudaMemcpy(score, score_d, sizeof(double) * POP, cudaMemcpyDeviceToHost);
  
  low_idx = -1;
  low = 0;
  for(int i =0; i < POP; ++i)
    if((low_idx == -1 || score[i] < low) && score[i] != -1){
      low = score[i];
      low_idx = i;
    }
  
  if(low_idx != -1){
    //printf("%f %f\n", low, evaluate(low_idx));
    get_result(low_idx);
    display(tournamentNo, low_idx);
  }
}

void get_result(int idx){
  unsigned long long prod, sum;
  prod = 1;
  sum = 0;

  for(int i = 0; i < LEN; ++i){
    if(gene[idx][i] == 1)
      prod *= value[idx][i];
    else
      sum += value[idx][i];
  }

  printf("sum :%llu  prod: %llu\n", sum, prod);
}

void display(int tournaments, int n){
  printf("=========================================================================\n");
  printf("After %d tournaments, Solution sum pile (should be %d) cards are : \n", tournaments, SUMTAG);
  for(int i = 0; i < LEN; i++){
      if(gene[n][i] == 0){
          printf("%d ", value[n][i]);
      } 
  }
  printf("\n");
  printf("Solution product pile (should be %d) cards are : \n", PRODTAG);
  for(int i = 0; i < LEN; i++){
      if(gene[n][i] == 1){
          printf("%d ", value[n][i]);
      } 
  }
  
  for(int i = 0; i < LEN; i++)
      assert(gene[n][i] == 1 || gene[n][i] == 0);
  printf("\n=========================================================================\n")
}


double evaluate(int n){
  unsigned long long sum = 0, prod = 1;
  double scaled_sum_error, scaled_prod_error, combined_error;
  for(int i = 0; i < LEN; i++){
      if(gene[n][i] == 0){
          sum += value[n][i];
      }
      else{
         prod *= value[n][i];
      }
  }

  scaled_sum_error = (double)(sum - (double)SUMTAG) / (double)SUMTAG;
  if(scaled_sum_error < 0.0) scaled_sum_error *= -1;
  scaled_prod_error = (double)(prod - (double)PRODTAG) / (double)PRODTAG;
  if(scaled_prod_error < 0.0) scaled_prod_error *= -1;
  combined_error = scaled_sum_error + scaled_prod_error;
  return combined_error;
}


void init_pop(){
  for(int i = 0; i < POP; i++){
      for(int j = 0; j < LEN; j++){
          if(random_double() < 0.5){
              gene[i][j] = 0;
          }
          else{
              gene[i][j] = 1;
          }
      }
    score[i] = -1;
  }
  
  for(int i = 0; i < POP; i++){
      for(int j = 0; j < LEN; j++){
          value[i][j] = rand() % 9 + 1;
      }
  }

  for(int i = 0; i < POP; ++i)
    for(int j = 0; j < LEN; ++j)
    seed[i][j] = rand() % 10000000;
}

int main(){
  srand(getpid());  
  run();
  return 0;
}

