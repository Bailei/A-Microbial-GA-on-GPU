#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <unistd.h>

#define POP 300
#define LEN 30
#define MUT 0.1
#define REC 0.5
#define END 10000
#define SUMTAG 150
#define PRODTAG 3600

int gene[POP][LEN];
int value[POP][LEN];

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

void run(){
    init_pop();
    int low_idx = -1;
    double low = 0;   
    int tournamentNo;
    
    for(tournamentNo = 0; tournamentNo < END; tournamentNo++){
        for(int i = 0; i < POP; ++i){
          score[i] = evaluate(i);
          if(low_idx == -1 || low > score[i]){
            low = score[i];
            low_idx = i;
          }
        }
        for(int j = 0; j < POP; ++j)
          if(j != low_idx){
            for(int i = 0; i < LEN; i++){
                if(random_double() < REC){
                    gene[j][i] = gene[low_idx][i];
                }
                if(random_double() < MUT){
                    gene[j][i] = 1 - gene[j][i];  
                }
            }        
              score[j] = evaluate(j);
        }
    }
    
    low_idx = -1;
    low = 0;
    for(int i =0; i < POP; ++i)
      if((low_idx == -1 || score[i] < low) && score[i] != -1){
        low = score[i];
        low_idx = i;
      }
    
    if(low_idx != -1){
     // printf("%f %f\n", low, evaluate(low_idx));
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
    printf("\n=========================================================================\n");
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
}

int main(){
    srand(getpid()); 
    
    for(int i = 0; i < POP; i++){
        for(int j = 0; j < LEN; j++){
            value[i][j] = rand() % 9 + 1;
        }
    }
    run();
    return 0;
}

