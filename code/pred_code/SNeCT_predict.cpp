/*
* @file        search_profile_for_query.cpp
* @author      Dongjin Choi (skywalker5@snu.ac.kr), Seoul National University
* @author      Sael Lee (sael@suny.ac.kr), The State University of New York, Korea
* @version     1.0
* @date        2017-08-04
*
* 
*
* This software is free of charge under research purposes.
* For commercial purposes, please contact the author.
*
* Usage:
*   - make
*/

/////    Header files     /////

#include <stdio.h>
#include <algorithm>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <armadillo>
#include <omp.h>
#define ARMA_USE_LAPACK
#define ARMA_USE_BLAS

using namespace std;
using namespace arma;

///////////////////////////////


/////////      Pre-defined values      ///////////

#define MAX_ORDER 4						//The maximum number of order
#define MAX_INPUT_DIMENSIONALITY 100000     //The max dimensionality/mode length of input tensor
#define MAX_CORE_TENSOR_DIMENSIONALITY 100	//The max dimensionality/mode length of core tensor
#define MAX_ENTRY 100000						//The max number of entries in input tensor
#define MAX_CORE_SIZE 625000					//The max number of entries in core tensor
#define MAX_ITER 1000						//The maximum iteration number

/////////////////////////////////////////////////


/////////      Variables           ///////////

int threadsNum, dimensionality[MAX_ORDER], coreSize[MAX_ORDER], queryIndex[MAX_ENTRY][MAX_ORDER], queryEntryNum, coreNum = 1, coreIndex[MAX_CORE_SIZE][MAX_ORDER], iterNum, testTCNum, trainTCNum;
int i, j, k, l, aa, bb, ee, ff, gg, hh, ii, jj, kk, ll;
int indexPermute[MAX_ENTRY], testTCGA[MAX_INPUT_DIMENSIONALITY], trainTCGA[MAX_INPUT_DIMENSIONALITY], tcgaCancer[MAX_INPUT_DIMENSIONALITY], testCancer[MAX_INPUT_DIMENSIONALITY], trainCancer[MAX_INPUT_DIMENSIONALITY];
int top10tcga[11], top10tcga2[11];
double queryEntries[MAX_ENTRY], sTime, sTime1, sTime2;
double facMat[MAX_ORDER][MAX_INPUT_DIMENSIONALITY][MAX_CORE_TENSOR_DIMENSIONALITY], coreEntries[MAX_CORE_SIZE], queryProfile[MAX_CORE_TENSOR_DIMENSIONALITY], tcgaDistance[MAX_INPUT_DIMENSIONALITY], tcgaDistance2[MAX_INPUT_DIMENSIONALITY];
vector<int> coreWhere[MAX_ORDER][MAX_CORE_TENSOR_DIMENSIONALITY];


int totalN;

double lambdaReg;
double initialLearnRate;
double learnRate;

//double tempCore[MAX_CORE_SIZE];
//int Mul[MAX_ORDER], tempPermu[MAX_ORDER], rowcount;
double alpha;
int iter = 0;

/////////////////////////////////////////////////

char* queryPath;
char* factor1Path;
char* factor2Path;
char* factor3Path;
char* corePath;
char* ResultPath;
char* TestTCGAPath;
char* TrainTCGAPath;
char* tcgaCancerPath;

/////////////////////////////////////////////////

double frand(double x, double y) {//return the random value in (x,y) interval
	return ((y - x)*((double)rand() / RAND_MAX)) + x;
}

void Initialize() {
	double Timee = clock();
	FILE *fin = fopen(queryPath, "r");
	initialLearnRate = 0.01;
	learnRate = initialLearnRate;
	alpha = 0.5;
	lambdaReg = 0.1;
	omp_set_num_threads(20);
	for (i = 1; i <= queryEntryNum; i++) {
		fscanf(fin, "%d", &k);
		for (j = 2; j <= 3; j++) {
			fscanf(fin, "%d", &k);
			queryIndex[i][j] = k;
		}
		fscanf(fin, "%lf", &queryEntries[i]);
	}
	fclose(fin);
	FILE *fin1 = fopen(factor1Path, "r");
	for (int i = 1; i <= dimensionality[1]; i++) {
		for (int j = 1; j <= coreSize[1]; j++) {
			fscanf(fin, "%lf", &facMat[1][i][j]);
		}
	}
	fclose(fin1);
	FILE *fin2 = fopen(factor2Path, "r");
	for (int i = 1; i <= dimensionality[2]; i++) {
		for (int j = 1; j <= coreSize[2]; j++) {
			fscanf(fin, "%lf", &facMat[2][i][j]);
		}
	}
	fclose(fin2);
	FILE *fin3 = fopen(factor3Path, "r");
	for (int i = 1; i <= dimensionality[3]; i++) {
		for (int j = 1; j <= coreSize[3]; j++) {
			fscanf(fin, "%lf", &facMat[3][i][j]);
		}
	}
	fclose(fin3);
	FILE *fcore = fopen(corePath, "r");
	for (i = 1; i <= coreNum; i++) {
		for (j = 1; j <= 3; j++) {
			fscanf(fcore, "%d", &coreIndex[i][j]);
		}
		fscanf(fcore, "%lf", &coreEntries[i]);

		for (j = 1; j <= 3; j++) {
			coreWhere[j][coreIndex[i][j]].push_back(i);
		}
	}
	fclose(fcore);
	double initVal = pow((1.0 / coreNum), (1.0 / 4.0));
	for (i = 1; i <= coreSize[1]; i++) {
		queryProfile[i]= frand(initVal / 2, initVal);
	}
	FILE *ftctest = fopen(TestTCGAPath, "r");
	for (i = 1; i <= testTCNum; i++) {
		fscanf(ftctest, "%d", &testTCGA[i]);
		fscanf(ftctest, "%d", &testCancer[i]);
	}
	fclose(ftctest);
	FILE *ftctrain = fopen(TrainTCGAPath, "r");
	for (i = 1; i <= trainTCNum; i++) {
		fscanf(ftctrain, "%d", &trainTCGA[i]);
		fscanf(ftctrain, "%d", &trainCancer[i]);
	}
	fclose(ftctrain);
	FILE *ftcCancer = fopen(tcgaCancerPath, "r");
	for (i = 1; i <= dimensionality[1]; i++) {
		fscanf(ftcCancer, "%d", &j);
		fscanf(ftcCancer, "%d", &tcgaCancer[i]);
	}
	fclose(ftcCancer);
	
	printf("Elapsed Time:\t%lf\n", (clock() - Timee) / CLOCKS_PER_SEC);
	printf("Reading Done.\n");

}
double abss(double x) {
	return x > 0 ? x : -x;
}

//[Input] Input tensor X, initialized core tensor G, and factor matrices A^{(n)} (n=1...N)  
//[Output] Updated factor matrices A^{(n)} (n=1...N)
//[Function] Update all factor matrices according to the differential equation.
void Update_Profile() {
	int i, temp;
	//Generate random permutation
	for (i = totalN; i >= 1; --i) {
		indexPermute[i] = i;
	}
	for (i = totalN; i >= 1; --i) {
		j = (rand() % i) + 1;
		temp = indexPermute[i];
		indexPermute[i] = indexPermute[j];
		indexPermute[j] = temp;
	}
#pragma omp parallel for schedule(static)
	for (i = 1; i <= totalN; i++)
	{
		int current_input_entry = indexPermute[i];
		double currentVal = queryEntries[current_input_entry];

		double current_estimation = 0;
		
		double CoreProducts[MAX_CORE_TENSOR_DIMENSIONALITY];
		for (j = 1; j <= coreSize[1]; j++) {
			CoreProducts[j] = 0;
		}
		int ii;
		for (ii = 1; ii <= coreNum; ii++) {
			double temp = coreEntries[ii];
			int jj;
			for (jj = 2; jj <= 3; jj++) {
				temp *= facMat[jj][queryIndex[current_input_entry][jj]][coreIndex[ii][jj]];
			}
			CoreProducts[coreIndex[ii][1]] += temp;
			temp *= queryProfile[coreIndex[ii][1]];
			current_estimation += temp;
		}
		for (k = 1; k <= coreSize[1]; k++) {
			queryProfile[k]=queryProfile[k]
				- learnRate*(lambdaReg/(double)queryEntryNum*queryProfile[k]
					- (currentVal - current_estimation)*CoreProducts[k]);
		}
	}

}

//[Input] Input tensor and initialized core tensor and factor matrices
//[Output] Updated core tensor and factor matrices
//[Function] Performing main algorithm which updates core tensor and factor matrices iteratively
void Search_Profile() {
	printf("Search Profile START\n");

	double sTime = omp_get_wtime();
	double avertime = 0;
	learnRate = initialLearnRate;
	while (1) {

		double itertime = omp_get_wtime(), steptime;
		steptime = itertime;

		Update_Profile();
		printf("Iteration Time : %lf\n", omp_get_wtime() - steptime);
		steptime = omp_get_wtime();

		avertime += omp_get_wtime() - itertime;
		printf("iter%d :      Elapsed Time : %lf\n", ++iter, omp_get_wtime() - itertime);
		
		learnRate = initialLearnRate / (1 + alpha*iter);
		if (iter == iterNum) break;
	}

	avertime /= iter;

	printf("\niterNum ended.\tAverage iteration time : %lf\n", avertime);

}
double euclid_dist(double x[], double y[], int size) {//return the random value in (x,y) interval
	double dist = 0;
	for ( k = 1; k <= size; k++)
	{
		dist += pow((x[k]) - (y[k]),2.0) ;
	}
	return pow(dist,0.5);
}
double cosine_dist(double x[], double y[], int size) {//return the random value in (x,y) interval
	double inner_prod = 0;
	double x_norm = 0, y_norm = 0;
	for (k = 1; k <= size; k++)
	{
		inner_prod += (x[k])*(y[k]);
		x_norm += x[k] * x[k];
		y_norm += y[k] * y[k];
	}
	return 1-inner_prod/(pow(x_norm*y_norm, 0.5));
}

void Find_Top_K() {
#pragma omp parallel for schedule(static)
	for (i = 1; i <= dimensionality[1]; i++)
	{
		int breakFlag = 0;
		for (j = 1; j <= testTCNum; j++) {
			if (i == testTCGA[j]) {
				tcgaDistance[i] = 100000;
				breakFlag = 1;
				break;
			}
		}
		if (breakFlag) continue;
		tcgaDistance[i] = euclid_dist(queryProfile, facMat[1][i], coreSize[1]);
	}
	for (i = 1; i <= 10; i++) {
		double minDist=1000000;
		for (j = 1; j <= dimensionality[1]; j++) {
			if (tcgaDistance[j] < minDist) {
				int flag = 0;
				for (k = 1; k <= i - 1; k++) {
					if (j == top10tcga[k]) flag = 1;
				}
				if (!flag) {
					minDist = tcgaDistance[j];
					top10tcga[i] = j;
				}
			}
		}
	}
#pragma omp parallel for schedule(static)
	for (i = 1; i <= dimensionality[1]; i++)
	{
		int breakFlag = 0;
		for (j = 1; j <= testTCNum; j++) {
			if (i == testTCGA[j]) {
				tcgaDistance2[i] = 100000;
				breakFlag = 1;
				break;
			}
		}
		if (breakFlag) continue;
		tcgaDistance2[i] = cosine_dist(queryProfile, facMat[1][i], coreSize[1]);
	}
	for (i = 1; i <= 10; i++) {
		double minDist = 1000000;
		for (j = 1; j <= dimensionality[1]; j++) {
			if (tcgaDistance2[j] < minDist) {
				int flag = 0;
				for (k = 1; k <= i - 1; k++) {
					if (j == top10tcga2[k]) flag = 1;
				}
				if (!flag) {
					minDist = tcgaDistance2[j];
					top10tcga2[i] = j;
				}
			}
		}
	}
}

//[Input] Updated core tensor G and factor matrices A^{(n)} (n=1...N)
//[Output] core tensor G in sparse tensor format and factor matrices A^{(n)} (n=1...N) in full-dense matrix format
//[Function] Writing all factor matrices and core tensor in result path
void Print() {
	printf("\nWriting Extracted Profile to file...\n");
	char temp[50];
	sprintf(temp, "mkdir %s", ResultPath);
	system(temp);
	sprintf(temp, "%s/PROFILE", ResultPath);
	FILE *fin = fopen(temp, "w");
	for (k = 1; k <= coreSize[1]; k++) {
		fprintf(fin, "%f\t", queryProfile[k]);
	}
	fprintf(fin, "\n");
	char temp2[50];
	sprintf(temp2, "%s/TOP10", ResultPath);
	FILE *fin2 = fopen(temp2, "w");
	for (k = 1; k <= 10; k++) {
		fprintf(fin2, "%d\t%d\t%d\t%d\n", top10tcga[k],tcgaCancer[top10tcga[k]], top10tcga2[k], tcgaCancer[top10tcga2[k]]);
	}
	char temp3[50];
	sprintf(temp3, "%s/TIME", ResultPath);
	FILE *fin3 = fopen(temp3, "w");
	fprintf(fin3, "%f", sTime2);
}

//[Input] Path of configuration file, input tensor file, and result directory
//[Output] Core tensor G and factor matrices A^{(n)} (n=1...N)
//[Function] Performing P-Tucker to factorize partially observable tensor
int main(int argc, char* argv[]) {
	if (argc != 20) {
		printf("please input proper arguments\n");
		return 0;
	}

	queryPath = argv[1];
	queryEntryNum = atoi(argv[2]);
	totalN = queryEntryNum;
	factor2Path = argv[3];
	dimensionality[2] = atoi(argv[4]);
	factor3Path = argv[5];
	dimensionality[3] = atoi(argv[6]);
	corePath = argv[7];
	coreSize[1] = atoi(argv[8]);
	coreNum *= coreSize[1];
	coreSize[2] = atoi(argv[9]);
	coreNum *= coreSize[2];
	coreSize[3] = atoi(argv[10]);
	coreNum *= coreSize[3];
	iterNum = atoi(argv[11]);
	ResultPath = argv[12];
	TestTCGAPath = argv[13];
	testTCNum = atoi(argv[14]);
	TrainTCGAPath = argv[15];
	trainTCNum = atoi(argv[16]);
	factor1Path = argv[17];
	dimensionality[1] = atoi(argv[18]);
	tcgaCancerPath = argv[19];

	srand((unsigned)time(NULL));

	sTime = clock();
	sTime1 = omp_get_wtime();
cout<<1<<endl;
	Initialize();
cout<<2<<endl;
	Search_Profile();

	Find_Top_K();

	sTime2 = omp_get_wtime()-sTime1;

	Print();

	return 0;
}
