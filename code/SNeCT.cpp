/*
* @file        SNeCT.cpp
* @author      Dongjin Choi (skywalker5@snu.ac.kr), Seoul National University
* @author      Lee Sael (sael@sunykorea.ac.kr), SUNY Korea
* @version     1.0
* @date        2017-10-10
*
* SNeCT: Integrative cancer data analysis via large scale network constrained tensor decomposition
*
* This software is free of charge under research purposes.
* For commercial purposes, please contact the author.
*
* Usage:
* To compile SNeCT, type following command:
*   - make all
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

#define MAX_ORDER 4							//The max order/way of input tensor
#define MAX_INPUT_DIMENSIONALITY 15000     //The max dimensionality/mode length of input tensor
#define MAX_CORE_TENSOR_DIMENSIONALITY 100	//The max dimensionality/mode length of core tensor
#define MAX_ENTRY 2500000						//The max number of entries in input tensor
#define MAX_CORE_SIZE 100000					//The max number of entries in core tensor
#define MAX_ITER 2000						//The maximum iteration number

/////////////////////////////////////////////////


/////////      Variables           ///////////

int threadsNum, order, dimensionality[MAX_ORDER], coreSize[MAX_ORDER], trainIndex[MAX_ENTRY][MAX_ORDER], trainEntryNum, coreNum = 1, coreIndex[MAX_CORE_SIZE][MAX_ORDER], coupleDim[MAX_ORDER], iterNum=100, nanFlag = 0, nanCount = 0;
int i, j, k, l, aa, bb, ee, ff, gg, hh, ii, jj, kk, ll;
int indexPermute[MAX_ORDER*MAX_ENTRY];
double trainEntries[MAX_ENTRY], sTime, trainRMSE, minv = 2147483647, maxv = -2147483647;
double facMat[MAX_ORDER][MAX_INPUT_DIMENSIONALITY][MAX_CORE_TENSOR_DIMENSIONALITY], coreEntries[MAX_CORE_SIZE];

int numCoupledMat;
int coupleEntryNum[MAX_ORDER*2];
int entryNumCum[MAX_ORDER*2];
int totalN = 0;
int coupleMatIndex[MAX_ORDER][MAX_ENTRY][3];
double lambdaGraph;
double coupledEntries[MAX_ORDER][MAX_ENTRY];
int coupleWhere[MAX_ORDER][2][MAX_INPUT_DIMENSIONALITY];

double errorForTrain[MAX_ENTRY], trainNorm, error;

vector<int> trainWhere[MAX_ORDER][MAX_INPUT_DIMENSIONALITY], coreWhere[MAX_ORDER][MAX_CORE_TENSOR_DIMENSIONALITY];
double lambdaReg=0.1;
double initialLearnRate=0.001;
double learnRate;
double tempCore[MAX_CORE_SIZE];
int Mul[MAX_ORDER], tempPermu[MAX_ORDER], rowcount;
double timeHistory[MAX_ITER], trainRmseHistory[MAX_ITER];
double alpha=0.5;
int iter = 0;

/////////////////////////////////////////////////

char* ConfigPath;
char* TrainPath;
char CoupledPath[MAX_ORDER][100];
char* ResultPath;

/////////////////////////////////////////////////

//[Input] Lower range x, upper range y
//[Output] Random double precision number between x and y
//[Function] Generate random floating point number between given two numbers
double frand(double x, double y) { //return the random value in (x,y) interval
	return ((y - x)*((double)rand() / RAND_MAX)) + x;
}

//[Input] A double precision number x
//[Output] Absolute value of x
//[Function] Get absolute value of input x
double abss(double x) { //return the absolute value of x
	return x > 0 ? x : -x;
}

//[Input] Input tensor as a sparse tensor format, and a network constraint as a sparse matrix format
//[Output] Input tensor X and network constraint Y loaded on memory
//[Function] Getting all entries of input tensor X and network constraint Y
void Getting_Input() {
	FILE *fin = fopen(TrainPath, "r");
	FILE *fcouple;
	FILE *config = fopen(ConfigPath, "r");
	//INPUT
	double Timee = clock();
	printf("Reading input\n");
	fscanf(config, "%d", &order);
	for (i = 1; i <= order; i++) {
		fscanf(config, "%d", &dimensionality[i]);
	}
	for (i = 1; i <= order; i++) {
		fscanf(config, "%d", &coreSize[i]);
		coreNum *= coreSize[i];
	}
	fscanf(config, "%d", &threadsNum);
	omp_set_num_threads(threadsNum);

	fscanf(config, "%d", &trainEntryNum);
	totalN += trainEntryNum;

	fscanf(config, "%d", &numCoupledMat);

	for (i = 1; i <= numCoupledMat; i++) {
		fscanf(config, "%d", &coupleDim[i]);
		fscanf(config, "%s", &CoupledPath[i]);
		fscanf(config, "%d", &coupleEntryNum[i]);
		entryNumCum[i] = totalN;
		totalN += coupleEntryNum[i];
	}
	fclose(config);

	entryNumCum[numCoupledMat + 1] = totalN;


	for (i = 1; i <= numCoupledMat; i++) {
		fcouple = fopen(CoupledPath[i], "r");
		for (j = 1; j <= coupleEntryNum[i]; j++) {
			fscanf(fcouple, "%d", &k);
			coupleMatIndex[i][j][1] = k;

			fscanf(fcouple, "%d", &k);
			coupleMatIndex[i][j][2] = k;
			coupleWhere[i][1][k]++;
			fscanf(fcouple, "%lf", &coupledEntries[i][j]);
		}
	}

	for (i = 1; i <= trainEntryNum; i++) {
		for (j = 1; j <= order; j++) {
			fscanf(fin, "%d", &k);
			trainIndex[i][j] = k;
			trainWhere[j][k].push_back(i);
		}
		fscanf(fin, "%lf", &trainEntries[i]);
		trainNorm += trainEntries[i] * trainEntries[i];
		if (minv > trainEntries[i]) minv = trainEntries[i];
		if (maxv < trainEntries[i]) maxv = trainEntries[i];
	}
	trainNorm = sqrt(trainNorm);

	fclose(fin);

	printf("Elapsed Time:\t%lf\n", (clock() - Timee) / CLOCKS_PER_SEC);
	printf("Reading Done.\nNorm : %lf\nInitialize\n", trainNorm);
}

//[Input] Size of the input tensor and core tensor size
//[Output] Initialized core tensor G and factor matrices U^{(n)} (n=1...N)
//[Function] Initialize all factor matrices and core tensor.
void Initialize() {	//INITIALIZE
	double Timee = clock();
	iter = 0;
	double initVal = pow((maxv / coreNum), (1 / double(order + 1)));

	for (i = 1; i <= order; i++) {
		for (j = 1; j <= dimensionality[i]; j++) {
			for (k = 1; k <= coreSize[i]; k++) {
				facMat[i][j][k] = frand(initVal / 2, initVal);
			}
		}
	}
	for (i = 1; i <= coreNum; i++) {
		coreEntries[i] = frand(initVal / 2, initVal);

		for (j = 1; j <= order; j++) {
			coreIndex[i][j] = coreIndex[i - 1][j];
		}
		coreIndex[i][order]++;  k = order;
		while (coreIndex[i][k] > coreSize[k]) {
			coreIndex[i][k] -= coreSize[k];
			coreIndex[i][k - 1]++; k--;
		}
		if (i == 1) {
			for (j = 1; j <= order; j++) coreIndex[i][j] = 1;
		}

		for (j = 1; j <= order; j++) {
			if (nanCount == 1) {
				coreWhere[j][coreIndex[i][j]].push_back(i);
			}
		}
	}
	printf("Elapsed Time:\t%lf\n", (clock() - Timee) / CLOCKS_PER_SEC);
}

//[Input] Input tensor X, initialized core tensor G, and factor matrices U^{(n)} (n=1...N)  
//[Output] Updated factor matrices U^{(n)} (n=1...N)
//[Function] Update all factor matrices using the asynchronous SGD method.
void Update_Factor_Matrices() {
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
		if (1 <= indexPermute[i] && indexPermute[i] <= trainEntryNum) {
			int current_input_entry = indexPermute[i];
			double currentVal = trainEntries[current_input_entry];

			double current_estimation = 0;
			double CoreProducts[MAX_CORE_SIZE];
			int ii;
			for (ii = 1; ii <= coreNum; ii++) {
				double temp = coreEntries[ii];
				int jj;
				for (jj = 1; jj <= order; jj++) {
					temp *= facMat[jj][trainIndex[current_input_entry][jj]][coreIndex[ii][jj]];
				}
				CoreProducts[ii] = temp;
				current_estimation += temp;
			}
			double Sigma[MAX_CORE_TENSOR_DIMENSIONALITY];
			//Updating Factor matrices
			int jjj;
			for (jjj = 1; jjj <= order; jjj++) {//i-th Factor Matrix
				int l;
				int column_size = coreSize[jjj];
				for (l = 1; l <= column_size; l++) {
					int core_nonzeros = coreWhere[jjj][l].size();
					int k;
					Sigma[l] = 0;
					if (abss(facMat[jjj][trainIndex[current_input_entry][jjj]][l]) < 0.00000001) {
						facMat[jjj][trainIndex[current_input_entry][jjj]][l] = 0.0000001;
						continue;
					}
					for (k = 0; k < core_nonzeros; k++) {
						int current_core_entry = coreWhere[jjj][l][k];
						Sigma[l] += CoreProducts[current_core_entry];
					}
					Sigma[l] /= facMat[jjj][trainIndex[current_input_entry][jjj]][l];
				}

				for (k = 1; k <= column_size; k++) {
					int II = trainIndex[current_input_entry][jjj];
					facMat[jjj][II][k] = facMat[jjj][II][k]
						- learnRate*(lambdaReg / (double)(trainWhere[jjj][II].size())*facMat[jjj][II][k]
							- (currentVal - current_estimation)*Sigma[k]);
				}

			}
			//Update_Core_Tensor

			if (i%threadsNum == 0) {
				int kk;
				for (kk = 1; kk <= coreNum; kk++) {
					double temp2;
					if (abss(coreEntries[kk]) < 0.00000001) {
						coreEntries[kk] = 0.0000001;
					}
					temp2 = CoreProducts[kk] / coreEntries[kk];
					coreEntries[kk] = coreEntries[kk] + learnRate*(currentVal - current_estimation)*temp2 - learnRate*lambdaReg*coreEntries[kk] / trainEntryNum;
				}
			}

		}
		else {
			int coupleMode;
			int current_input_entry;
			int ii;
			int cplNum;
			for (ii = 1; ii <= numCoupledMat; ii++) {
				if (indexPermute[i]>entryNumCum[ii + 1]) {
					continue;
				}
				coupleMode = coupleDim[ii];
				cplNum = ii;
				current_input_entry = indexPermute[i] - entryNumCum[ii];
				break;
			}
			double currentCoupledVal = coupledEntries[cplNum][current_input_entry];
			int column_size = coreSize[coupleMode];

			//Updating Factor matrix vector

			double Sigma[MAX_CORE_TENSOR_DIMENSIONALITY];
			int l;
			for (l = 1; l <= column_size; l++) {
				Sigma[l] = lambdaGraph* currentCoupledVal*(
					facMat[coupleMode][coupleMatIndex[cplNum][current_input_entry][1]][l]
					- facMat[coupleMode][coupleMatIndex[cplNum][current_input_entry][2]][l]);
			}
			for (l = 1; l <= column_size; l++) {
				facMat[coupleMode][coupleMatIndex[cplNum][current_input_entry][1]][l] -=
					learnRate*Sigma[l];
			}
			for (l = 1; l <= column_size; l++) {
				facMat[coupleMode][coupleMatIndex[cplNum][current_input_entry][2]][l] +=
					learnRate*Sigma[l];
			}
		}
	}

}

//[Input] The index set of observable entries in X
//[Output] Total reconstruction error of the value of the observable entries in X
//[Function] Getting reconstruction error by subtracting reconstructed values from observed entries
void Reconstruction() {
	error = 0;
#pragma omp parallel for 
	for (i = 1; i <= trainEntryNum; i++) {
		errorForTrain[i] = trainEntries[i];
	}

#pragma omp parallel for 
	for (i = 1; i <= trainEntryNum; i++) {
		int j;
		for (j = 1; j <= coreNum; j++) {
			double temp = coreEntries[j];
			int k;
			for (k = 1; k <= order; k++) {
				temp *= facMat[k][trainIndex[i][k]][coreIndex[j][k]];
			}
			errorForTrain[i] -= temp;
		}
		errorForTrain[i]=errorForTrain[i] * errorForTrain[i];
	}

#pragma omp parallel for reduction(+:error)

	for (i = 1; i <= trainEntryNum; i++) {
		error += errorForTrain[i];
	}
	if (trainNorm == 0) trainRMSE = 1;
	else trainRMSE = sqrt(error) / sqrt(trainEntryNum);

}

//[Input] Updated factor matrices U^{(n)} (n=1...N)
//[Output] Orthonormal factor matrices U^{(n)} (n=1...N) and updated core tensor G
//[Function] Orthogonalize all factor matrices and update core tensor simultaneously.
void Orthogonalize() {
	Mul[order] = 1;
	for (i = order - 1; i >= 1; i--) {
		Mul[i] = Mul[i + 1] * coreSize[i + 1];
	}
	for (i = 1; i <= order; i++) {
		mat Q, R;
		mat X = mat(dimensionality[i], coreSize[i]);
		for (k = 1; k <= dimensionality[i]; k++) {
			for (l = 1; l <= coreSize[i]; l++) {
				X(k - 1, l - 1) = facMat[i][k][l];
			}
		}
		qr_econ(Q, R, X);
		double coeff = 1;
		for (k = 1; k <= dimensionality[i]; k++) {
			for (l = 1; l <= coreSize[i]; l++) {
				facMat[i][k][l] = Q(k - 1, l - 1)*coeff;
			}
		}
		for (j = 1; j <= coreNum; j++) {
			tempCore[j] = 0;
		}
		for (j = 1; j <= coreNum; j++) {
			for (k = 1; k <= coreSize[i]; k++) {
				int cur = j + (k - coreIndex[j][i])*Mul[i];
				tempCore[cur] += coreEntries[j] * (R(k - 1, coreIndex[j][i] - 1) / coeff);
			}
		}
		for (j = 1; j <= coreNum; j++) {
			coreEntries[j] = tempCore[j];
		}

	}
}

//[Input] Input tensor and initialized core tensor and factor matrices
//[Output] Updated core tensor and factor matrices
//[Function] Performing main algorithm which updates core tensor and factor matrices iteratively
void SNeCT() {
	printf("SNeCT START\n");

	double sTime = omp_get_wtime();
	double avertime = 0;
	learnRate = initialLearnRate;
	while (1) {

		double itertime = omp_get_wtime(), steptime;
		steptime = itertime;

		Update_Factor_Matrices();
		printf("Factor Time : %lf\n", omp_get_wtime() - steptime);
		steptime = omp_get_wtime();

		Reconstruction();
		printf("Recon Time : %lf\n", omp_get_wtime() - steptime);
		steptime = omp_get_wtime();

		avertime += omp_get_wtime() - itertime;
		printf("iter%d :      RMSE : %lf\tElapsed Time : %lf\n", ++iter, trainRMSE, omp_get_wtime() - itertime);
		
		learnRate = initialLearnRate / (1+alpha*iter);
		timeHistory[iter - 1] = omp_get_wtime() - itertime;
		trainRmseHistory[iter - 1] = trainRMSE;
		if (trainRMSE != trainRMSE) {
			nanFlag = 1;
			break;
		}
		if (iter == iterNum) break;
	}

	avertime /= iter;

	printf("\nAll iterations ended.\tRMSE : %lf\tAverage iteration time : %lf\n", trainRMSE, avertime);

	printf("\nOrthogonalize and update core tensor...\n\n");

	Orthogonalize();

	printf("\nTotal update ended.\tFinal RMSE : %lf\tTotal Elapsed time: %lf\n", trainRMSE, omp_get_wtime() - sTime);
}

//[Input] Factorized results: core tensor G and factor matrices U^{(n)} (n=1...N)
//[Output] Core tensor G in sparse tensor format and factor matrices U^{(n)} (n=1...N) in full-dense matrix format
//[Function] Writing all factor matrices and core tensor in the result path
void Print() {
	printf("\nWriting factor matrices and the core tensor to file...\n");
	char temp[50];
	sprintf(temp, "mkdir %s", ResultPath);
	system(temp);
	for (i = 1; i <= order; i++) {
		sprintf(temp, "%s/FACTOR%d", ResultPath, i);
		FILE *fin = fopen(temp, "w");
		for (j = 1; j <= dimensionality[i]; j++) {
			for (k = 1; k <= coreSize[i]; k++) {
				fprintf(fin, "%f\t", facMat[i][j][k]);
			}
			fprintf(fin, "\n");
		}
		fclose(fin);
	}
	sprintf(temp, "%s/CORETENSOR", ResultPath);
	FILE *fcore = fopen(temp, "w");
	for (i = 1; i <= coreNum; i++) {
		for (j = 1; j <= order; j++) {
			fprintf(fcore, "%d\t", coreIndex[i][j]);
		}
		fprintf(fcore, "%f\n", coreEntries[i]);
	}
	fclose(fcore);
}

//[Input] History of running time per iteration and train RMSE per iteration
//[Output] A file in which running time and train RMSE for each iteration is written
//[Function] Writing running time and train RMSE for each iteration in an output file
void PrintTime() {
	printf("\nWriting Time and error to file...\n");
	char temp[50];
	sprintf(temp, "mkdir %s", ResultPath);
	system(temp);
	sprintf(temp, "%s/TIMEERROR", ResultPath);
	FILE *ftime = fopen(temp, "w");
	for (i = 0; i < iter; i++) {
		fprintf(ftime, "%f\t%f\n", timeHistory[i], trainRmseHistory[i]);
	}
	fclose(ftime);
}

//[Input] Path of configuration file, input tensor file, and result directory
//[Output] Core tensor G and factor matrices U^{(n)} (n=1...N)
//[Function] Performing SNeCT which decomposes a network-constrained tensor
int main(int argc, char* argv[]) {
	if (argc == 4) {

		ConfigPath = argv[1];
		TrainPath = argv[2];
		ResultPath = argv[3];
	}
	else {
		printf("please input proper arguments\n");
		return 0;
	}

	srand((unsigned)time(NULL));

	sTime = clock();

	Getting_Input();

	do {
		nanFlag = 0;
		nanCount++;

		Initialize();

		SNeCT();
	} while (nanFlag && nanCount<10);

	Print();

	//PrintTime(); //Use for experiment

	return 0;
}
