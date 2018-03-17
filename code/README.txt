######################################################################################################
#   SNeCT: Integrative cancer data analysis via large scale network constrained tensor decomposition
#   
#   Authors: Dongjin Choi (skywalker5@snu.ac.kr), Seoul National University
#            Lee Sael (saellee@snu.ac.kr), SUNY Korea
#   
#   Version : 1.0
#   Date: 2017-10-10
#   Main contact: Dongjin Choi
#
#   This software is free of charge under research purposes.
#   For commercial purposes, please contact the author.
######################################################################################################

1. Introduction

    SNeCT is a fast and scalable method for tensor decomposition with network constraint
    
	SNeCT performs tensor decomposition with network constraint into HOSVD format.
    Moreover, factors extracted from SNeCT are used to analyze subtypes for each modes, patients, genes, and platforms.

2. Files
	
	This package contains following files

	- Makefile : generate excutable SNeCT
	- SNeCT.cpp : source code of SNeCT
	- SNeCT_prediction.cpp : source code of profile prediction for new query tensor
	- demo/ : demo running files
	  - demo/demo.sh : shell script for demo run
	  - demo/demo_prediction.sh : shell script for demo run of query profile prediction
	  - demo/pancan12_demo.tensor : small tensor data for demo run
	  - demo/gene_network_demo.matrix : small matrix data for demo run
	  - demo/query_tensor : small query tensor for patient profile prediction demo
	  - demo/config_demo.txt : SNeCT configuration for demo run
	  - demo/SNeCT : SNeCT executable file
	  - demo/SNeCT_predict : SNeCT_predict executable file
	- lib/ : library files
	  - lib/armadillo-7.700.0.tar.xz : armadillo linear algebra library
	  - lib/blas-3.7.0.tgz : BLAS linear algebra package
	  - lib/lapack-3.7.0.tgz : LAPACK linear algebra package

3. Usage

	**This program is available in Linux environment
	
	[Step 1] Install Armadillo and OpenMP libraries.

		Armadillo and OpenMP libraries are required to run SNeCT.

		Armadillo library is attached in lib directory, and also available at http://arma.sourceforge.net/download.html.

		Notice that Armadillo needs LAPACK and BLAS libraries, and they are also attached in lib directory.

		OpenMP version 2.0 is required for parallel SGD update of SNeCT.

	[Step 2] Adjusting config and source files

		Before you run SNeCT, you need to edit configuration and source files appropriately.

		The format of configuration file is as follows.

		[Line 1]  Tensor order 'N'
		[Line 2]  Tensor dimensionalities I_n (n=1...N) 'I_1' 'I_2' ... 'I_N'
		[Line 3]  Tensor rank J_n (n=1...N) 'J_1' 'J_2' ... 'J_N'
		[Line 4]  Number of parallel threads P
		[Line 5]  Number of tensor entries
		[Line 6]  Number of network constraint
		[Next lines repeated for the number of coupled matrices]
			[Line 6+i]  'Constrained mode' 'Path of i-th network matrix' 'Number of entries of i-th network matrix'
			
		
		After editing configuration file, all pre-defined values in source file (SNeCT.cpp) must be adjusted.
		The details of pre-defined values are as follows.

		#define MAX_ORDER 							<- Must be larger than tensor order N
		#define MAX_INPUT_DIMENSIONALITY 			<- Must be larger than max of tensor dimensionalities I_n
		#define MAX_CORE_TENSOR_DIMENSIONALITY		<- Must be larger than max of tensor ranks J_n
		#define MAX_ENTRY		  					<- Must be larger than number of observable entries of input tensor 
		#define MAX_CORE_SIZE						<- Must be larger than number entries of core tensor
		#define MAX_ITER	 						<- Maximum iteration number, default value is set to 2000
 
 		*The example of configuration file is in demo folder.
 	
 	[Step 3] Compile and run SNeCT

		If you successfully install all libraries, "make" command will create a binary file, "SNeCT".

		The binary file takes three arguments, which are path of configuration file, path of input tensor file, path of result directory.

		ex) ./SNeCT config.txt input_train.txt result

		If you put command properly, SNeCT will write all values of factor matrices and core tensor in the path of result.

		ex) result/FACTOR1, result/CORETENSOR

	[Step 4] Query tensor profiling

		After you successfully factorize the given data tensor, you can use SNeCT_predict to get embedding of a new patient without decomposing the whole tensor. 

		SNeCT_predict takes 14 additional arguments, which are path of input query tensor, number of query entries, path of first factor matrix, path of second factor matrix, path of third factor matrix, length of first mode, length of second mode, length of third mode, path of core tensor, first mode rank size, second mode rank size, third mode rank size, path to store prediction results.

		ex) ./SNeCT_predict query_tensor 1000 result/FACTOR1 result/FACTOR2 result/FACTOR3 100 1000 5 result/CORETENSOR 10 10 5 prediction

4. Demo

	Please follow this procedure to run demo to understand how to run SNeCT.

	1. cd demo
	2. sh demo.sh

	Then, SNeCT is run on a demo tensor and gene network which is a part of PanCan12 dataset, created as size of 100x1,000x5 with 279,906 observable entries.
	After execution, you can see factorization results in 'result' directory, while the intermediate process is presented on screen. The following output files are generated.
	'demo/result/FACTOR<n>' shows the n-th factor matrix. The factor matrix is a stack of vectors representing latent factor weights of each entity.
	'demo/result/CORETENSOR' shows the core tensor of Tucker decomposition. The core tensor represents the degree of strength between factors in different dimensions.

	After factorization is over successfully, follow this procedure to run prediction demo.

	3. sh demo_prediction.sh

	Then, you can see predicted profile of query patient in 'prediction' directory. The following output files are generated.
	'demo/prediction/PROFILE' shows the predicted profile vector of the query patient tensor. 
	'demo/prediction/TOP10' shows the ID of top-10 similar patients of the query patient.