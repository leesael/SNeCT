######################################################################################################
#   SNeCT: Integrative cancer data analysis via large scale network constrained tensor decomposition
#   
#   Authors: Dongjin Choi (skywalker5@snu.ac.kr), Seoul National University
#            Lee Sael (sael@sunykorea.ac.kr), SUNY Korea
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
    
	SNeCT performs tensor decomposition with network constraintinto HOSVD format.
    Moreover, factors extracted from SNeCT are used to analyze subtypes for each modes, patients, genes, and platforms.
	
2. Usage
	
	[Step 1] Install Armadillo and OpenMP libraries.

		Armadillo and OpenMP libraries are required to run SNeCT.

		Armadillo library is attached in lib directory, and also available at http://arma.sourceforge.net/download.html.

		Notice that Armadillo needs LAPACK and BLAS libraries, and they are also attached in lib directory.

		Above OpenMP version 2.0 is required for SNeCT.

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
 		*Notice that SNeCT ver 1.0 uses static variables for faster performance. We are planning to make dynamic version for next version.
 	
 	[Step 3] Compile and run S3CMTF

		If you successfully install all libraries, "make" command will create a binary file, "SNeCT".

		The binary file takes three arguments, which are path of configuration file, path of input tensor file, path of result directory.

		ex) ./SNeCT config.txt input_train.txt result

		If you put command properly, SNeCT will write all values of factor matrices and core tensor in the path of result.

		ex) result/FACTOR1, result/CORETENSOR

3. Demo

	Please see demo folder to understand how to run S3CMTF. Demo tensor is a part of PanCan12 dataset, created as 100x1,000x5 size with 279,906 observable entries.