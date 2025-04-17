The files in this folder are used for compilation into LAMMPS
Written by Han Chenxu on October 11, 2023
If you have any questions, please contact me via email: hanchenxu615@gmail.com

Note: Subject to my code level, this code is the initial stage
When you want to try convolution more than 6 times,Please use gcnff3 as a base in this document
Familiarize yourself with the following rules:  
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'	//To add more interaction_blocks -begin- //	'
'				... 			'
' Code that needs to be reviewed and repeated is here! 	'
'			_$num_ just revalue the $num	'
'				...			'
'	//To add more interaction_blocks -end-//	'
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''

Below, to explain the installation method:
 1. Make sure your GCC supports c++ 14  (GCC 6.1 up until GCC 10 ) and CMake version is greater than 3.10;
		▲  cmake --version
		▲  gcc --version
 2. Download the stable version of LAMMPS;
 3. Download LibTorch auxiliary package;(You must make sure that GCC supports the version you download);
 4. Move the folder ${GCNFF}/USER-GCNFF to the ${lammps-yourdownload}/src ;
 5. Modify ${lammps-yourdownload}/cmake/CMakeLists.txt to CMakeLists.txt(I provided in ${GCNFF}/example/lammps/cmake/CMakeLists.txt);
	I mainly fixed two parts of the code:  
		one, "USER-GCNFF" was added to the potential list;
		two, Link to libtorch here writen by HCX:
			"find_package(Torch REQUIRED)"
			"target_link_libraries(lammps PRIVATE "${TORCH_LIBRARIES}")"
 6. Build using following orders, and you will generate an executable lmp in the build folder.
	  ▲    cd ${lammps-29Oct20}
      	  ▲    mkdir build & cd build
	  ▲    cmake -DCMAKE_PREFIX_PATH=${libtorch} ../cmake
	  ▲    cmake -D PKG_USER-GCNFF=yes .
	  ▲    cmake -D PKG_${PKGNAME_YOUNEED}=yes/no .
	  ▲    make
Below are all the software versions that I have successfully installed:
	CMake:	cmake-3.10.2-Linux-x86_64
	c++(GCC): 8.3.1 20190311  or 7.5.0
	Lammps:	lammps-29Oct20
	Pytorch(C++ version): libtorch-shared-with-deps-1.8.1+cpu
	
Finally, how to use it in the input file of lammps?
	'''''''''''''''''''''''''''''''''''''''''
	'	pair_style    gcnffX		'
	'	pair_coeff    * * model args	'
	'''''''''''''''''''''''''''''''''''''''''
	▲	X= Number of GCN (=NUM_CONV in input.json)
	▲	model=The name of the potential file exported from GCNFF
	▲	args=List of element names
	
For  Example: 
	pair_style     gcnff3
	pair_coeff     * * potential.txt Hf O
