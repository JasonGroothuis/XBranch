######################################################
# NVIDIA Cuda Compiler detection and setup           #
#                                                    #
# From                                               #
# http://forums.nvidia.com/index.php?showtopic=53006 #
######################################################

# If cuda is requested to be enabled
AC_DEFUN([CHECK_NVCC],
[AC_MSG_CHECKING([for nvcc])

        # Search nvcc compiler
        AC_PATH_PROG(NVCC, nvcc, "no")
        AS_IF([test "x$NVCC" = "xno"],[
                        AC_MSG_ERROR([NVCC compiler not found!])
        ])

        # Check nvcc version, should be at least 3.0

	CUDA_VERSION=$(nvcc --version | grep "release " | sed "s|.*release \([[0-9]]\).\([[0-9]]\).*|\1\2|")
        AS_IF([test $CUDA_VERSION -ge 32],
                        [],
                        [AC_MSG_ERROR([NVCC compiler version is NOT at least 3.2!])
        ])

#	AC_ARG_WITH(cuda,
#		[AS_HELP_STRING([--with-cuda], [Enable cuda based modules for architecture ARCH (see nvcc option -arch).])],
#		[CUDA_ARCH+=" -arch=$withval"],
#		[CUDA_ARCH=" -gencode=arch=compute_10,code=sm_10 -gencode=arch=compute_13,code=sm_13 -gencode=arch=compute_20,code=compute_20"])

                # Set CUDA_CFLAGS to $NVCC, where substring "bin/nvcc"
                # is substituted by "include".
                CUDA_CFLAGS+=" -I${NVCC/bin\/nvcc/include}"
                CFLAGS+=" -I${NVCC/bin\/nvcc/include}"

                #Set CUDA_CFLAGS to $NVCC, where substring "bin/nvcc"
                #is substituted by "lib".
                CUDA_LIBS+=" -L${NVCC/bin\/nvcc/lib}"

                # If $build_cpu contains "_64", append "64" to CUDA_LIBS
                AS_IF([echo $build_cpu | grep -q "_64"],
                     [CUDA_LIBS+="64"])

                # Append " -lcudart -lcufft" to CUDA_LIBS
                CUDA_LIBS+=" -lcudart -lcufft"

        # Symbolize that cuda is wanted
        with_cuda=$enableval

        # Make variables available in Makefile.am
        AC_SUBST(CFLAGS)
        AC_SUBST(CUDA_CFLAGS)
        AC_SUBST(CUDA_LIBS)
        AC_SUBST(NVCC)

	# If debug flag is set apply debugging compilation flags, otherwise build compilation flags
        AS_IF([test "x$DEBUG" = "xtrue"],
                [NVCCFLAGS="-g --use_fast_math --ptxas-options=\"-v\" --compiler-options \"\$(AM_CXXFLAGS) \$(CXXFLAGS) -fno-strict-aliasing\""],
                [NVCCFLAGS="-O3 --use_fast_math --ptxas-options=\"-v\" --compiler-options \"\$(AM_CXXFLAGS) \$(CXXFLAGS) -fno-strict-aliasing\""])

        # If $build_cpu contains "_64", append "-m64" to NVCCFLAGS
        # AS_IF([echo $build_cpu | grep -q "_64"],
        if test x_$target_cpu = x_x86_64 ; then
                NVCCFLAGS+=" -m64 "
        else
                NVCCFLAGS+=" -m32 "
        fi
        # Add architecture to flags
        NVCCFLAGS+=" -gencode arch=compute_10,code=sm_10 -gencode arch=compute_11,code=sm_11 -gencode arch=compute_12,code=sm_12 -gencode arch=compute_13,code=sm_13"
	NVCCFLAGS+=" -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21 -gencode arch=compute_20,code=compute_20"
])
# Make NVCCFLAGS available in Makefile.am
AC_SUBST(NVCCFLAGS)

AC_SUBST(CUDA_VERSION)



