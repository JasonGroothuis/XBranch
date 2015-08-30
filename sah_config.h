/* sah_config.h.  Generated from sah_config.h.in by configure.  */
/* sah_config.h.in.  Generated from configure.ac by autoheader.  */


#ifndef _SAH_CONFIG_H_
#define _SAH_CONFIG_H_

#ifdef _WIN32
#include "win-sah_config.h"
#else


/* Define if building universal (internal helper macro) */
/* #undef AC_APPLE_UNIVERSAL_BUILD */

/* Define to 1 to build a graphical application */
/* #undef BOINC_APP_GRAPHICS */

/* Define to a string identifying your compiler */
#define COMPILER_STRING "g++ (GCC) 4.4.5 20101112 (Red Hat 4.4.5-2)"

/* Define to 1 if your system stores words within floats with the most
   significant word first */
/* #undef FLOAT_WORDS_BIGENDIAN */

/* Define to 1 if you have the `alloca' function. */
/* #undef HAVE_ALLOCA */

/* Define to 1 if you have the <alloca.h> header file. */
#define HAVE_ALLOCA_H 1

/* Use the Apple OpenGL framework. */
/* #undef HAVE_APPLE_OPENGL_FRAMEWORK */

/* Define to 1 if you have the `atanf' function. */
/* #undef HAVE_ATANF */

/* Define to 1 if you have the `atexit' function. */
#define HAVE_ATEXIT 1

/* Define to 1 if you have the `atoll' function. */
#define HAVE_ATOLL 1

/* Define to 1 if the system has the type `bool'. */
#define HAVE_BOOL 1

/* Define to 1 if you have the <complex.h> header file. */
#define HAVE_COMPLEX_H 1

/* Define to 1 if you have the `cosf' function. */
/* #undef HAVE_COSF */

/* Define to 1 if you have the <dirent.h> header file. */
#define HAVE_DIRENT_H 1

/* Define to 1 if you have the <dlfcn.h> header file. */
#define HAVE_DLFCN_H 1

/* Define to 1 if you have the `dlopen' function. */
/* #undef HAVE_DLOPEN */

/* Define to 1 if you have the <emmintrin.h> header file. */
/* #undef HAVE_EMMINTRIN_H */

/* Define to 1 if you have the <errno.h> header file. */
#define HAVE_ERRNO_H 1

/* Define to 1 if you have the `exit' function. */
#define HAVE_EXIT 1

/* Define to 1 if you have the <fcntl.h> header file. */
#define HAVE_FCNTL_H 1

/* Define to 1 if you have the <floatingpoint.h> header file. */
/* #undef HAVE_FLOATINGPOINT_H */

/* Define to 1 if you have the <float.h> header file. */
#define HAVE_FLOAT_H 1

/* Define to 1 if you have the `floor' function. */
/* #undef HAVE_FLOOR */

/* Define to 1 if you have the `fork' function. */
#define HAVE_FORK 1

/* Define to 1 if you have the `getcwd' function. */
#define HAVE_GETCWD 1

/* Define to 1 if you have the `gethrtime' function. */
/* #undef HAVE_GETHRTIME */

/* Define to 1 if you have the <glaux.h> header file. */
/* #undef HAVE_GLAUX_H */

/* Define to 1 if you have the <glut/glut.h> header file. */
/* #undef HAVE_GLUT_GLUT_H */

/* Define to 1 if you have the <glut.h> header file. */
/* #undef HAVE_GLUT_H */

/* Define to 1 if you have the <glu.h> header file. */
/* #undef HAVE_GLU_H */

/* Define to 1 if you have the <GL/glaux.h> header file. */
/* #undef HAVE_GL_GLAUX_H */

/* Define to 1 if you have the <GL/glut.h> header file. */
/* #undef HAVE_GL_GLUT_H */

/* Define to 1 if you have the <GL/glu.h> header file. */
/* #undef HAVE_GL_GLU_H */

/* Define to 1 if you have the <GL/gl.h> header file. */
/* #undef HAVE_GL_GL_H */

/* Define to 1 if you have the <gl.h> header file. */
/* #undef HAVE_GL_H */

/* Define to 1 if the system has the type `hrtime_t'. */
/* #undef HAVE_HRTIME_T */

/* Define to 1 if you have the <ieeefp.h> header file. */
/* #undef HAVE_IEEEFP_H */

/* Define to 1 if the system has the type `int32_t'. */
#define HAVE_INT32_T 1

/* Define to 1 if the system has the type `int64_t'. */
#define HAVE_INT64_T 1

/* Define to 1 if you have the <inttypes.h> header file. */
#define HAVE_INTTYPES_H 1

/* Define to 1 if you have the `isnan' function. */
#define HAVE_ISNAN 1

/* Define to 1 if you have the math library */
#define HAVE_LIBM 1

/* Define to 1 if you have the <limits.h> header file. */
#define HAVE_LIMITS_H 1

/* Define to 1 if the type `long double' works and has more range or precision
   than `double'. */
#define HAVE_LONG_DOUBLE 1

/* Define to 1 if the type `long double' works and has more range or precision
   than `double'. */
#define HAVE_LONG_DOUBLE_WIDER 1

/* Define to 1 if the system has the type `long long'. */
#define HAVE_LONG_LONG 1

/* Define to 1 if you have the `mach_absolute_time' function. */
/* #undef HAVE_MACH_ABSOLUTE_TIME */

/* Define to 1 if you have the <mach/mach_time.h> header file. */
/* #undef HAVE_MACH_MACH_TIME_H */

/* Define to 1 if your system has a GNU libc compatible `malloc' function, and
   to 0 otherwise. */
#define HAVE_MALLOC 1

/* Define to 1 if you have the <malloc.h> header file. */
#define HAVE_MALLOC_H 1

/* Define to 1 if you have the <math.h> header file. */
#define HAVE_MATH_H 1

/* Define to 1 if you have the `memalign' function. */
#define HAVE_MEMALIGN 1

/* Define to 1 if you have the <memory.h> header file. */
#define HAVE_MEMORY_H 1

/* Define to 1 if you have the `memset' function. */
#define HAVE_MEMSET 1

/* Define to 1 if you have the <MesaGL/glaux.h> header file. */
/* #undef HAVE_MESAGL_GLAUX_H */

/* Define to 1 if you have the <MesaGL/glut.h> header file. */
/* #undef HAVE_MESAGL_GLUT_H */

/* Define to 1 if you have the <MesaGL/glu.h> header file. */
/* #undef HAVE_MESAGL_GLU_H */

/* Define to 1 if you have the <MesaGL/gl.h> header file. */
/* #undef HAVE_MESAGL_GL_H */

/* Define to 1 if you have the `munmap' function. */
#define HAVE_MUNMAP 1

/* Define if your C++ compiler supports namespaces */
#define HAVE_NAMESPACES 1

/* Define to 1 if the system has the type `off64_t'. */
#define HAVE_OFF64_T 1

/* Define to 1 if you have the <OpenGL/glaux.h> header file. */
/* #undef HAVE_OPENGL_GLAUX_H */

/* Define to 1 if you have the <OpenGL/glut.h> header file. */
/* #undef HAVE_OPENGL_GLUT_H */

/* Define to 1 if you have the <OpenGL/glu.h> header file. */
/* #undef HAVE_OPENGL_GLU_H */

/* Define to 1 if you have the <OpenGL/gl.h> header file. */
/* #undef HAVE_OPENGL_GL_H */

/* Have pthread */
#define HAVE_PTHREAD 1

/* Define to 1 if the system has the type `ptrdiff_t'. */
#define HAVE_PTRDIFF_T 1

/* Define to 1 if you have the `putenv' function. */
#define HAVE_PUTENV 1

/* Define to 1 if your system has a GNU libc compatible `realloc' function,
   and to 0 otherwise. */
#define HAVE_REALLOC 1

/* Define to 1 if you have the <setjmp.h> header file. */
#define HAVE_SETJMP_H 1

/* Define to 1 if you have the <signal.h> header file. */
#define HAVE_SIGNAL_H 1

/* Define to 1 if you have the `sincos' function. */
/* #undef HAVE_SINCOS */

/* Define to 1 if you have the `sincosf' function. */
/* #undef HAVE_SINCOSF */

/* Define to 1 if you have the `sinf' function. */
/* #undef HAVE_SINF */

/* Define to 1 if you have the `sqrt' function. */
/* #undef HAVE_SQRT */

/* Define to 1 if the system has the type `ssize_t'. */
#define HAVE_SSIZE_T 1

/* Define to 1 if `stat' has the bug that it succeeds when given the
   zero-length file name argument. */
/* #undef HAVE_STAT_EMPTY_STRING_BUG */

/* Define to 1 if stdbool.h conforms to C99. */
#define HAVE_STDBOOL_H 1

/* Define to 1 if you have the <stdint.h> header file. */
#define HAVE_STDINT_H 1

/* Define to 1 if you have the <stdlib.h> header file. */
#define HAVE_STDLIB_H 1

/* Define to 1 if max is in namespace std:: */
#define HAVE_STD_MAX 1

/* Define to 1 if min is in namespace std:: */
#define HAVE_STD_MIN 1

/* Define to 1 if transform is in namespace std:: */
#define HAVE_STD_TRANSFORM 1

/* Define to 1 if you have the `strchr' function. */
#define HAVE_STRCHR 1

/* Define to 1 if you have the `strftime' function. */
#define HAVE_STRFTIME 1

/* Define to 1 if you have the <strings.h> header file. */
#define HAVE_STRINGS_H 1

/* Define to 1 if you have the <string.h> header file. */
#define HAVE_STRING_H 1

/* Define to 1 if you have the `strstr' function. */
#define HAVE_STRSTR 1

/* Define to 1 if `st_blocks' is a member of `struct stat'. */
#define HAVE_STRUCT_STAT_ST_BLOCKS 1

/* Define to 1 if your `struct stat' has `st_blocks'. Deprecated, use
   `HAVE_STRUCT_STAT_ST_BLOCKS' instead. */
#define HAVE_ST_BLOCKS 1

/* Define to 1 if you have the <sys/ioctl.h> header file. */
#define HAVE_SYS_IOCTL_H 1

/* Define to 1 if you have the <sys/statvfs.h> header file. */
#define HAVE_SYS_STATVFS_H 1

/* Define to 1 if you have the <sys/stat.h> header file. */
#define HAVE_SYS_STAT_H 1

/* Define to 1 if you have the <sys/time.h> header file. */
#define HAVE_SYS_TIME_H 1

/* Define to 1 if you have the <sys/types.h> header file. */
#define HAVE_SYS_TYPES_H 1

/* Define to 1 if the system has the type `uint64_t'. */
#define HAVE_UINT64_T 1

/* Define to 1 if the system has the type `uint_fast64_t'. */
#define HAVE_UINT_FAST64_T 1

/* Define to 1 if you have the <unistd.h> header file. */
#define HAVE_UNISTD_H 1

/* Define to 1 if you have the `vfork' function. */
#define HAVE_VFORK 1

/* Define to 1 if you have the <vfork.h> header file. */
/* #undef HAVE_VFORK_H */

/* Define to 1 if you have the <windows.h> header file. */
/* #undef HAVE_WINDOWS_H */

/* Define to 1 if `fork' works. */
#define HAVE_WORKING_FORK 1

/* Define to 1 if `vfork' works. */
#define HAVE_WORKING_VFORK 1

/* Define to 1 if you have the <xmmintrin.h> header file. */
#define HAVE_XMMINTRIN_H 1

/* Define to 1 if you have the `_aligned_malloc' function. */
/* #undef HAVE__ALIGNED_MALLOC */

/* Define to 1 if you have the `_alloca' function. */
/* #undef HAVE__ALLOCA */

/* Define to 1 if the system has the type `_Bool'. */
/* #undef HAVE__BOOL */

/* Define to 1 if you have the `_exit' function. */
#define HAVE__EXIT 1

/* Define to 1 if the system has the type `_int32'. */
/* #undef HAVE__INT32 */

/* Define to 1 if the system has the type `_int64'. */
/* #undef HAVE__INT64 */

/* Define to 1 if you have the `_isnan' function. */
/* #undef HAVE__ISNAN */

/* Define to 1 if the system has the type `_uint64'. */
/* #undef HAVE__UINT64 */

/* Platform identification used to identify applications for this BOINC core
   client */
#define HOSTTYPE "i686-pc-linux-gnu"

/* Alternate identification used to identify applications for this BOINC core
   client */
/* #undef HOSTTYPEALT */

/* "Define to 1 if largefile support causes missing symbols in C++" */
/* #undef LARGEFILE_BREAKS_CXX */

/* Define to 1 if `lstat' dereferences a symlink specified with a trailing
   slash. */
#define LSTAT_FOLLOWS_SLASHED_SYMLINK 1

/* Define to 1 if `major', `minor', and `makedev' are declared in <mkdev.h>.
   */
/* #undef MAJOR_IN_MKDEV */

/* Define to 1 if `major', `minor', and `makedev' are declared in
   <sysmacros.h>. */
/* #undef MAJOR_IN_SYSMACROS */

/* Define to 1 if your C compiler doesn't accept -c and -o together. */
/* #undef NO_MINUS_C_MINUS_O */

/* Name of package */
#define PACKAGE "xcuda_mb"

/* Define to the address where bug reports for this package should be sent. */
#define PACKAGE_BUGREPORT "http://lunatics.kwsn.net/"

/* Define to the full name of this package. */
#define PACKAGE_NAME "XCuda_MB"

/* Define to the full name and version of this package. */
#define PACKAGE_STRING "XCuda_MB 6.41"

/* Define to the one symbol short name of this package. */
#define PACKAGE_TARNAME "xcuda_mb"

/* Define to the home page for this package. */
#define PACKAGE_URL ""

/* Define to the version of this package. */
#define PACKAGE_VERSION "6.41"

/* Define as directory containing the project config.xml */
#define PROJECTDIR "~boincadm/projects/ap"

/* Define to the necessary symbol if this constant uses a non-standard name on
   your system. */
/* #undef PTHREAD_CREATE_JOINABLE */

/* Define to the BOINC application name for setiathome */
#define SAH_APP_NAME "XCuda_MB"

/* The size of `long double', as computed by sizeof. */
#define SIZEOF_LONG_DOUBLE 12

/* The size of `long int', as computed by sizeof. */
#define SIZEOF_LONG_INT 4

/* Define to 1 if you have the ANSI C header files. */
#define STDC_HEADERS 1

/* Define to be the subversion revision number */
#define SVN_REV "Revision: 1919"

/* Define to 1 if your <sys/time.h> declares `struct tm'. */
/* #undef TM_IN_SYS_TIME */

/* Define to 1 if you want to use 3D-Now optimizations */
/* #undef USE_3DNOW */

/* Define to 1 if you want to use ALTIVEC optimizations */
/* #undef USE_ALTIVEC */

/* Define to 1 to use ASMLIB to determine processor capabilities */
/* #undef USE_ASMLIB */

/* Define to 1 if you want to use the gcc -ffast-math optimization */
#define USE_FAST_MATH 1

/* Define to 1 to use SIMD intrinsics rather than inline assembly */
#define USE_INTRINSICS 1

/* "Define to 1 if you want to use the Intel Performance Primitives" */
/* #undef USE_IPP */

/* Define to 1 if you want to use MMX optimizations */
/* #undef USE_MMX */

/* Define to 1 if you want to use SSE optimizations */
#define USE_SSE 1

/* Define to 1 if you want to use SSE2 optimizations */
/* #undef USE_SSE2 */

/* Define to 1 if you want to use SSE3 optimizations */
/* #undef USE_SSE3 */

/* Version number of package */
#define VERSION "6.41"

/* SETI@home major version number */
#define VERSION_MAJOR 6

/* SETI@home minor version number */
#define VERSION_MINOR 41

/* Define WORDS_BIGENDIAN to 1 if your processor stores words with the most
   significant byte first (like Motorola and SPARC, unlike Intel). */
#if defined AC_APPLE_UNIVERSAL_BUILD
# if defined __BIG_ENDIAN__
#  define WORDS_BIGENDIAN 1
# endif
#else
# ifndef WORDS_BIGENDIAN
/* #  undef WORDS_BIGENDIAN */
# endif
#endif

/* Define to 1 if the X Window System is missing or not being used. */
/* #undef X_DISPLAY_MISSING */

/* Number of bits in a file offset, on hosts where this is settable. */
#define _FILE_OFFSET_BITS 64

/* Define for large files, on AIX-style hosts. */
/* #undef _LARGE_FILES */

/* Define to empty if `const' does not conform to ANSI C. */
/* #undef const */

/* Define to `__inline__' or `__inline' if that's what the C compiler
   calls it, or to nothing if 'inline' is not supported under any name.  */
#ifndef __cplusplus
/* #undef inline */
#endif

/* Define to rpl_malloc if the replacement function should be used. */
/* #undef malloc */

/* Define to `long int' if <sys/types.h> does not define. */
/* #undef off_t */

/* Define to `int' if <sys/types.h> does not define. */
/* #undef pid_t */

/* Define to rpl_realloc if the replacement function should be used. */
/* #undef realloc */

/* Define to `unsigned int' if <sys/types.h> does not define. */
/* #undef size_t */

/* Define as `fork' if `vfork' does not work. */
/* #undef vfork */

 

/* Define USE_NAMESPACES if you may access more than one database from the
 * same program
 */

#endif

#if defined(USE_INFORMIX) && defined(USE_MYSQL) && defined(HAVE_NAMESPACES)
#define USE_NAMESPACES
#endif

#if !defined(CUSTOM_STRING) && defined(COMPILER_STRING)
#define CUSTOM_STRING PACKAGE_STRING" "SVN_REV" "COMPILER_STRING
#endif

#include "std_fixes.h"

#endif

