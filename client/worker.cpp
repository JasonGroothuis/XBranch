// Copyright 2003 Regents of the University of California

// SETI_BOINC is free software; you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2, or (at your option) any later
// version.

// SETI_BOINC is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
// more details.

// You should have received a copy of the GNU General Public License along
// with SETI_BOINC; see the file COPYING.  If not, write to the Free Software
// Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.

// In addition, as a special exception, the Regents of the University of
// California give permission to link the code of this program with libraries
// that provide specific optimized fast Fourier transform (FFT) functions and
// distribute a linked executable.  You must obey the GNU General Public 
// License in all respects for all of the code used other than the FFT library
// itself.  Any modification required to support these libraries must be
// distributed in source code form.  If you modify this file, you may extend 
// this exception to your version of the file, but you are not obligated to 
// do so. If you do not wish to do so, delete this exception statement from 
// your version.

// worker.C: SETI-independent worker logic
// $Id: worker.cpp,v 1.32.2.14 2007/08/10 00:38:49 korpela Exp $
//
#include "sah_config.h"


#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif
#include <errno.h>
#ifndef _WIN32
#include <dlfcn.h>
#endif

#ifdef BOINC_APP_GRAPHICS
#include "sah_gfx_main.h"
#endif
#include "s_util.h"
#include "seti_header.h"
#include "seti.h"
#include "worker.h"
#include "chirpfft.h"
#include "analyzeFuncs.h"
#include "analyzeReport.h"

#include "filesys.h"
#include "boinc_api.h"

#include "worker.h"

#if 1
	#include "cuda/cudaAcceleration.h"
#endif

#if defined(_WIN32) && defined(USE_CUDA)
	#include <cuda_runtime_api.h>
#endif 

using std::string;

bool verbose;
bool bNoCUDA;
bool bLogPerf2StdErr;
bool bPollCUDA;
int  gCUDADevPref;

// this gets called first on all platforms
int common_init() {

#ifdef ALLOW_CFFT_FILE
  FILE* f;
  f = boinc_fopen(CFFT_FILENAME, "r");
  if (f) {
    cfft_file = &debug_cfft_file[0];
    fclose(f);
  }
#endif


  return 0;
}

// Do this when start on a work unit for the first time.
// Creates result header file.
static int initialize_for_wu() {
  int retval = 0;
  FILE* f;
  string path;

  boinc_resolve_filename_s(OUTFILE_FILENAME, path);
  f = boinc_fopen(path.c_str(), "wb");
  if (!f) SETIERROR(FOPEN_FAILED,"in initialize_for_wu()");
  xml_indent(2);
  fprintf(f, "<result>\n");
  retval = seti_write_wu_header(f, 1);
  fclose(f);

  return retval;
}

// parse input and state files
//
static int read_wu_state() {
    FILE* f;
    int retval=0;
    string path;
    FORCE_FRAME_POINTER;

    boinc_resolve_filename_s(WU_FILENAME, path);
    f = boinc_fopen(path.c_str(), "rb");
    if (f) {
#ifdef BOINC_APP_GRAPHICS
    if (!nographics())  sprintf(sah_graphics->status, "Scanning data file\n");
#endif
        retval = seti_parse_wu(f, analysis_state);
        fclose(f);
        if (retval) SETIERROR(retval,"from seti_parse_wu() in read_wu_state()");
    } else {
	char msg[1024];
	sprintf(msg,"(%s) in read_wu_state() errno=%d\n",path.c_str(),errno);
	SETIERROR(FOPEN_FAILED,msg);
    }

    retval = seti_init_state();
    if (retval) SETIERROR(retval,"from seti_init_state() in read_wu_state()");

#ifdef BOINC_APP_GRAPHICS
    if (!nographics()) sprintf(sah_graphics->status, "Scanning state file.\n");
#endif
    if (boinc_file_exists(STATE_FILENAME)) {
      retval = parse_state_file(analysis_state);
    } else {
      retval = initialize_for_wu();
      if (retval) SETIERROR(retval,"from initialize_for_wu() in read_wu_state()");
    }

    boinc_fraction_done(progress*remaining+(1.0-remaining)*(1.0-remaining));
    return 0;
}

void worker() {
  int retval=0;
  run_stage=POSTINIT;
  FORCE_FRAME_POINTER;
#if defined(__GNUC__) && defined (__i386__)
  __asm__ __volatile__ ("andl $-16, %esp");
#endif
#if defined(_WIN32) && defined(CUDA22)
// Cuda 2.2 native driver on Windows is not multithreaded, returns an error during ExitProcess()
// ... but TerminateProcess() is OK
  worker_use_rude_exit = true; //Boincapi customisation
// Cuda 2.3 and newer use ExitProcess() instead, to avoid corrupting the DLL global states,
// and terminating/crashing driver helper threads.
#endif
  try {
    retval = common_init();
    if (retval) SETIERROR(retval,"from common_init() in worker()");

    retval = read_wu_state();
    if (retval) SETIERROR(retval,"from read_wu_state() in worker()");
    
    retval = seti_do_work();
    if (retval) SETIERROR(retval,"from seti_do_work() in worker()");

    //boinc_fpops_cumulative((SETUP_FLOPS+analysis_state.FLOP_counter)*LOAD_STORE_ADJUSTMENT);
#if defined(_WIN32)
  fprintf(stderr,"Worker preemptively acknowledging a normal exit.->\n");
  worker_thread_exit_ack = true;
#endif
        boinc_finish(0);
  }
  catch (seti_error e) {
    if (e == RESULT_OVERFLOW) {
        fprintf(stderr, "SETI@Home Informational message -9 result_overflow\n");
        fprintf(stderr, "NOTE: The number of results detected equals the storage space allocated.\n");
        final_report(); // add flop and signal counts
	progress=1;
	remaining=0;
	boinc_fraction_done(progress);
        checkpoint(true);      // force a checkpoint
        //boinc_fpops_cumulative((SETUP_FLOPS+analysis_state.FLOP_counter)*LOAD_STORE_ADJUSTMENT);
#if defined(_WIN32)
  fprintf(stderr,"Worker preemptively acknowledging an overflow exit.->\n");
  worker_thread_exit_ack = true;
#endif
        boinc_finish(0);
    } else {
        e.print();
//  	fprintf(stderr,"Calling a hard exit(%d) due to an exception\n",static_cast<int>(e));
#ifdef _WIN32
    	ExitProcess(static_cast<int>(e));
#endif
    }
  }
}

