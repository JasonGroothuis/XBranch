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


#include "sah_config.h"

#include <stdio.h>
#include <iostream>
#include <string.h>
#include <string>
#include <stdlib.h>
#include <math.h>

#ifdef BOINC_APP_GRAPHICS
#include "graphics_api.h"
#ifdef DYNAMIC_GRAPHICS
#include "graphics_lib.h"
#endif
#endif

#include "util.h"
#include "s_util.h"
#include "analyze.h"
#include "gaussfit.h"
#include "seti.h"
#include "chirpfft.h"
#include "analyzeReport.h"
#include "analyzePoT.h"
#ifdef BOINC_APP_GRAPHICS
#include "sah_gfx_main.h"
#endif

#include "../db/schema_master.h"

#if defined(_WIN32) && defined(USE_CUDA)
	#include <cuda_runtime_api.h>
#endif 

extern SPIKE_INFO *tmp_spike;

int FindSpikes(
  float * fp_PowerSpectrum,
  int ul_NumDataPoints,
  int fft_num,
  SETI_WU_INFO& swi
) 
{
  //int i, j, k, m, bin, retval;
  int i, j, k, m, retval, blksize;
  float temp, partial;
  //float total, MeanPower, spike_score;
  float total, MeanPower;
  SPIKE_INFO si;

  i = j = k = m = 0;
  total = 0.0;

  blksize = UNSTDMAX(8, UNSTDMIN(pow2((unsigned int) sqrt((float) (ul_NumDataPoints / 32)) * 32), 512));

  for(int b = 0; b < ul_NumDataPoints/blksize; b++) 
    {
      partial = 0.0f;
      for(i = 0; i < blksize; i++) 
	{
	  partial += fp_PowerSpectrum[b*blksize+i];
	}
      total += partial;
    }
  MeanPower = total / ul_NumDataPoints;
  
//  for(i = 0; i < ul_NumDataPoints; i++) {
//    total += fp_PowerSpectrum[i];
//  }
//  MeanPower = total / ul_NumDataPoints;

  // Here we extract the spikes_to_report highest power events,
  // outputing them as we go.
  // Index usage:
  // i : walk power spectrum us_NumToReport times
  // j : walks power spectrum for each i
  // k : marks current high power candidate while j walks on
  // m : marks the current tail of the high power hit "list"


  for(i = 0; i < swi.analysis_cfg.spikes_per_spectrum; i++) 
    {
      
      temp = 0.0f;
      
      // Walk the array, looking for the first/next highest power.
      // Start j at 1, in order to skip the DC (ie 0) bin.
      // NOTE: this is a simple scan for high powers.  Nice and fast
      // for a very low i.  If i is substantial, this code should be
      // replaced with an index (q)sort.  Do *not* sort the power
      // spectrum itself in place - it's used elsewhere.
      float mval = fp_PowerSpectrum[m];
      for (j = 1; j < ul_NumDataPoints; j++) 
	{
	  float val = fp_PowerSpectrum[j];
	  if (val > temp) 
	    {
	      if (val < mval || m == 0) 
		{
		  temp = val;
		  k = j;
		}
	    }
	} // temp now = first/next highest power and k = it's bin number
      
      m = k; 		// save the "lowest" highest.
      float score = si.s.peak_power / SPIKE_SCORE_HIGH;
      bool sidone = false;
      // if best_spike.s.fft_len == 0, there is not yet a first spike
      if (si.score > best_spike->score || best_spike->s.fft_len == 0) 
	{
	  //  spike info
	  si.s.peak_power 	 = temp/MeanPower;
	  si.s.mean_power	 = 1.0;
	  si.bin 		 = k;
	  si.fft_ind 		 = fft_num;	
	  si.s.chirp_rate 	 = ChirpFftPairs[analysis_state.icfft].ChirpRate;
	  si.s.fft_len    	 = ChirpFftPairs[analysis_state.icfft].FftLen;
	  si.s.freq		 = cnvt_bin_hz(si.bin,  si.s.fft_len);
	  double t_offset=((double)si.fft_ind+0.5)*(double)si.s.fft_len/
	    swi.subband_sample_rate;
	  si.s.detection_freq=calc_detection_freq(si.s.freq,si.s.chirp_rate,t_offset);
	  si.s.time		 = swi.time_recorded + t_offset / 86400.0;
	  time_to_ra_dec(si.s.time, &si.s.ra, &si.s.decl);
	  
	  // Score used for "best of" and graphics.
	  si.score 		 = score;
	  si.score 	  	 = si.score > 0.0f ? (float)log10(si.score) : 0.0f;
	  
	  *best_spike 			= si;
	  sidone = true;
#ifdef BOINC_APP_GRAPHICS
	  if (!nographics()) sah_graphics->si.copy(&si);
#endif
	}
      
      // Report a signal if it excceeds threshold.
      if(si.s.peak_power > (swi.analysis_cfg.spike_thresh)) 
	{
	  if(!sidone)
	    {
	      //  spike info
	      si.s.peak_power 	 = temp/MeanPower;
	      si.s.mean_power	 = 1.0;
	      si.bin 		 = k;
	      si.fft_ind 		 = fft_num;	
	      si.s.chirp_rate 	 = ChirpFftPairs[analysis_state.icfft].ChirpRate;
	      si.s.fft_len    	 = ChirpFftPairs[analysis_state.icfft].FftLen;
	      si.s.freq		 = cnvt_bin_hz(si.bin,  si.s.fft_len);
	      double t_offset=((double)si.fft_ind+0.5)*(double)si.s.fft_len/
		swi.subband_sample_rate;
	      si.s.detection_freq=calc_detection_freq(si.s.freq,si.s.chirp_rate,t_offset);
	      si.s.time		 = swi.time_recorded + t_offset / 86400.0;
	      time_to_ra_dec(si.s.time, &si.s.ra, &si.s.decl);
	      
	      // Score used for "best of" and graphics.
	      si.score 		 = score;
	      si.score 	  	 = si.score > 0.0f ? (float)log10(si.score) : 0.0f;
	      
	      *best_spike 			= si;
	    }
	  retval = result_spike(si);
	  if (retval) SETIERROR(retval,"from result_spike()");
	}
    }
  return 0;
}



int 
FindSpikes2( int            ul_NumDataPoints,
             int            fft_num,
             SETI_WU_INFO&  swi,
             float          total,
             float          temp, // maximum
             int            pos  ) 
{

  // Here we extract the spikes_to_report highest power events,
  // outputing them as we go.
  // Index usage:
  // i : walk power spectrum us_NumToReport times
  // j : walks power spectrum for each i
  // k : marks current high power candidate while j walks on
  // m : marks the current tail of the high power hit "list"
  
  if (swi.analysis_cfg.spikes_per_spectrum > 0) 
    {    
      int        i = 0, 
	j = 0, 
	k = 0, 
	m = 0, 
	retval;
      float      fMeanPower = total / ul_NumDataPoints;
      float      fPeakPower = (temp / fMeanPower); //+0.005f;
      float      fScore = fPeakPower / SPIKE_SCORE_HIGH;
      
      fScore = (fScore > 0.0f) ? (float)log10( fScore ) : 0.0f;
      
      // If we'll need this result, compute pertinent data and place in tmp_spike
      if ( (fScore > best_spike->score) || 
	   (best_spike->s.fft_len == 0) || 
	   (fPeakPower > swi.analysis_cfg.spike_thresh) ) 
	{
	  k = pos;
	  m = k; 		// save the "lowest" highest.
	  
	  tmp_spike->s.peak_power = fPeakPower;
	  tmp_spike->s.mean_power = 1.0;
	  tmp_spike->score        = fScore;
	  tmp_spike->bin 		    = k;
	  tmp_spike->fft_ind 	    = fft_num;
	  
	  tmp_spike->s.chirp_rate  = ChirpFftPairs[analysis_state.icfft].ChirpRate;
	  tmp_spike->s.fft_len     = ChirpFftPairs[analysis_state.icfft].FftLen;
	  tmp_spike->s.freq		 = cnvt_bin_hz( tmp_spike->bin,  tmp_spike->s.fft_len );
          
	  double t_offset = ( (double)tmp_spike->fft_ind + 0.5 ) * 
	    (double)tmp_spike->s.fft_len / swi.subband_sample_rate;
	  tmp_spike->s.detection_freq = calc_detection_freq( tmp_spike->s.freq, tmp_spike->s.chirp_rate, t_offset );
	  tmp_spike->s.time		    = swi.time_recorded + t_offset / 86400.0;
	  
	  time_to_ra_dec( tmp_spike->s.time, &tmp_spike->s.ra, &tmp_spike->s.decl );
        }
      
      // This is the best spike, so copy to best_spike
      if ( (fScore > best_spike->score) || (best_spike->s.fft_len == 0) ) 
	{
	  *best_spike = *tmp_spike;
	  
#ifdef BOINC_APP_GRAPHICS
	  if( !nographics() ) 
	    {
	      sah_graphics->si.copy( tmp_spike, true );
	    }
#endif
        }
      
      // Report a signal if it exceeds threshold
      if( fPeakPower > swi.analysis_cfg.spike_thresh ) 
	{
	  retval = result_spike( *tmp_spike );
	  if( retval ) 
	    {
	      SETIERROR(retval,"from result_spike()");
            }
        }
    }

  return 0;
}
