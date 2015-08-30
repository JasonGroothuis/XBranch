				 Seti@home Enhanced Cuda Multibeam V7
				 Windows Platform, 32 bit executables
				   	README_x41zc.txt
				Revised for Public Release May 31st, 2013

	Applies to Berkeley distributed setiathome_7.00_windows_intelx86__cudaxx.exe, and
		Third party distributed Lunatics_x41zc_win32_cudaxx.exe

  Contents:
	1) x41zc general notes
	2) Manual installation under BOINC anonymous platform (advanced users***)
	3) Optional configuration (advanced users***, also see (a) example configuration)
	4) Known issues
	5) Release history
	6) Obtaining source code
	7) Support
	8) Legal
	a) Example Configuration
------------------------------------------------------------------------------------------------------------------
*** 'Advanced User', for the purposes of this document and the application it describes, refers to a User familiar 
with manually configuring and operating the BOINC Client to run under the 'Anonymous platform' mechanism, and/or, 
that understands the purpose of custom configuration options, as applicable.
------------------------------------------------------------------------------------------------------------------
1) x41zc general notes
	- This document is for the Windows versions of x41zc.
	- You may have obtained this Cuda application either as a standalone manual install package 
	(intended for advanced users***), as part of a Lunatics Unified Installer, or as a stock application 
	automatic download from the seti@home project. 
	- This application is intended to process seti@home multibeam V7 tasks, on Cuda enabled nVidia Graphics 
	Processing Units (GPUs)
	- This application is currently available in 5 'flavours',
                - Cuda 2.2 (min driver 185.85* ), exclusively for hosts that cannot upgrade to later Cuda version 
                 drivers. NOT to be used on Fermi/Kepler!
                - Cuda 2.3 (min driver 190.38* ), to be distributed as 'stock' for widest support of Video drivers,
		 and recommended for use on older GPUs prior to Fermi & Kepler, such as 8xxx series, 9xxx series,
		 GTX 200 series. Usually the best choice on Pre-Fermi. NOT to be used on Fermi/Kepler!
		- Cuda 3.2 (min driver 263.06* ), also to be distributed as 'stock' for widest support of Video
                 drivers, and recommended for use on older GPUs prior to Fermi & Kepler, such as 8xxx series, 9xxx 
                 series, GTX 200 series.
		- Cuda 4.2 (min driver 301.48* ), recommended for Fermi class GPUs for a significant 
 		 performance increase.
                - Cuda 5.0 (min driver 306.23* ), recommended for Kepler class GPUs for an additional performance 
                 increase.
	    *In some cases there may be earlier Beta, day0 or day1, drivers available for the given Cuda revision or
	     specific GPU release.
	- each executable uses the 32 bit CUDA runtime (cudart32_xx_xx.dll), and CUFFT libraries (cufft32_xx_xx.dll),
          [Cuda 3.2, 4.2 and 5.0] or cudart.dll and cufft.dll [Cuda 2.2 and 2.3]	
------------------------------------------------------------------------------------------------------------------
2) Manual installation under BOINC anonymous platform (advanced users***)
	- For standalone (non-installer) installation, the package archives contain example '.aistub' files
	which are text files containing template information intended for advanced user*** manual installation.
	- These .aistub files can either be used as is and merged into a preexisting installer based optimised 
	application installation using the installer supplied aimerge command,  tailored before merge, or 
	copy-pasted into an existing app_info.xml file making the usual configuration adjustments manually.
------------------------------------------------------------------------------------------------------------------
3) Optional configuration (advanced users***, also see (a) example configuration)
	Adjusting the host process priority from the default 'below normal' can help GPU utilisation & reduce CPU 
	usage for some systems.  You can either use an external tool, such as eFMer (Fred's) Priority v1.2, 
	Process Lasso (or similar), or alternatively use the internal priority control via .cfg text file, or
	app_info command line parameter under anonymous platform operation.
	- For process priority control via mbcuda.cfg in the project directory, a supplied mbcuda.cfg file with
	notes and commented out default settings is supplied.  The use of these settings is documented with 
	comments in that file.
	- For Berkeley stock distribution, these settings can be made in the version specific 
	  mbcuda-7.00-cudaxx.cfg file, which is distrib5uted as zero length, A sample is supplied in
		mbcuda-7.00-cudaxx.cfg.sample , These are plain text files that can be edited with notepad.
	- For individual control of host application process priorities on a per GPU basis, also in the supplied
	mbcuda.cfg text file there is an example per GPU process priority line. NOTE that x41zc changes the method
        used for controlling individual GPU process priorities, to use the same keys as the global entries in 
	optional GPU specific sections.  Individual PCI bus & slot identifiers can be retrieved from stderr text 
	output displayed on task result pages on the seti@home site
	- For command-line control (practical under anonymous platform operation), only general process priority
	control is supported, by adding a <cmdline>-ppp</cmdline> line into the appropriate app_info.xml section(s)
	, where -ppp can be:
             	-bel (for the default of below normal process priority)
		-nor (for normal process priority)
		-abo (for above normal process priority)
		-hig (for high process priority)
	- only the first 3 letters are needed
	- 2 new performance / usability related parameters are provided via the command line, and mbcuda.cfg
		-pfb nn  , command line option, nn may be 1 to 16,  default for PreFermis is 1, Fermis or Newer is 4
		-pfp nn  , nn may be 1 to 1000, default is 100 as per original 6.09 sources.
		equivalent keys in mbcuda.cfg are described there with examples
	- matching pci device specific setting (in mbcuda.cfg) takes the highest precedence, followed by cfg file
	process-priority key for global control, and lastly command line. 
	(more specific configs override less specific ones when multiple methods specify different priorities)
------------------------------------------------------------------------------------------------------------------
4) Known issues
	- Processing order induced overflow result mismatches, originate from original Cuda app designs, may be
	addressed in the future either by server side adjustment, and/or redesign in future builds.  Technically
	& scientifically the result signals are 'correct', though a different subset of those contained in the 
	task. 
	- reduced 'chirp noise' (against 6.08, 6.09, 6.10), original cuda application chirp accuracy was found to be 
	limited compared to CPU applications.  This accuracy is improved in this application, so there may be some
	signal variation against old cuda applications (stock 6.08, 6.09, 6.10 and third party optimised forms)
	- lack of algorithmic noise around threshold (from 6.03), some signal normalisation techniques in stock 
	6.03 applications were not algorithmically correct with respect to maintaining precision, therefore 
	sometimes signals close to threshold may not match, or one result may have more or less signals of any 
	type than others.  AKv8 optimised CPU applications are not known to suffer this 'algorithmic noise' and
	setiathome_enhanced V7 is planned to correct this noise for stock CPU applications
	- an obscure race condition in the pulsefinding, that can result in the PoT array looking 'wrong' under 
	certain conditions (originating from 6.08, 6.09 & 6.10), this is rare, discovered only recently and can 
	occur where there are closely spaced pulses.  It's primary effect is a cosmetic one, resulting in the 
	wrong pulsepot array data submitted to the server, which is expected to be displayed on NTPCKER online
	pages.  This will be addressed in a future build.
	-There are questionable optimisations in reference 6.98/6.99 and Cuda Gaussian fitting code, 
	which can lead to inconclusives.
	Tech Notes: 
	  -Gaussian noscore logic, Commented in code as being targeted at GPUs, though likely only effective to some
		degree on CPU.  On Cuda devices this added logic/complexity is not a net win, 
		due to those Kernels being memory bound (score calculations would have been hidden by memory accesses),
		while the conditional logic is expensive.  Cuda GaussFits process on a powerspectrum wide basis, though
		reduce reportables on a PoT basis like CPU.  This picture might change in the future with redesigned 
		result reductions, along with introduction of dynamic parallelism capabilities into the hardware 
		& runtime/drivers.
	  -Gaussian score precision, nVidia's texture cached score acceleration provides performance benefits at
		cost of score precision.  At the time of 6.08,6.09,6.10 few Cuda devices on the market had double 
		precision, and the texture cache LUT implementation uses single precision with linear interpolation.
		This approach is expected to yield occasional inconclusives, and is recommended for redesign in
		the future.  This may involve use of emulated double (df64 format, as with the chirp) and or true double
		precision where feasible.
------------------------------------------------------------------------------------------------------------------
5) Release history (Windows Platform)
x41zc (second public beta, updated for first public release)
	- Fixed a typo in the GPU side partial autocorrelation reductions, that would have caused skipping of
	  peaks in the first non-zero delay bin.
	- Deactivated VRAM memory map printing.
x41zb (first public beta)
        - mbcuda.cfg distributed as zero length file, to play nicely with Boinc allowing user configuration, sample
          configuration now supplied in readme.
	- Syncronisation changes in drivers/compilers for Kepler (and possibly Fermi class to a lesser extent)
          require additional pulsefinding kernel syncronisation, esp with Cuda 3.2. (PTX JIT compialtion was snuffing
          out implicit syncronisation that was necessary)
x41z (early beta)
        - increased configurability to include some performance & usability control
	- some minor boincapi customisations affecting performance on all Cuda GPUs
	- intended for supporting the full range of Cuda enabled GPUs, providing support for several Cuda revisions
	 , including older variants for systems that cannot install newer drivers 
		(e.g. certain Alienware gaming laptops), or encounter problems running newer driver architectures,
		, such as GTX 200 series and earlier, reduced performance under Cuda 3.0 and above.
x41x (Prior closed beta) 
	- performance enhancements,
	- V7 readiness to beta 6.98 spec,
	- accuracy and triplet search improvement
	- Fermi & Kepler GPU performance scaling
x41g 
	- improved reliability
	- preliminary GPU cross-generation computation precision matching
x38g
        - improved cross GPU generation precision match, and preliminary optimisation experiments 
x32f 	
	- Initial release based on setiathome_enhanced 6.09 sources,
	- Fermi GPU compatibility
------------------------------------------------------------------------------------------------------------------
6) Obtaining source code
Source code is available in several ways
- via the Berkeley svn repository, 
  https://setisvn.ssl.berkeley.edu/trac/browser/branches/sah_v7_opt/Xbranch  
- By direct request to where you obtained this application, if a website will supply downloads there
- If obtained in a Lunatics installer, via download from http://lunatics.kwsn.net
------------------------------------------------------------------------------------------------------------------
7) Support
- No specific support is prescribed, and please see all accompanying documentation for warnings & disclaimers etc.
- technical help can usually be obtained from seti@home Q&A and number crunching fora, as well as from Lunatics
site, from the GPU Users Group website, or Arkayn's site 'Crunchers Anonymous' http://www.arkayn.us/forum/index.php
------------------------------------------------------------------------------------------------------------------
8) Legal
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, specifically GNU General Public License, Version 2 or later.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.

Copyright (C) 2013 Jason Groothuis. All files, programs and similar contained herein are
distributed under the terms of the General Public License.
------------------------------------------------------------------------------------------------------------------
a) Example Configuration
;;; This configuration file is for optional control of Cuda Multibeam x41zc
;;; Currently, the available options are for
;;; application process priority control (without external tools), and
;;; per gpu priority control (useful for multiple Cuda GPU systems)
[mbcuda]
;;;;; Global applications settings, to apply to all Cuda devices
;;; You can uncomment the processpriority line below, by removing the ';', to engage machine global priority control of x41x
;;; possible options are 'belownormal' (which is the default), 'normal', 'abovenormal', or 'high'
;;; For dedicated crunching machines, 'abovenormal' is recommended
;;; raising global application priorities above the default
;;;   may have system dependant usability effects, and can have positive or negative effects on overall throughput 
;processpriority = abovenormal
;;; Pulsefinding: Advanced options for long pulsefinds (affect display usability & long kernel runs)
;;; defaults are conservative.
;;; WARNING: Excessive values may induce display lag, driver timeout & recovery, or errors. 
;;; pulsefinding blocks per multiprocessor (1-16), default is 1 for Pre-Fermi, 4 for Fermi or newer GPUs
;pfblockspersm = 8
;;; pulsefinding maximum periods per kernel launch  (1-1000), default is 100, as per 6.09
;pfperiodsperlaunch = 200

;[bus1slot0]
;;; Optional GPU specifc control (requires Cuda 3.2 or newer app), example
;processpriority = abovenormal
;pfblockspersm = 8
;pfperiodsperlaunch = 200
------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------
