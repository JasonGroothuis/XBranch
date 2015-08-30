/*
 * validrun.c
 *
 * Functions for determining if a section of the tape buffer can be
 * turned into a valid work unit
 *
 * $Id: mb_validrun.cpp,v 1.1.2.1 2006/12/14 22:24:45 korpela Exp $
 *
 */

#include "sah_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include "setilib.h"
#include "splitparms.h"
#include "splittypes.h"
#include "splitter.h"
#include "message.h"


bool valid_run(std::vector<dr2_compact_block_t> &tapebuffer) {
  unsigned long start_dataseq=tapebuffer[0].header.dataseq;
  unsigned long end_dataseq=tapebuffer[tapebuffer.size()-1].header.dataseq;
  bool valid=((end_dataseq-start_dataseq)==(tapebuffer.size()-1));
  int i;
  if (!valid) {
    log_messages.printf(SCHED_MSG_LOG::MSG_NORMAL,"Missing frames between %lu and %lu\n",tapebuffer[0].header.dataseq,tapebuffer[tapebuffer.size()-1].header.dataseq);
    // find the last "in sequence" frame.
    for (i=tapebuffer.size()-1;i>0;i--) {
      if (tapebuffer[i-1].header.dataseq != (tapebuffer[i].header.dataseq-1)) {
        // delete the frames prior to that one.
	tapebuffer.erase(tapebuffer.begin(),tapebuffer.begin()+i);
      }
    }
  }
  if (valid) {
    for (i=0;valid && (i<tapebuffer.size());i++) {
      if (tapebuffer[i].header.sky_freq != tapebuffer[0].header.sky_freq) {
        // if the frequency changed
        log_messages.printf(SCHED_MSG_LOG::MSG_NORMAL,"Frequency changed from %f to %f\n",tapebuffer[i].header.sky_freq,tapebuffer[0].header.sky_freq);
        tapebuffer.erase(tapebuffer.begin(),tapebuffer.begin()+i);
        valid=false;
      }
    }
  }
 
  return(valid);
}


/*
 * $Log: mb_validrun.cpp,v $
 * Revision 1.1.2.1  2006/12/14 22:24:45  korpela
 * *** empty log message ***
 *
 * Revision 1.2.4.1  2006/01/13 00:37:58  korpela
 * Moved splitter to using standard BOINC logging mechanisms.  All stderr now
 * goes to "error.log"
 *
 * Added command line parameters "-iterations=" (number of workunit groups to
 * create before exiting), "-trigger_file_path=" (path to the splitter_stop trigger
 * file.  Default is /disks/setifiler1/wutape/tapedir/splitter_stop).
 *
 * Reduced deadlines by a factor of three.  We now need a 30 MFLOP machine to meet
 * the deadline.
 *
 * Revision 1.2  2003/09/11 18:53:38  korpela
 * *** empty log message ***
 *
 * Revision 1.1  2003/07/29 20:35:57  korpela
 *
 * renames .C files to .cpp
 *
 * Revision 1.1  2003/06/03 00:23:43  korpela
 *
 * Again
 *
 * Revision 3.0  2001/08/01 19:04:57  korpela
 * Check this in before Paul screws it up.
 *
 * Revision 2.7  2000/12/01 01:13:29  korpela
 * *** empty log message ***
 *
 * Revision 2.6  1999/06/07 21:00:52  korpela
 * *** empty log message ***
 *
 * Revision 2.5  1999/03/27  16:19:35  korpela
 * *** empty log message ***
 *
 * Revision 2.4  1999/02/22  22:21:09  korpela
 * added -nodb option
 *
 * Revision 2.3  1999/02/11  16:46:28  korpela
 * Added checkpointing.
 *
 * Revision 2.2  1998/11/04  23:08:25  korpela
 * Byte and bit order change.
 *
 * Revision 2.1  1998/11/02  16:41:21  korpela
 * Minor Change.
 *
 * Revision 2.0  1998/10/30  22:00:04  korpela
 * Conversion to C++ and merger with client source tree.
 *
 * Revision 1.3  1998/10/30  20:26:03  korpela
 * Bug Fixes.  Now mostly working.
 *
 * Revision 1.2  1998/10/27  00:59:22  korpela
 * Bug fixes.
 *
 * Revision 1.1  1998/10/22  17:48:20  korpela
 * Initial revision
 *
 *
 */
