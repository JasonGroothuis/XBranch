#include "sah_config.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>

#include "parse.h"
#include "boinc_db.h"
#include "error_numbers.h"

#include "sah_result.h"
#include "sah_boinc_db.h"

//#include "boinc_db.h"
#include "util.h"
#include "sched_config.h"
#include "sched_util.h"
#include "sched_msgs.h"

struct VALIDATE_STATS {
    int nstrong_compare;
    int nstrong;
    int nweak_compare;
    int nweak;
    int bad_results;

    void print();
};

int overflow_result(RESULT& r);

void VALIDATE_STATS::print() {
    printf("Strongly similar: %d out of %d\n", nstrong, nstrong_compare);
    printf("Weakly similar: %d out of %d\n", nweak, nweak_compare);
}

VALIDATE_STATS validate_stats;

// check_set() is called from BOINC code and is passed a vector of all
// received results for work unit.  check_set() determines the canonical
// result and flags each result as to whether it is similar enough to the 
// canonical result to be given credit.  check_set provides BOINC with both 
// the canonical ID and the amount of credit to be granted to each validated 
// result.  As a matter of policy the validator does not do values checking.
// The canonical result could have bad values.  The detection and flagging
// of this situation is a function of the assimilator.
int check_set(
    vector<RESULT>& results, WORKUNIT& wu, int& canonicalid, double& granted_credit, bool& retry) {

    // Note that SAH_RESULT is not the same type as the standard sah
    // result as it appears in the app and the science backend.  Rather
    // it simply contains a vector of all the signals returned in a 
    // a given result, along with functions used to validate that result
    // (ie that set of signals). 
    // I should rename the type.  jeffc
    vector<SAH_RESULT> sah_results;
    //SAH_RESULT s;
    RESULT r;
    DB_RESULT db_result;

    unsigned int i, j, k, good_result_count=0;
    bool found, err_opendir=false;
    double max_credit, min_credit, sum;
    int max_credit_i=-1, min_credit_i=-1, nvalid, retval, result_flags;
    vector<bool> bad_result;

    retry=false;  // init

    log_messages.printf(
         SCHED_MSG_LOG::MSG_DEBUG,

         "check_set() checking %d results\n",
         results.size()
    );

    // read and parse the result files
    //
    for (i=0; i<results.size(); i++) {
	SAH_RESULT s = {0};
	log_messages.printf(
                SCHED_MSG_LOG::MSG_DEBUG,
                "[RESULT#%d] getting result file %s\n",
                results[i].id, results[i].name
        );
        retval = get_result_file(results[i], s);
	if (retval) {
		log_messages.printf(
                	SCHED_MSG_LOG::MSG_DEBUG,
                	"[RESULT#%d] read/parse of %s FAILED with retval %d\n",
			results[i].id, results[i].name, retval
		);
		// A directory problem may be transient. 
		if (retval == ERR_OPENDIR) {
			retry = true;
		} else {
			// a non-transient, non-recoverable error
			results[i].outcome =  RESULT_OUTCOME_VALIDATE_ERROR;
                        results[i].validate_state = VALIDATE_STATE_INVALID;
                        
                        //db_result = results[i];
			//retval = db_result.update();
            		//if (retval) goto return_retval;
		}
                retval = 0;
	} else {
		good_result_count++;
	}
        sah_results.push_back(s);    // s may be a null result in case of IO error
    }  

    // If IO errors took us below min_quorum, bail.
    if (good_result_count < wu.min_quorum) {
        log_messages.printf(
                SCHED_MSG_LOG::MSG_DEBUG,
          "[WU#%d] IO error(s) led to less than quorum results.  Will retry WU upon receiving more results.\n",
                wu.id
        );
	canonicalid = 0;
	retval = 0;
	goto return_retval;
    }

    // flag results with bad values
    for (i=0; i<sah_results.size(); i++) {
	if (!sah_results[i].have_result) continue;
	if (sah_results[i].bad_values()) {
		log_messages.printf(
                	SCHED_MSG_LOG::MSG_DEBUG,
                	"[RESULT#%d] has one or more bad values.\n",
                	results[i].id
        	);
		bad_result.push_back(true);
		validate_stats.bad_results++;
	}
    } 

    // see if there's a pair of results that are strongly similar
    // Not all results are *neccessarily* checked for overflow.  Any
    // result that may become the canonical reult is checked and thus
    // a valid overflow indicator is always paased on to the assimilator.
    found = false;
    for (i=0; i<sah_results.size()-1; i++) {
	if (!sah_results[i].have_result) continue;
	if (overflow_result(results[i])) {
		result_flags = (int)results[i].opaque;
		result_flags |= RESULT_FLAG_OVERFLOW;
		results[i].opaque = (double)result_flags;
		log_messages.printf(
                        SCHED_MSG_LOG::MSG_DEBUG,
                       	"[RESULT#%d] is an OVERFLOW result\n",
                        results[i].id
                );
	}
        for (j=i+1; j<sah_results.size(); j++) {
	    if (!sah_results[j].have_result) continue;
	    if (overflow_result(results[j])) {
			result_flags = (int)results[j].opaque;
			result_flags |= RESULT_FLAG_OVERFLOW;
			results[j].opaque = (double)result_flags;
	     		log_messages.printf(
                        	SCHED_MSG_LOG::MSG_DEBUG,
                        	"[RESULT#%d] is an OVERFLOW result\n",
                        	results[j].id
                	);
	    }
            validate_stats.nstrong_compare++;
            if (sah_results[i].strongly_similar(sah_results[j])) {
		log_messages.printf(
			SCHED_MSG_LOG::MSG_DEBUG,
                	"[RESULT#%d (%d signals) and RESULT#%d (%d signals)] ARE strongly similar\n", 
			results[i].id, sah_results[i].num_signals, results[j].id, sah_results[j].num_signals
                );
                found = true;
                validate_stats.nstrong++;
                break;
            }
	    log_messages.printf(
		SCHED_MSG_LOG::MSG_DEBUG,
                	"[RESULT#%d (%d signals) and RESULT#%d (%d signals)] are NOT strongly similar\n", 
			results[i].id, sah_results[i].num_signals, results[j].id, sah_results[j].num_signals
            );
        }
        if (found) break;
    }

    if (found) {
	// At this point results[i] is the canonical result and results[j] 
	// is strongly similar to results[i].
        canonicalid = results[i].id;
        max_credit = 0;
        min_credit = 0;
        nvalid = 0;
        for (k=0; k<sah_results.size(); k++) {
	    if (!sah_results[k].have_result) continue;
            if (k == i || k == j) {
                results[k].validate_state = VALIDATE_STATE_VALID;
            } else {
                validate_stats.nweak_compare++;
                if (sah_results[k].weakly_similar(sah_results[i])) {
                    validate_stats.nweak++;
		    log_messages.printf(
                	SCHED_MSG_LOG::MSG_DEBUG,
                	"[CANONICAL RESULT#%d (%d signals) and RESULT#%d (%d signals)] ARE weakly similar\n",
                	results[i].id, sah_results[i].num_signals, results[k].id, sah_results[k].num_signals
        	    );
                    results[k].validate_state = VALIDATE_STATE_VALID;
                } else {
		    log_messages.printf(
                	SCHED_MSG_LOG::MSG_DEBUG,
                	"[CANONICAL RESULT#%d (%d signals) and RESULT#%d (%d signals)] are NOT weakly similar\n",
                	results[i].id, sah_results[i].num_signals, results[k].id, sah_results[k].num_signals
        	    );
                    results[k].validate_state = VALIDATE_STATE_INVALID;
                }
            }

            if (results[k].validate_state == VALIDATE_STATE_VALID) {
                if (results[k].claimed_credit < 0) {
                    results[k].claimed_credit = 0;
                }
                if (nvalid == 0) {
                    max_credit = min_credit = results[k].claimed_credit;
		    max_credit_i = min_credit_i = k;
                } else {
                    if (results[k].claimed_credit >= max_credit) {
                        	max_credit = results[k].claimed_credit;
                        	max_credit_i = k;
                    }
                    if (results[k].claimed_credit <= min_credit) {
                        	min_credit = results[k].claimed_credit;
                        	min_credit_i = k;
                    }
                }
                nvalid++;
            }
        }

        // the granted credit is the average of claimed credits
        // of valid results, discarding the largest and smallest
        //
        if (nvalid == 2) {
            granted_credit = min_credit;
        } else {
	    // Take care of case where all claimed credits are equal.
	    if (max_credit == min_credit) {
		granted_credit = min_credit;
	   } else {
           	sum = 0;
            	for (k=0; k<results.size(); k++) {
	    		if (!sah_results[k].have_result) continue;
               	 	if (results[k].validate_state != VALIDATE_STATE_VALID) continue;
               		if (k == max_credit_i) continue;
               	 	if (k == min_credit_i) continue;
               	 	sum += results[k].claimed_credit;
            	}
            	granted_credit = sum/(nvalid-2);
	   }
        }
        for (k=0; k<results.size(); k++) {
            if (results[k].validate_state == VALIDATE_STATE_VALID) {
                results[k].granted_credit = granted_credit;
            }
        }
    } else {
        canonicalid = 0;
    }

    retval = 0;
return_retval:
    return retval;
}

// check_pair() is called by BOINC code to validate any results arriving
// after the canonical result has been chosen.  
int check_pair(RESULT& new_result, RESULT& canonical, bool& retry) {
    int retval;
    SAH_RESULT sah_new, sah_canonical;
    DB_RESULT db_result;

    retry=false;  // init

    log_messages.printf(
            SCHED_MSG_LOG::MSG_DEBUG,
            "[RESULT#%d] getting new result file %s\n",
            new_result.id, new_result.name
    );
    retval = get_result_file((RESULT&)new_result, sah_new);
    if (retval) {
            log_messages.printf(
                    SCHED_MSG_LOG::MSG_DEBUG,
                    "[RESULT#%d] read/parse of %s FAILED with retval %d\n",
                    new_result.id, new_result.name, retval
            );
            // A directory problem may be transient. 
            if (retval == ERR_OPENDIR) {
                    retry = true;
		    retval = 0;
		    goto return_retval;
            } else {
                    // a non-transient, non-recoverable error
                    new_result.outcome =  RESULT_OUTCOME_VALIDATE_ERROR;
                    new_result.validate_state = VALIDATE_STATE_INVALID;	
                    //db_result = new_result;
                    //retval = db_result.update();
                    //if (retval) goto return_retval;
		    retval = 0; 
                    goto return_retval;
            }
    } 

    log_messages.printf(
            SCHED_MSG_LOG::MSG_DEBUG,
            "[RESULT#%d] getting canonical result file %s\n",
            canonical.id, canonical.name
    );
    retval = get_result_file((RESULT &)canonical, sah_canonical);
    if (retval) {
            log_messages.printf(
                    SCHED_MSG_LOG::MSG_DEBUG,
                    "[RESULT#%d] read/parse of %s FAILED with retval %d\n",
                    canonical.id, canonical.name, retval
            );
            // A directory problem may be transient. 
            if (retval == ERR_OPENDIR) {
                    retry = true;
		    retval = 0;
		    goto return_retval;
            } else {
                    // a non-transient, non-recoverable error - set new_result to error.
                    new_result.outcome =  RESULT_OUTCOME_VALIDATE_ERROR;
		    new_result.validate_state = VALIDATE_STATE_INVALID; 
                    retval = 0; 
                    //db_result = new_result;
                    //retval = db_result.update();
                    //if (retval) goto return_retval;
		    goto return_retval;
            }
    } 

    if (sah_canonical.weakly_similar(sah_new)) {
    	log_messages.printf(
                SCHED_MSG_LOG::MSG_DEBUG,
                "[CANONICAL RESULT#%d (%d signals) and NEW RESULT#%d (%d signals)] ARE weakly similar\n",
		canonical.id, sah_canonical.num_signals, new_result.id, sah_new.num_signals 
    	);
	new_result.validate_state = VALIDATE_STATE_VALID;
    } else {
    	log_messages.printf(
                SCHED_MSG_LOG::MSG_DEBUG,
                "[CANONICAL RESULT#%d (%d signals) and NEW RESULT#%d (%d signals)] are NOT weakly similar\n",
		canonical.id, sah_canonical.num_signals, new_result.id, sah_new.num_signals 
    	);
	new_result.validate_state = VALIDATE_STATE_INVALID;
    }

    retval = 0;
return_retval:
    return retval;
}

int overflow_result(RESULT& r) {
   
	if (strstr(r.stderr_out, "result_overflow")) {
       	return 1;
   	} else {
       	return 0;
   	}
}


#if 0
// test program:
// sah_validate file1 ... filen
//
int main(int argc, char** argv) {
    vector<RESULT> results;
    double credit, test_credit=5.5;
    int i, canonical, retval;
    RESULT r;

    memset(&validate_stats, 0, sizeof(validate_stats));

    for (i=1; i<argc; i++) {
        memset(&r, 0, sizeof(r));
        r.claimed_credit = (test_credit += 3);
        r.id = i;
        sprintf(r.xml_doc_out,
            "<file_info>\n"
            "    <name>%s</name>\n"
            "</file_info>\n",
            argv[i]
        );
        results.push_back(r);
    }

    retval = check_set(results, canonical, credit);
    if (retval) {
        printf("error: %d\n", retval);
    }

    for (i=0; i < results.size(); i++) {
	printf("validate_state of %s is %d\n", argv[i+1], results[i].validate_state);
    }

    if (canonical) {
        printf("canonical result is %d; credit is %f\n", canonical, credit);
    } else {
        printf("no canonical result found\n");
    }
    validate_stats.print();
}
#endif
