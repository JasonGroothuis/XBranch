/*
 *  angdist.h
 *
 *  Computes angular distance between two lat/lon points
 *
 * $Id: mb_angdist.h,v 1.1.2.1 2006/12/14 22:24:40 korpela Exp $
 *
 */

double angdist(double r1, double d1, double r2, double d2) ;
double angdist(const coordinate_t &a,const coordinate_t &b);

/*
 * $Log: mb_angdist.h,v $
 * Revision 1.1.2.1  2006/12/14 22:24:40  korpela
 * *** empty log message ***
 *
 * Revision 1.1  2003/06/03 00:16:09  korpela
 *
 * Initial splitter under CVS control.
 *
 * Revision 3.0  2001/08/01 19:04:57  korpela
 * Check this in before Paul screws it up.
 *
 * Revision 2.1  1998/11/02 16:41:21  korpela
 * Minor Change.
 *
 * Revision 2.0  1998/10/30  22:00:04  korpela
 * Conversion to C++ and merger with client source tree.
 *
 * Revision 1.1  1998/10/27  01:03:21  korpela
 * Initial revision
 *
 *
 */

