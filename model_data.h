/*
  -------------------------------------------------------------------
  
  Copyright (C) 2012-2018, Andrew W. Steiner
  
  This file is part of Bamr.
  
  Bamr is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 3 of the License, or
  (at your option) any later version.
  
  Bamr is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  
  You should have received a copy of the GNU General Public License
  along with Bamr. If not, see <http://www.gnu.org/licenses/>.

  -------------------------------------------------------------------
*/
/** \file model_data.h
    \brief Definition of \ref model_data
*/
#ifndef MODEL_DATA_H
#define MODEL_DATA_H

#include <iostream>

#ifdef BAMR_MPI
#include <mpi.h>
#endif

#include <o2scl/table3d.h>

#ifdef BAMR_READLINE
#include <o2scl/cli_readline.h>
#else
#include <o2scl/cli.h>
#endif

namespace bamr {
  
  /** \brief Data at each MC point
   */
  class model_data {
    
  public:

    /// Radii
    ubvector rad;

    /// Masses
    ubvector mass;
    
    /// Weights
    ubvector wgts;

    /// M vs. R data
    o2scl::table_units<> mvsr;

    /// EOS data
    o2scl::table_units<> eos;
    
    model_data() {
    }
    
    /** \brief Copy constructor
     */
    model_data(const model_data &md) {
      rad=md.rad;
      mass=md.mass;
      wgts=md.wgts;
      mvsr=md.mvsr;
      eos=md.eos;
    }

#ifdef O2SCL_NEVER_DEFINED
    
  private:
    
    /** \brief Make operator= copy constructor private
     */
    model_data &operator=(const model_data &e);

#endif
    
  };

}

#endif
