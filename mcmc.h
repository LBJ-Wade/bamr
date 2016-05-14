/*
  -------------------------------------------------------------------
  
  Copyright (C) 2012-2016, Andrew W. Steiner
  
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
/** \file mcmc.h
    \brief Definition of main mcmc class
*/
#ifndef MCMC_H
#define MCMC_H

#include <iostream>

#include <boost/numeric/ublas/vector.hpp>

#ifndef NO_MPI
#include <mpi.h>
#endif

#include <o2scl/rng_gsl.h>
#include <o2scl/uniform_grid.h>
#include <o2scl/table3d.h>
#include <o2scl/hdf_file.h>
#include <o2scl/exception.h>
#include <o2scl/prob_dens_func.h>
#include <o2scl/cholesky.h>
#include <o2scl/vector.h>

#ifdef O2SCL_READLINE
#include <o2scl/cli_readline.h>
#else
#include <o2scl/cli.h>
#endif

/** \brief Main namespace
    
    The mcmc namespace which holds all classes and functions.
    
    This file is documented in mcmc.h .
*/
namespace mcmc_namespace {
  
  typedef boost::numeric::ublas::vector<double> ubvector;
  typedef boost::numeric::ublas::matrix<double> ubmatrix;
  
  /** \brief Desc
   */
  class default_model {
    
  public:
    
    virtual void init() {
      return;
    };
    
    /** \brief Desc
     */
    size_t nparams;

    /** \brief Desc
     */
    virtual double compute_point(ubvector &pars, std::ofstream &scr_out,
				 int &success, ubvector &dat)=0;
    
    /** \brief Specify an initial point
     */
    virtual void initial_point(size_t ix, ubvector &pars) {
      ubvector low(nparams), high(nparams);
      low_limits(low);
      high_limits(high);
      for(size_t i=0;i<nparams;i++) {
	pars[i]=(low[i]+high[i])/2.0;
      }
      return;
    }
    
    /// \name Functions for MCMC parameters
    //@{
    /** \brief Set the lower boundaries for all the parameters
     */
    virtual void low_limits(ubvector &pars)=0;

    /** \brief Set the upper boundaries for all the parameters
     */
    virtual void high_limits(ubvector &pars)=0;

    /// Set up the parameter names
    virtual void param_names(std::vector<std::string> &names)=0;
    
    /// Set up the parameter units
    virtual void param_units(std::vector<std::string> &units)=0;
    //@}

    /// \name Functions for model parameters fixed during the MCMC run
    //@{
    /** \brief Setup model parameters */
    virtual void setup_params(o2scl::cli &cl) {
      return;
    }
    
    /** \brief Remove model parameters */
    virtual void remove_params(o2scl::cli &cl) {
      return;
    }
    //@}

  };

  /** \brief A generic MCMC simulator
   */
  template<class data_t=ubvector,
    class model_t=default_model> class mcmc_base {

  protected:

  /// Parameter lower limits
  ubvector low;
  /// Parameter upper limits
  ubvector high;
  
  /// Number of parameters
  size_t nparams;
  
  /// Parameter names
  std::vector<std::string> param_names;
  
  /// Parameter units
  std::vector<std::string> param_units;
  
  /// If true, use affine-invariant Monte Carlo
  bool aff_inv;
  
  /** \brief Number of walkers for affine-invariant MC or 1 
      otherwise (default 1)
  */
  size_t nwalk;

  /// \name Member data for the Metropolis-Hastings step
  //@{
  /// A Gaussian probability distribution
  o2scl::prob_dens_gaussian pdg;
    
  /// If true, then use Metropolis-Hastings with a multivariate Gaussian
  int hg_mode;
    
  /// The Cholesky decomposition of the covariance matrix
  ubmatrix hg_chol;
    
  /// The inverse of the covariance matrix
  ubmatrix hg_covar_inv;
    
  /// The normalization factor
  double hg_norm;
    
  /// The location of the peak
  ubvector hg_best;
  //@}

  /// Maximum number of iterations (default 0)
  int max_iters;

  /// MCMC stepsize factor (default 10.0)
  double step_fac;

  /** \brief Number of warm up steps (successful steps not
      iterations) (default 0)
	
      \note Not to be confused with <tt>warm_up</tt>, which is 
      a boolean local variable in some functions not an int.
  */
  int n_warm_up;

  /** \brief Time in seconds (default is 86400 seconds or 1 day)
   */
  double max_time;

  /** \brief If non-zero, use as the seed for the random number 
      generator (default 0)
  */
  int user_seed;

  /// \name MPI properties
  //@{
  /// The MPI processor rank
  int mpi_rank;

  /// The MPI number of processors
  int mpi_nprocs;

  /// The MPI starting time
  double mpi_start_time;
  //@}
    
  /// The number of Metropolis steps which were accepted
  size_t mc_accept;

  /// The number of Metropolis steps which were rejected
  size_t mc_reject;

  /// Total number of mcmc iterations
  size_t mcmc_iterations;

  /** \brief Error handler for each thread
   */
  o2scl::err_hnd_cpp error_handler;

  /** \brief Prefix for output filenames
   */
  std::string prefix;
    
  /// If true, scr_out has been opened
  bool file_opened;

  /// \name Member data for the Metropolis-Hastings step
  //@{
  /// Return the approximate likelihood
  double approx_like(ubvector &pars) {
    double ret=hg_norm;
    ubvector q(nparams), vtmp(nparams);
    for(size_t i=0;i<nparams;i++) {
      q[i]=pars[i]-hg_best[i];
    }
    vtmp=prod(hg_covar_inv,q);
    ret*=exp(-0.5*inner_prod(q,vtmp));
    return ret;
  }
  //@}

#ifdef O2SCL_READLINE
  /// Command-line interface
  o2scl::cli_readline cl;
#else
  /// Command-line interface
  o2scl::cli cl;
#endif

  /// Random number generator
  o2scl::rng_gsl gr;
  
  /// The screen output file
  std::ofstream scr_out;

  /// If true, output next point (default true)
  bool output_next;

  /** \brief The arguments sent to the command-line
   */
  std::vector<std::string> cl_args;

  /// Vector of data objects
  std::vector<data_t> data_arr;
  
  /// Model object (initialized in constructor)
  std::shared_ptr<model_t> mod;

  /// \name Parameter objects for the 'set' command
  //@{
  o2scl::cli::parameter_double p_max_time;
  o2scl::cli::parameter_double p_step_fac;
  o2scl::cli::parameter_int p_n_warm_up;
  o2scl::cli::parameter_int p_grid_size;
  o2scl::cli::parameter_int p_user_seed;
  o2scl::cli::parameter_int p_max_iters;
  o2scl::cli::parameter_bool p_output_next;
  o2scl::cli::parameter_bool p_aff_inv;
  o2scl::cli::parameter_string p_prefix;
  //@}

  public:
  
  mcmc_base(std::shared_ptr<model_t> &m) {

    // Parameters
    prefix="mcmc";
    max_iters=0;
    user_seed=0;
    n_warm_up=0;

    // Default to 24 hours
    max_time=3.6e3*24;
    output_next=true;

    // MC step parameters
    aff_inv=false;
    hg_mode=0;
    step_fac=10.0;
    nwalk=1;

    // Initial values
    mpi_nprocs=1;
    mpi_rank=0;

    file_opened=false;

    mod=m;
    mod->setup_params(this->cl);
  }
  
  /** \brief Set up the 'cli' object
      
      This function just adds the four commands and the 'set' parameters
  */
  virtual void setup_cli() {

    // ---------------------------------------
    // Set parameters
    
    p_max_time.d=&this->max_time;
    p_max_time.help=
      "Maximum run time in seconds (default 86400 sec or 1 day).";
    this->cl.par_list.insert(std::make_pair("max_time",&p_max_time));
    
    p_step_fac.d=&this->step_fac;
    p_step_fac.help=
      ((std::string)"MCMC step factor. The step size for each ")+
      "variable is taken as the difference between the high and low "+
      "limits divided by this factor (default 10.0). This factor can "+
      "be increased if the acceptance rate is too small, but care must "+
      "be taken, e.g. if the conditional probability is multimodal. If "+
      "this step size is smaller than 1.0, it is reset to 1.0 .";
    this->cl.par_list.insert(std::make_pair("step_fac",&p_step_fac));

    p_n_warm_up.i=&this->n_warm_up;
    p_n_warm_up.help=((std::string)"Minimum number of warm up iterations ")+
      "(default 0).";
    this->cl.par_list.insert(std::make_pair("n_warm_up",&p_n_warm_up));

    p_user_seed.i=&this->user_seed;
    p_user_seed.help=((std::string)"Seed for multiplier for random number ")+
      "generator. If zero is given (the default), then mcmc() uses "+
      "time(0) to generate a random seed.";
    this->cl.par_list.insert(std::make_pair("user_seed",&p_user_seed));

    p_max_iters.i=&this->max_iters;
    p_max_iters.help=((std::string)"If non-zero, limit the number of ")+
      "iterations to be less than the specified number (default zero).";
    this->cl.par_list.insert(std::make_pair("max_iters",&p_max_iters));

    p_output_next.b=&this->output_next;
    p_output_next.help=((std::string)"If true, output next point ")+
      "to the '_scr' file before calling TOV solver (default true).";
    this->cl.par_list.insert(std::make_pair("output_next",&p_output_next));

    p_aff_inv.b=&this->aff_inv;
    p_aff_inv.help="";
    this->cl.par_list.insert(std::make_pair("aff_inv",&p_aff_inv));

    p_prefix.str=&this->prefix;
    p_prefix.help="Output file prefix (default 'bamr').";
    this->cl.par_list.insert(std::make_pair("prefix",&p_prefix));

    return;
  }    

  /// Main wrapper for parsing command-line arguments
  virtual void run(int argc, char *argv[]) {

    // ---------------------------------------
    // Set error handler for this thread
  
    o2scl::err_hnd=&this->error_handler;
  
    // ---------------------------------------
    // Process command-line arguments and run
  
    setup_cli();

#ifndef NO_MPI
    // Get MPI rank, etc.
    MPI_Comm_rank(MPI_COMM_WORLD,&this->mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD,&this->mpi_nprocs);
#endif

    // Process arguments
    for(int i=0;i<argc;i++) {
      this->cl_args.push_back(argv[i]);
    }

    this->cl.prompt="mcmc> ";
    this->cl.run_auto(argc,argv);

    if (file_opened) {
      // Close main output file
      this->scr_out.close();
    }
 
    return;
  }    

  };

  /** \brief A generic MCMC simulator with HDF5 file I/O
   */
  template<class data_t=ubvector,
    class model_t=default_model> class mcmc_class : 
    public mcmc_base<data_t,model_t> {
    
  public:
  
  /// \name Other variables
  //@{
  /// The first point in the parameter space
  ubvector initial_point;
    
  /// The file containing the initial point
  std::string initial_point_file;

  /// \name Integer designating how to set the initial point
  //@{
  int initial_point_type;
  static const int fp_unspecified=-1;
  static const int fp_last=-2;
  static const int fp_best=-3;
  //@}

  /// \name Desc
  //@{
  std::vector<int> ret_codes;
  static const int ix_success=0;
  static const int ix_zero_wgt=1;
  //@}
  
  /// If true, then \ref first_update() has been called
  bool first_file_update;
  
  /// \name Parameter objects for the 'set' command
  //@{
  o2scl::cli::parameter_int p_max_chain_size;
  o2scl::cli::parameter_int p_file_update_iters;
  //@}
  
  /** \brief The number of MCMC successes between file updates
      (default 40)
  */
  int file_update_iters;

  /** \brief Maximum size of Markov chain (default 10000)
   */
  int max_chain_size;
    
  /// Number of Markov chain segments
  size_t n_chains;

  /// Number of chains
  size_t chain_size;
  
  /// Main data table for Markov chain
  o2scl::table_units<> tc;
    
  /** \brief Set up the 'cli' object
      
      This function just adds the four commands and the 'set' parameters
  */
  virtual void setup_cli() {

    mcmc_base<data_t,model_t>::setup_cli();
    
    // ---------------------------------------
    // Set commands/options

    static const size_t nopt=2;
    o2scl::comm_option_s options[nopt]={
      {'m',"mcmc","Perform the Markov Chain Monte Carlo simulation.",
       0,0,"",((std::string)"This is the main part of ")+
       "the code which performs the simulation. Make sure to set the "+
       "model first using the 'model' command first.",
       new o2scl::comm_option_mfptr<mcmc_class>(this,&mcmc_class::mcmc),
       o2scl::cli::comm_option_both},
      {'f',"first-point","Set the starting point in the parameter space",
       1,-1,"<mode> [...]",
       ((std::string)"Mode can be one of 'best', 'last', 'N', or 'values'. ")+
       "If mode is 'best', then it uses the point with the largest "+
       "weight and the second argument specifies the file. If mode is "+
       "'last' then it uses the last point and the second argument "+
       "specifies the file. If mode is 'N' then it uses the Nth point, "+
       "the second argument specifies the value of N and the third "+
       "argument specifies the file. If mode is 'values', then the remaining "+
       "arguments specify all the parameter values. On the command-line, "+
       "enclose negative values in quotes and parentheses, i.e. \"(-1.00)\" "+
       "to ensure they do not get confused with other options.",
       new o2scl::comm_option_mfptr<mcmc_class>
       (this,&mcmc_class::set_initial_point),
       o2scl::cli::comm_option_both}
      /*
	{'s',"hastings","Specify distribution for M-H step",
	1,1,"<filename>",
	((string)"Desc. ")+"Desc2.",
	new comm_option_mfptr<mcmc_class>(this,&mcmc_class::hastings),
	cli::comm_option_both}
      */
    };
    this->cl.set_comm_option_vec(nopt,options);

    p_file_update_iters.i=&file_update_iters;
    p_file_update_iters.help=
      ((std::string)"Number of MCMC successes between ")+
      "file upates (default 10, minimum value 1).";
    this->cl.par_list.insert(std::make_pair("file_update_iters",
					    &p_file_update_iters));
    
    p_max_chain_size.i=&max_chain_size;
    p_max_chain_size.help=
      ((std::string)"Maximum Markov chain size (default ")+
      "10000).";
    this->cl.par_list.insert(std::make_pair("max_chain_size",
					    &p_max_chain_size));

    return;
  }
 
  /** \brief Desc
   */
  virtual int mcmc_init() {
    return 0;
  };
  
  /** \brief Desc
   */
  virtual void first_update(o2scl_hdf::hdf_file &hf) {
    
    hf.sets_vec("param_names",this->param_names);
    
    hf.set_szt("nparams",this->nparams);
    hf.setd("max_time",this->max_time);
    hf.seti("user_seed",this->user_seed);
    hf.seti("n_warm_up",this->n_warm_up);
    hf.setd("step_fac",this->step_fac);
    hf.seti("max_iters",this->max_iters);
    //hf.seti("debug_line",debug_line);
    hf.seti("file_update_iters",file_update_iters);
    hf.seti("output_next",this->output_next);
    hf.seti("initial_point_type",initial_point_type);
    hf.sets("initial_point_file",initial_point_file);
    hf.setd_vec_copy("initial_point",initial_point);
    
    hf.setd_vec_copy("low",this->low);
    hf.setd_vec_copy("high",this->high);
    
    hf.sets_vec("cl_args",this->cl_args);
    
    return;
  }
  
  /** \brief Output the best point so far
   */
  virtual void output_best(ubvector &best, double w_best, data_t &dat) {
    if (this->file_opened==false) {
      // Open main output file
      this->scr_out.open((this->prefix+"_"+
			  std::to_string(this->mpi_rank)+"_scr").c_str());
      this->scr_out.setf(std::ios::scientific);
      this->file_opened=true;
      this->scr_out << "Opened main file in function 'output_best()'."
		    << std::endl;
    }
    this->scr_out << "Best: ";
    o2scl::vector_out(this->scr_out,best);
    this->scr_out << " " << w_best << std::endl;
    return;
  }

  /** \brief Create strings which contain column names and units
   */
  virtual void table_names_units(std::string &s, std::string &u) {
    s="mult weight ";
    u=". . ";
    for(size_t i=0;i<this->nparams;i++) {
      s+=((std::string)"param_")+this->param_names[i]+" ";
      if (this->param_units[i].length()>0) {
	u+=this->param_units[i]+" ";
      } else {
	u+=". ";
      }
    }
    return;
  }

  /** \brief Fill \c line with data for insertion into the table
   */
  virtual void fill_line(ubvector &pars, double weight, data_t &dat,
			 std::vector<double> &line) {

    // Initial multiplier
    line.push_back(1.0);
    line.push_back(weight);
    for(size_t i=0;i<pars.size();i++) {
      line.push_back(pars[i]);
    }
    
    return;
  }
  
  /** \brief Update the files
   */
  void update_files() {
    
    // Open file
    o2scl_hdf::hdf_file hf;
    hf.open_or_create(this->prefix+"_"+
		      std::to_string(this->mpi_rank)+"_out");
    
    // First time, output some initial quantities
    if (first_file_update==false) {
      first_update(hf);
      first_file_update=true;
    }
    
    hf.set_szt("mc_accept",this->mc_accept);
    hf.set_szt("mc_reject",this->mc_reject);
    hf.set_szt("mcmc_iterations",this->mcmc_iterations);
    hf.seti_vec("ret_codes",ret_codes);
    
    // Store Markov chain
    if (n_chains==0) n_chains++;
    hf.set_szt("n_chains",n_chains);
    std::string ch_name="markov_chain"+std::to_string(this->n_chains-1);
    hdf_output(hf,tc,ch_name);
    if (((int)tc.get_nlines())==max_chain_size) {
      tc.clear_data();
      n_chains++;
    }
    
    hf.close();

    // Close file
    return;
  }

  /** \brief Add a measurement to the table
   */
  virtual void add_measurement(ubvector &pars, double weight, data_t &dat,
			       bool new_meas) {

    // Test to see if we need to add a new line of data or
    // increment the weight on the previous line
    if (tc.get_nlines()<=(this->nwalk-1) || new_meas==true) {

      std::vector<double> line;
      fill_line(pars,weight,dat,line);
      
      // Done adding values, check size and add to table
      if (line.size()!=tc.get_ncolumns()) {
	this->scr_out << "line.size(): " << line.size() << std::endl;
	this->scr_out << "tc.get_ncolumns(): "
		      << tc.get_ncolumns() << std::endl;
	for(size_t i=0;i<line.size() && i<tc.get_ncolumns();i++) {
	  this->scr_out << line[i] << " "
			<< tc.get_column_name(i) << std::endl;
	}
	O2SCL_ERR("Alignment problem in mcmc_class::add_measurement().",
		  o2scl::exc_efailed);
      }
      tc.line_of_data(line.size(),line);
      
      if (false) {
	std::vector<std::string> sc_in, sc_out;
	for(size_t k=0;k<line.size();k++) {
	  sc_in.push_back(tc.get_column_name(k)+": "+o2scl::dtos(line[k]));
	}
	o2scl::screenify(line.size(),sc_in,sc_out);
	for(size_t k=0;k<sc_out.size();k++) {
	  std::cout << sc_out[k] << std::endl;
	}
	std::cout << "Press a key and enter to continue." << std::endl;
	char ch;
	std::cin >> ch;
      }
      
    } else if (tc.get_nlines()>0) {

      // Otherwise, just increment the multiplier on the previous line
      tc.set("mult",tc.get_nlines()-this->nwalk,
	     tc.get("mult",tc.get_nlines()-this->nwalk)+1.0);
    }
  
    return;
  };

  /** \brief Choose a Metropolis-Hastings step
   */
  virtual int hastings(std::vector<std::string> &sv, 
		       bool itive_com) {

#ifdef NEVER_DEFINED
    bool debug=true;

    if (file_opened==false) {
      // Open main output file
      scr_out.open((prefix+"_"+std::to_string(mpi_rank)+"_scr").c_str());
      scr_out.setf(ios::scientific);
      file_opened=true;
      scr_out << "Opened main file in command 'hastings'." << endl;
    }

    if (sv.size()<2) {
      cout << "No arguments given to 'hastings'." << endl;
      return exc_efailed;
    }

    if (model_type.length()==0) {
      cout << "No model selected in 'hastings'." << endl;
      return exc_efailed;
    }

#ifndef MCMC_NO_MPI
    int buffer=0, tag=0;
    if (mpi_nprocs>1 && mpi_rank>0) {
      MPI_Recv(&buffer,1,MPI_INT,mpi_rank-1,tag,MPI_COMM_WORLD,
	       MPI_STATUS_IGNORE);
    }
#endif
  
    // Read the data file
    std::string fname=sv[1];
    scr_out << "Opening file " << fname << " for hastings." << endl;
    hdf_file hf;
    hf.open(fname);
    table_units<> file_tab;
    hdf_input(hf,file_tab,"markov_chain0");
    hf.close();
    scr_out << "Done opening file " << fname << " for hastings." << endl;

#ifndef MCMC_NO_MPI
    if (mpi_nprocs>1 && mpi_rank<mpi_nprocs-1) {
      MPI_Send(&buffer,1,MPI_INT,mpi_rank+1,tag,MPI_COMM_WORLD);
    }
#endif

    // Create a new column equal to mult times weight
    file_tab.function_column("mult*weight","mwgt");
  
    // Remove
    double max_mwgt=file_tab.max("mwgt");
    if (debug) scr_out << "lines: " << file_tab.get_nlines() << endl;
    file_tab.add_constant("max_mwgt",max_mwgt);
    file_tab.delete_rows("mwgt<0.1*max_mwgt");
    if (debug) scr_out << "lines: " << file_tab.get_nlines() << endl;
  
    // The total number of variables
    size_t nv=nparams+nsources;
    if (debug) {
      scr_out << nparams << " parameters and " << nsources << " sources."
	      << endl;
    }
    hg_best.resize(nv);
  
    // Find the average values
    for(size_t i=0;i<nparams;i++) {
      string str_i=((string)"param_")+data_arr[0].modp->param_name(i);
      hg_best[i]=wvector_mean(file_tab.get_nlines(),file_tab[str_i],
			      file_tab["mwgt"]);
    }
    for(size_t i=0;i<nsources;i++) {
      string str_i=((string)"Mns_")+source_names[i];
      hg_best[i+nparams]=wvector_mean(file_tab.get_nlines(),file_tab[str_i],
				      file_tab["mwgt"]);
    }
  
    // Construct the covariance matrix
    ubmatrix covar(nv,nv);
    for(size_t i=0;i<nparams;i++) {
      string str_i=((string)"param_")+data_arr[0].modp->param_name(i);
      for(size_t j=i;j<nparams;j++) {
	string str_j=((string)"param_")+data_arr[0].modp->param_name(j);
	covar(i,j)=wvector_covariance(file_tab.get_nlines(),
				      file_tab[str_i],file_tab[str_j],
				      file_tab["mult"]);
	if (debug) {
	  scr_out << "Covar: " << i << " " << j << " "
		  << covar(i,j) << endl;
	}
	covar(j,i)=covar(i,j);
      }
      for(size_t j=0;j<nsources;j++) {
	string str_j=((string)"Mns_")+source_names[j];
	covar(i,j+nparams)=wvector_covariance(file_tab.get_nlines(),
					      file_tab[str_i],file_tab[str_j],
					      file_tab["mult"]);
	if (debug) {
	  scr_out << "Covar: " << i << " " << j+nparams << " "
		  << covar(i,j+nparams) << endl;
	}
	covar(j+nparams,i)=covar(i,j+nparams);
      }
    }
    for(size_t i=0;i<nsources;i++) {
      string str_i=((string)"Mns_")+source_names[i];
      for(size_t j=i;j<nsources;j++) {
	string str_j=((string)"Mns_")+source_names[j];
	covar(i+nparams,j+nparams)=
	  wvector_covariance(file_tab.get_nlines(),
			     file_tab[str_i],file_tab[str_j],
			     file_tab["mult"]);
	if (debug) {
	  scr_out << "Covar: " << i+nparams << " " << j+nparams << " "
		  << covar(i+nparams,j+nparams) << endl;
	}
	covar(j+nparams,i+nparams)=covar(i+nparams,j+nparams);
      }
    }

    // Perform the Cholesky decomposition
    hg_chol=covar;
    o2scl_linalg::cholesky_decomp(nv,hg_chol);

    // Find the inverse
    hg_covar_inv=hg_chol;
    o2scl_linalg::cholesky_invert<ubmatrix>(nv,hg_covar_inv);
  
    // Force hg_chol to be lower triangular
    for(size_t i=0;i<nv;i++) {
      for(size_t j=0;j<nv;j++) {
	if (i<j) hg_chol(i,j)=0.0;
      }
    }

    // Compute the normalization, weighted by the likelihood function
    hg_norm=1.0;
    size_t step=file_tab.get_nlines()/20;
    if (step<1) step=1;
    double renorm=0.0;
    double wgt_sum=0.0;
    for(size_t i=0;i<file_tab.get_nlines();i+=step) {
      ubvector e(nparams,nsources);
      for(size_t j=0;j<nparams;j++) {
	string str_j=((string)"param_")+data_arr[0].modp->param_name(j);
	e.params[j]=file_tab.get(str_j,i);
      }
      for(size_t j=0;j<nsources;j++) {
	string str_j=((string)"Mns_")+source_names[j];
	e.mass[j]=file_tab.get(str_j,i);
      }
      double wgt=file_tab.get("mult",i)*file_tab.get("weight",i);
      double rat=wgt/approx_like(e);
      renorm+=wgt*wgt/approx_like(e);
      if (debug) {
	scr_out << wgt << " " << approx_like(e) << " " << rat << endl;
      }
      wgt_sum+=wgt;
    }
    renorm/=((double)wgt_sum);
    hg_norm*=renorm;
    if (debug) {
      scr_out << "New normalization: " << hg_norm << endl;
    }

    step=file_tab.get_nlines()/20;
    if (step<1) step=1;
    for(size_t i=0;i<file_tab.get_nlines();i+=step) {
      ubvector e(nparams,nsources);
      for(size_t j=0;j<nparams;j++) {
	string str_j=((string)"param_")+data_arr[0].modp->param_name(j);
	e.params[j]=file_tab.get(str_j,i);
      }
      for(size_t j=0;j<nsources;j++) {
	string str_j=((string)"Mns_")+source_names[j];
	e.mass[j]=file_tab.get(str_j,i);
      }
      double wgt=file_tab.get("mult",i)*file_tab.get("weight",i);
      double rat=wgt/approx_like(e);
      if (debug) {
	scr_out << wgt << " " << approx_like(e) << " " << rat << endl;
      }
    }
    hg_mode=1;

#endif
  
    return 0;
  }

  /** \brief Set the first point
   */
  int set_initial_point(std::vector<std::string> &sv, 
		      bool itive_com) {

    if (sv.size()<2) {
      std::cout << "No arguments given to 'first-point'." << std::endl;
      return o2scl::exc_efailed;
    }

    if (sv[1]==((std::string)"values")) {

      initial_point.resize(sv.size()-1);
      for(size_t i=2;i<sv.size();i++) {
	initial_point[i-2]=o2scl::function_to_double(sv[i]);
      }
      initial_point_type=fp_unspecified;

    } else if (sv[1]==((std::string)"prefix")) {
  
      initial_point_type=fp_last;
      initial_point_file=sv[2]+((std::string)"_")+
      std::to_string(this->mpi_rank)+"_out";
      
    } else if (sv[1]==((std::string)"last")) {
      initial_point_type=fp_last;
      initial_point_file=sv[2];
    } else if (sv[1]==((std::string)"best")) {
      initial_point_type=fp_best;
      initial_point_file=sv[2];
    } else if (sv[1]==((std::string)"N")) {
      initial_point_type=o2scl::stoi(sv[2]);
      initial_point_file=sv[3];
    }

    return 0;
  }

  /** \brief Desc
   */
  virtual int mcmc(std::vector<std::string> &sv, bool itive_com) {

    // Shortcut for this->mod
    model_t &m=*this->mod;
    
    bool debug=false;
    
    // Set number of parameters
    this->nparams=m.nparams;
    
    // Make sure that first_update() is called when necessary
    first_file_update=false;
      
#ifndef NO_MPI
    this->mpi_start_time=MPI_Wtime();
#else
    this->mpi_start_time=time(0);
#endif
      
    if (this->file_opened==false) {
      // Open main output file
      this->scr_out.open((this->prefix+"_"+
			  std::to_string(this->mpi_rank)+"_scr").c_str());
      this->scr_out.setf(std::ios::scientific);
      this->file_opened=true;
      this->scr_out << "Opened main file in command 'mcmc'." << std::endl;
    }
      
    // Fix file_update_iters if necessary
    if (file_update_iters<1) {
      this->scr_out << "Parameter 'file_update_iters' less "
      << "than 1. Set equal to 1." << std::endl;
      file_update_iters=1;
    }
      
    if (max_chain_size<1) {
      O2SCL_ERR("Parameter 'max_chain_size' must be larger than 1.",
		o2scl::exc_einval);
    }
      
    // Fix step_fac if it's too small
    if (this->step_fac<1.0) {
      this->step_fac=1.0;
      this->scr_out << "Fixed 'step_fac' to 1.0." << std::endl;
    }

    // Get parameter names and units
    m.param_names(this->param_names);
    m.param_units(this->param_units);
    
    // -----------------------------------------------------------
    // Init table
    
    std::string s, u;
    table_names_units(s,u);
    tc.line_of_names(s);
    
    {
      size_t ctr=0;
      std::string unit;
      std::istringstream is(u);
      while(is >> unit) {
	if (unit!=((std::string)".")) {
	  tc.set_unit(tc.get_column_name(ctr),unit);
	}
	ctr++;
      } 
      if (ctr!=tc.get_ncolumns()) {
	O2SCL_ERR("Column/unit alignment in mcmc_class::mcmc().",
		  o2scl::exc_esanity);
      }
    }
    // -----------------------------------------------------------
    
    // Run init() function (have to make sure to do this after opening
    // this->scr_out). 
    int iret=mcmc_init();
    if (iret!=0) return iret;
      
    // Set RNG seed
    unsigned long int seed=time(0);
    if (this->user_seed!=0) {
      seed=this->user_seed;
    }
    seed*=(this->mpi_rank+1);
    this->gr.set_seed(seed);
    this->pdg.set_seed(seed);
    this->scr_out << "Using seed " << seed 
    << " for processor " << this->mpi_rank+1 << "/" 
    << this->mpi_nprocs << "." << std::endl;
    this->scr_out.precision(12);
    this->scr_out << " Start time: " << this->mpi_start_time << std::endl;
    this->scr_out.precision(6);
      
    // First MC point
    std::vector<ubvector> current(1);
    std::vector<double> w_current(1);
    std::vector<bool> step_flags(1);
    this->data_arr.resize(2);
    w_current[0]=0.0;
    
    // For stretch-moves, allocate for each walker
    if (this->aff_inv) {
      current.resize(this->nwalk);
      w_current.resize(this->nwalk);
      step_flags.resize(this->nwalk);
      this->data_arr.resize(2*this->nwalk);
      for(size_t i=0;i<this->nwalk;i++) {
	step_flags[i]=false;
	w_current[i]=0.0;
      }
    }

    // Allocate memory for in all points
    for(size_t i=0;i<this->nwalk;i++) {
      current[i].resize(this->nparams);
    }

#ifndef NO_MPI
    int buffer=0, tag=0;
    if (this->mpi_nprocs>1 && this->mpi_rank>0) {
      MPI_Recv(&buffer,1,MPI_INT,this->mpi_rank-1,tag,MPI_COMM_WORLD,
	       MPI_STATUS_IGNORE);
    }
#endif

    if (initial_point_file.length()>0) {
  
      if (initial_point_type==fp_last) {

	// Read file 
	this->scr_out << "Reading last point from file '" << initial_point_file
	<< "'." << std::endl;
	o2scl_hdf::hdf_file hf;
	hf.open(initial_point_file);
      
	// Read table
	size_t file_n_chains;
	hf.get_szt("n_chains",file_n_chains);
	std::string chain_name=std::string("markov_chain")+
	o2scl::szttos(file_n_chains-1);
	o2scl::table_units<> file_tab;
	hdf_input(hf,file_tab,chain_name);
	size_t last_line=file_tab.get_nlines()-1;
      
	// Get parameters
	for(size_t i=0;i<this->nparams;i++) {
	  std::string pname=((std::string)"param_")+this->param_names[i];
	  current[0][i]=file_tab.get(pname,last_line);
	  this->scr_out << "Parameter named "
			<< this->param_names[i] << " " 
			<< current[0][i] << std::endl;
	}
      
	// Finish up
	this->scr_out << std::endl;
	hf.close();

      } else if (initial_point_type==fp_best) {
	
	std::vector<double> best_point;
	o2scl_hdf::hdf_file hf;
	hf.open(initial_point_file);
	hf.getd_vec("best_point",best_point);
	hf.close();
	this->scr_out << "Reading best point from file '" << initial_point_file
	<< "'." << std::endl;
	for(size_t i=0;i<this->nparams;i++) {
	  current[0][i]=best_point[i];
	  this->scr_out << "Parameter " << i << " : "
			<< current[0][i] << std::endl;
	}
	this->scr_out << "Best weight: "
	<< best_point[this->nparams] << std::endl;
	this->scr_out << std::endl;

      } else {

	// Read file 
	this->scr_out << "Reading "
	<< initial_point_type << "th point from file '" 
	<< initial_point_file
	<< "'." << std::endl;
	o2scl_hdf::hdf_file hf;
	hf.open(initial_point_file);
      
	// Read table
	size_t file_n_chains, row=initial_point_type;
	hf.get_szt("n_chains",file_n_chains);
      
	o2scl::table_units<> file_tab;
	for(size_t k=0;k<file_n_chains;k++) {
	  std::string chain_name=std::string("markov_chain")+o2scl::szttos(k);
	  hdf_input(hf,file_tab,chain_name);
	  if (file_tab.get_nlines()>row) {
	    k=file_n_chains;
	  } else {
	    row-=file_tab.get_nlines();
	  }
	}
	if (row>=file_tab.get_nlines()) {
	  this->scr_out << "Couldn't find point " << initial_point_type 
	  << " in file. Using last point." << std::endl;
	  row=file_tab.get_nlines()-1;
	}
      
	// Get parameters
	for(size_t i=0;i<this->nparams;i++) {
	  std::string pname=((std::string)"param_")+this->param_names[i];
	  current[0][i]=file_tab.get(pname,row);
	  this->scr_out << "Parameter named "
	  << this->param_names[i] << " " 
	  << current[0][i] << std::endl;
	}
      
	// Finish up
	this->scr_out << std::endl;
	hf.close();
      }

    } else if (initial_point.size()>0) {
    
      this->scr_out << "First point from command-line." << std::endl;
      for(size_t i=0;i<this->nparams;i++) {
	current[0][i]=initial_point[i];
	this->scr_out << current[0][i] << std::endl;
      }
      this->scr_out << std::endl;

    } else {

      this->scr_out << "First point from default." << std::endl;
      m.initial_point(current[0]);

    }

#ifndef NO_MPI
    if (this->mpi_nprocs>1 && this->mpi_rank<this->mpi_nprocs-1) {
      MPI_Send(&buffer,1,MPI_INT,this->mpi_rank+1,tag,MPI_COMM_WORLD);
    }
#endif

    this->scr_out << "First point: ";
    o2scl::vector_out(this->scr_out,current[0],true);
    
    // Set lower and upper bounds for parameters
    this->low.resize(this->nparams);
    this->high.resize(this->nparams);
    m.low_limits(this->low);
    m.high_limits(this->high);

    n_chains=0;

    // Entry objects (Must be after read_input() since nsources is set
    // in that function.)
    ubvector next(this->nparams), best(this->nparams);

    // Weights for each entry
    double w_next=0.0, w_best=0.0;

    // Warm-up flag, not to be confused with 'n_warm_up', i.e. the
    // number of warm_up iterations.
    bool warm_up=true;
    if (this->n_warm_up==0) warm_up=false;

    // Keep track of successful and failed MH moves
    this->mc_accept=0;
    this->mc_reject=0;

    // ---------------------------------------------------
    // Compute initial point and initial weights

    int suc;
    double q_current=0.0, q_next=0.0;

    if (this->aff_inv) {
      // Stretch-move steps

      size_t ij_best=0;

      // Initialize each walker in turn
      for(size_t ij=0;ij<this->nwalk;ij++) {

	size_t init_iters=0;
	bool done=false;

	while (!done) {

	  // Begin with the intial point
	  ubvector first(this->nparams);
	  m.initial_point(first);

	  // Make a perturbation from the initial point
	  for(size_t ik=0;ik<this->nparams;ik++) {
	    do {
	      current[ij][ik]=first[ik]+
		(this->gr.random()*2.0-1.0)*
		(this->high[ik]-this->low[ik])/100.0;
	    } while (current[ij][ik]>=this->high[ik] ||
		     current[ij][ik]<=this->low[ik]);
	  }
	
	  // Compute the weight
	  w_current[ij]=m.compute_point(current[ij],this->scr_out,
					   suc,this->data_arr[ij]);
	  this->scr_out << "SM Init: " << ij << " ";
	  o2scl::vector_out(this->scr_out,current[ij]);
	  this->scr_out << " " << w_current[ij] << std::endl;

	  // Keep track of the best point and the best index
	  if (ij==0) {
	    w_best=w_current[0];
	  } else if (w_current[ij]>w_best) {
	    ij_best=ij;
	    w_best=w_current[ij];
	  }

	  // Increment iteration count
	  init_iters++;

	  // If we have a good point, stop the loop
	  if (w_current[ij]>0.0) {
	    done=true;
	    ret_codes[suc]++;
	  } else if (init_iters>1000) {
	    this->scr_out << "Failed to construct initial walkers."
			  << std::endl;
	    return 1;
	  }
	}

	// For the initial point for this walker, add it
	// to the result table
	if (warm_up==false) {
	  // Add the initial point if there's no warm up
	  add_measurement(current[ij],w_current[ij],this->data_arr[ij],
			  true);
	}
      }

      // Output the best initial walker if necessary
      {
	best=current[ij_best];
	output_best(best,w_best,this->data_arr[ij_best]);
      }

    } else {

      // Normal or Metropolis-Hastings steps

      // Compute weight for initial point
      w_current[0]=m.compute_point
      (current[0],this->scr_out,suc,this->data_arr[0]);
      ret_codes[suc]++;
      this->scr_out << "Initial weight: " << w_current[0] << std::endl;
      if (w_current[0]<=0.0) {
	this->scr_out << "Initial weight zero. Aborting." << std::endl;
	exit(-1);
      }

      // Compute the initial Hastings proposal weight
      if (this->hg_mode>0) {
	q_current=this->approx_like(current[0]);
      }

      // Add measurement to output table and output best 
      {
	if (warm_up==false) {
	  // Add the initial point if there's no warm up
	  add_measurement(current[0],w_current[0],this->data_arr[0],
			  true);
	}
	best=current[0];
	w_best=w_current[0];
	output_best(current[0],w_current[0],this->data_arr[0]);
      }
    
    }

    // ---------------------------------------------------

    // Keep track of total number of different points in the parameter
    // space that are considered (some of them may not result in TOV
    // calls because, for example, the EOS was acausal).
    this->mcmc_iterations=0;

    // Main loop
    bool main_done=false;

    // The MCMC is arbitrarily broken up into 20 'blocks', making
    // it easier to keep track of progress and ensure file updates
    size_t block_counter=0;

    while (!main_done) {

      // Walker to move for smove
      size_t ik=0;
      double smove_z=0.0;
    
      // ---------------------------------------------------
      // Select next point
    
      if (this->aff_inv) {

	// Choose walker to move
	ik=this->mcmc_iterations % this->nwalk;
      
	bool in_bounds;
	size_t step_iters=0;
      
	do {

	  in_bounds=true;
	
	  // Choose jth walker
	  size_t ij;
	  do {
	    ij=((size_t)(this->gr.random()*((double)this->nwalk)));
	  } while (ij==ik || ij>=this->nwalk);
	
	  // Select z 
	  double p=this->gr.random();
	  double a=this->step_fac;
	  smove_z=(1.0-2.0*p+2.0*a*p+p*p-2.0*a*p*p+a*a*p*p)/a;
	
	  // Create new trial point
	  for(size_t i=0;i<this->nparams;i++) {
	    next[i]=current[ij][i]+smove_z*(current[ik][i]-current[ij][i]);
	    if (next[i]>=this->high[i] || next[i]<=this->low[i]) {
	      in_bounds=false;
	    }
	  }
	
	  step_iters++;
	  if (step_iters==1000) {
	    this->scr_out << "Failed to find suitable step at point 1."
			  << std::endl;
	    std::cerr << "Failed to find suitable step at point 1."
		      << std::endl;
	    return 2;
	  }

	} while (in_bounds==false);

      } else if (this->hg_mode>0) {
      
	// Make a Metropolis-Hastings step based on previous data
      
	ubvector hg_temp(this->nparams), hg_z(this->nparams);
      
	bool out_of_range;
	int hg_it=0;
      
	do {
	
	  for(size_t k=0;k<this->nparams;k++) {
	    hg_z[k]=this->pdg.sample();
	  }
	  hg_temp=prod(this->hg_chol,hg_z);
	  for(size_t k=0;k<this->nparams;k++) {
	    next[k]=this->hg_best[k]+hg_temp[k];
	  }
	
	  out_of_range=false;
	  for(size_t k=0;k<this->nparams;k++) {
	    if (next[k]<this->low[k] || next[k]>this->high[k]) {
	      out_of_range=true;
	    }
	  }

	  hg_it++;
	  if (hg_it>1000) {
	    O2SCL_ERR("Sanity check in hg step.",o2scl::exc_esanity);
	  }

	} while (out_of_range==true);

	q_next=this->approx_like(next);

      } else {

	// Make a step, ensure that we're in bounds and that
	// the masses are not too large
	for(size_t k=0;k<this->nparams;k++) {
	  
	  next[k]=current[0][k]+(this->gr.random()*2.0-1.0)*
	    (this->high[k]-this->low[k])/this->step_fac;
	
	  // If it's out of range, redo step near boundary
	  if (next[k]<this->low[k]) {
	    next[k]=this->low[k]+this->gr.random()*
	      (this->high[k]-this->low[k])/this->step_fac;
	  } else if (next[k]>this->high[k]) {
	    next[k]=this->high[k]-this->gr.random()*
	      (this->high[k]-this->low[k])/this->step_fac;
	  }
	  
	  if (next[k]<this->low[k] || next[k]>this->high[k]) {
	    O2SCL_ERR("Sanity check in parameter step.",o2scl::exc_esanity);
	  }
	}
      
      }

      // End of select next point
      // ---------------------------------------------------

      // Output the next point
      if (this->output_next) {
	this->scr_out << "Iteration, next: "
		      << this->mcmc_iterations << " " ;
	o2scl::vector_out(this->scr_out,next,true);
      }
      
      // ---------------------------------------------------
      // Compute next weight

      if (this->aff_inv) {
	if (step_flags[ik]==false) {
	  w_next=m.compute_point(next,this->scr_out,suc,
				 this->data_arr[ik+this->nwalk]);
	} else {
	  w_next=m.compute_point(next,this->scr_out,suc,
				 this->data_arr[ik]);
	}
      } else {
	if (step_flags[0]) {
	  w_next=m.compute_point(next,this->scr_out,suc,this->data_arr[1]);
	} else {
	  w_next=m.compute_point(next,this->scr_out,suc,this->data_arr[0]);
	}
      }
      ret_codes[suc]++;

      // ---------------------------------------------------
    
      // Test to ensure new point is good
      if (suc==ix_success && w_next<=0.0) {
	this->scr_out << "Rejected: Zero weight." << std::endl;
	suc=ix_zero_wgt;
      }

      bool force_file_update=false;

      // If the new point is still good, compare with
      // the Metropolis algorithm
      if (suc==ix_success) {

	if (debug) {
	  std::cout << step_flags[0] << " Next: " 
		    << next[0] << " " << w_next << std::endl;
	}
	
	bool accept=false;
	double r=this->gr.random();

	// Metropolis algorithm
	if (this->aff_inv) {
	  if (r<pow(smove_z,((double)this->nwalk)-1.0)*w_next/w_current[ik]) {
	    accept=true;
	  }
	} else if (this->hg_mode>0) {
	  if (r<w_next*q_current/w_current[0]/q_next) {
	    accept=true;
	  }
	} else {
	  if (r<w_next/w_current[0]) {
	    accept=true;
	  }
	}
	
	if (debug) {
	  std::cout << "Metropolis: " << r << " "
		    << w_next/w_current[0] << " " << accept << std::endl;
	}
	
	if (accept) {

	  this->mc_accept++;

	  // Store results from new point
	  if (!warm_up) {
	    if (this->aff_inv) {
	      if (step_flags[ik]==false) {
		add_measurement(next,w_next,this->data_arr[ik+this->nwalk],
				true);
	      } else {
		add_measurement(next,w_next,this->data_arr[ik],true);
	      }
	    } else {
	      if (step_flags[0]==false) {
		add_measurement(next,w_next,this->data_arr[1],true);
	      } else {
		add_measurement(next,w_next,this->data_arr[0],true);
	      }
	    }
	    if (debug) {
	      std::cout << step_flags[0] << " Adding new: " 
			<< next[0] << " " << w_next << std::endl;
	    }
	  }

	  // Output the new point
	  this->scr_out << "MC Acc: " << this->mc_accept << " ";
	  o2scl::vector_out(this->scr_out,next);
	  this->scr_out << " " << w_next << std::endl;
	  
	  // Keep track of best point
	  if (w_next>w_best) {
	    best=next;
	    w_best=w_next;
	    output_best(best,w_best,this->data_arr[0]);
	    force_file_update=true;
	  }

	  // Prepare for next point
	  if (this->aff_inv) {
	    current[ik]=next;
	    w_current[ik]=w_next;
	  } else {
	    current[0]=next;
	    w_current[0]=w_next;
	  }

	  // Flip "first_half" parameter
	  if (this->aff_inv) {
	    step_flags[ik]=!(step_flags[ik]);
	  } else {
	    step_flags[0]=!(step_flags[0]);
	    if (debug) {
	      std::cout << "Flip: " << step_flags[0] << std::endl;
	    }
	  }
	  
	} else {
	    
	  // Point was rejected
	  
	  this->mc_reject++;

	  // Repeat measurement of old point
	  if (!warm_up) {
	    if (this->aff_inv) {
	      if (step_flags[ik]==false) {
		add_measurement(current[ik],w_current[ik],this->data_arr[ik],
				false);
	      } else {
		add_measurement(current[ik],w_current[ik],
				this->data_arr[ik+this->nwalk],
				false);
	      }
	      if (debug) {
		std::cout << step_flags[ik] << " Adding old: "
			  << current[ik][0] << " " << w_current[ik]
			  << std::endl;
	      }
	    } else {
	      if (step_flags[0]==false) {
		add_measurement(current[0],w_current[0],this->data_arr[0],
				false);
	      } else {
		add_measurement(current[0],w_current[0],this->data_arr[1],
				false);
	      }
	      if (debug) {
		std::cout << step_flags[0] << " Adding old: "
			  << current[0][0] << " " << w_current[0] << " "
			  << std::endl;
	      }
	    }
	  }

	  // Output the old point
	  if (this->aff_inv) {
	    this->scr_out << "MC Rej: " << this->mc_accept << " ";
	    o2scl::vector_out(this->scr_out,current[ik]);
	    this->scr_out << " " << w_current[ik] << " "
			  << w_next << std::endl;
	  } else {
	    this->scr_out << "MC Rej: " << this->mc_accept << " ";
	    o2scl::vector_out(this->scr_out,current[0]);
	    this->scr_out << " " << w_current[0] << " "
			  << w_next << std::endl;
	  }

	  // Keep track of best point
	  if (w_next>w_best) {
	    best=next;
	    w_best=w_next;
	    //output_best(best,w_best,this->data_arr[0]);
	    std::cout << "fixme." << std::endl;
	    exit(-1);
	    force_file_update=true;
	    this->scr_out << "Best point with rejected step: "
			  << w_next << " " << w_best << std::endl;
	  }

	}
	  
	// End of "if (suc==ix_success)"
      }

      // ---------------------------------------------------------------

      // After the warm-up is over, the calculation is abritrarily
      // broken up into 20 blocks. The purpose of these blocks is
      // simply to allow easier tracking of progress and to force
      // periodic file updates.

      // This section determines if we have finished a block or if we
      // have finished the full calculation. Note that the value of
      // mcmc_iterations isn't incremented until later.
    
      if (warm_up==false) {

	// If 'max_iters' is zero, then presume we're running over
	// a fixed time interval
	if (this->max_iters==0) {

	  // Determine time elapsed
#ifndef NO_MPI
	  double elapsed=MPI_Wtime()-this->mpi_start_time;
#else
	  double elapsed=time(0)-this->mpi_start_time;
#endif
	  // Force a file update when we've finished a block or if
	  // we've run out of time
	  if (elapsed>this->max_time/
	      ((double)20)*((double)(block_counter+1)) ||
	      (elapsed>this->max_time && block_counter==19)) {
	    force_file_update=true;
	    block_counter++;
	    this->scr_out << "Finished block " << block_counter
			  << " of 20." << std::endl;
	  }

	  // Output elapsed time every 10 iterations. The value of
	  // mcmc_iterations isn't increased until later.
	  if ((this->mcmc_iterations+1)%10==0) {
	    this->scr_out << "Elapsed time: " << elapsed << " of "
			  << this->max_time << " seconds" << std::endl;
	  }
	
	  if (elapsed>this->max_time) {
	    main_done=true;
	  }

	} else {

	  // Otherwise, 'max_iters' is non-zero, so we're completing
	  // a fixed number of iterations.

	  // Force a file update when we've finished a block
	  if (((int)this->mcmc_iterations)+1>
	      this->max_iters*(((int)block_counter)+1)/20) {
	    force_file_update=true;
	    block_counter++;
	    this->scr_out << "Iteration "
			  << this->mcmc_iterations+1 << " of " 
			  << this->max_iters << ", and finished block " 
			  << block_counter << " of 20." << std::endl;
	  }

	  if (((int)this->mcmc_iterations)+1>this->max_iters) {
	    this->scr_out << "Iteration count, " << this->mcmc_iterations 
			  << ", exceed maximum number, "
			  << this->max_iters << "."
			  << std::endl;
	    main_done=true;
	  }
	
	}
     
      }

      // --------------------------------------------------------------
      // Store a copy of measurements in file if 'force_file_update' is
      // true and for a fixed interval of MCMC successes and if the
      // table is at the maximum size. By default file_default_iters is
      // 10 and so the files are updated for every 10 MCMC successes.
    
      if (!warm_up && (force_file_update ||
		       ((int)tc.get_nlines())==max_chain_size || 
		       (this->mcmc_iterations+1) % file_update_iters==0)) {
	this->scr_out << "Updating files." << std::endl;
	update_files();
	this->scr_out << "Done updating files." << std::endl;
      }

      // --------------------------------------------------------------

      // Increment iteration counter
      this->mcmc_iterations++;

      // Leave warm_up mode if necessary
      if (((int)this->mcmc_iterations)>this->n_warm_up && warm_up==true) {
	warm_up=false;
	this->scr_out << "Setting warm_up to false. Reset start time."
		      << std::endl;
#ifndef NO_MPI
	this->max_time-=MPI_Wtime()-this->mpi_start_time;
	this->scr_out << "Resetting max_time to : " << this->max_time
		      << std::endl;
	this->mpi_start_time=MPI_Wtime();
#else
	this->max_time-=time(0)-this->mpi_start_time;
	this->scr_out << "Resetting max_time to : " << this->max_time
		      << std::endl;
	this->mpi_start_time=time(0);
#endif
	this->scr_out.precision(12);
	this->scr_out << " Start time: " << this->mpi_start_time << std::endl;
	this->scr_out.precision(6);
      }
    
      // End of main loop
    }
  
    return 0;
  }

  mcmc_class(std::shared_ptr<model_t> &m) : mcmc_base<data_t,model_t>(m) {

    first_file_update=false;
    
    initial_point_file="";
    initial_point_type=fp_unspecified;

    file_update_iters=40;
    max_chain_size=10000;

    chain_size=0;
    n_chains=0;
    
    ret_codes.resize(100);
    for(size_t i=0;i<100;i++) ret_codes[i]=0;

  }

  };

  // End of mcmc_namespace namespace
}

#endif
