/** \file main.h
    \brief File containing user's guide documentation
*/
/** \mainpage 

    \section ug_section User's Guide
    
    This document describes the open-source MPI implementation of a
    Bayesian analysis of mass and radius data to determine the mass
    versus radius curve and the equation of state of dense matter.
    This package will principally be useful for those physicists and
    astrophysicists who are already familiar with C++ and are
    interested in modifying this code for their own use.

    This code was originally supported by Chandra grant TM1-12003X.

    This is a beta version. Use at your own risk.

    Currently, \bm is dual-hosted as an SVN respostory at
    http://www.sourceforge.net/projects/bamr and a git repository at
    http://www.github.com/awsteiner/bamr . This HTML documentation 
    is hosted at http://bamr.sourceforge.net.

    If you are considering using this code for your research, I
    encourage you to contact me so that I can help you with the
    details and so that you can let me know if and how this code is
    useful to you. Nevertheless, you are not required to contact me
    and I will be improving documentation and updating this code as
    time permits.
    
    \hline
    \section contents_sect Contents

    - \ref install_sect
    - \ref dl_sect
    - \ref usage_sect
    - \ref infile_sect 
    - \ref outfile_sect
    - \ref detail_sect
    - \ref ack_sect
    - \ref ref_sect
    - \ref license_page

    \hline
    \section install_sect Installation
    
    The \bm executable requires the installation of 
    <a href="http://www.gnu.org/software/gsl">GSL</a> 
    (versions 1.15 and later),     
    <a href="http://www.hdfgroup.org">HDF5</a> 
    (versions 1.8.4 and later), 
    \htmlonly
    the current development version of
    <a href="http://o2scl.sourceforge.net">O<span style='position:
    relative; top: 0.3em; font-size: 0.8em'>2</span>scl</a> (in the
    <tt>branches/dev</tt> folder of the 
    <a href="http://o2scl.sourceforge.net">O<span style='position:
    relative; top: 0.3em; font-size: 0.8em'>2</span>scl</a> distribution),
    \endhtmlonly
    \latexonly
    O$_2$scl
    \endlatexonly
    and MPI (tested with openmpi-1.4.2). After these four packages are
    successfully installed, you will need to edit the \c makefile and
    then compile \bm before execution.

    \hline
    \section dl_sect Download

    The most recent release version can be obtained from either
    of the following:
    \verbatim
    svn checkout svn://svn.code.sf.net/p/bamr/code/trunk bamr
    git clone https://github.com/awsteiner/bamr.git
    \endverbatim

    \hline
    \section usage_sect Basic Usage
    
    The basic usage is something like
    \verbatim
    ./bamr -model twop -run default.in -mcmc run1
    \endverbatim
    to perform a one day run with model \c twop with the input
    file in \c default.in. 

    There are several variables which can be modified with the
    \c set command, e.g. 
    \verbatim
    ./bamr -model twop -set max_time 43200 -run default.in -mcmc run2
    \endverbatim

    An example of an MPI invocation is
    \verbatim
    mpirun -np 4 ./bamr -set model twop -run default.in -mcmc run3 &
    \endverbatim
    which runs with four processors on the current machine.

    Also try
    \verbatim
    ./bamr -help
    \endverbatim
    which outputs some additional information on the 
    relevant functions and parameters. 

    \hline
    \section infile_sect Data Files

    The data files are HDF5 files (typically named with a <tt>.o2</tt>
    extension) which contain one \ref o2scl::table3d object giving the
    probability density of a neutron star observation as a slice in
    that table.

    \hline
    \section outfile_sect Output Files

    Output is stored in HDF files with a prefix given by the
    argument to the \c mcmc command, one set of files
    for each processor. Output includes files with the 
    following suffixes (where X is the processor index):
    - \c _X_out: Main output file containing full Markov chain
    and most of the parameter values
    - \c _X_scr: Running output of entire simulation

    If the executable is run directly (without <tt>mpirun</tt>)
    then X is always zero.

    \hline
    \section detail_sect Some Details

    The basic functionality is provided in the \ref bamr::bamr_class
    and each Monte Carlo point is an object of type \ref bamr::entry.
    All of the "models" (EOS parameterizations) are children of \ref
    bamr::model class.

    If the initial guess has no probability, then the code will fail.
    This is indicated by the line \c "Initial weight zero." in
    the \c _scr file. The easiest fix is just to change the initial 
    guess.

    In order to make the output more efficient, the table 
    representing the full Markov chain is divided up into 
    tables with about 10,000 rows each. The total number of
    tables is stored in <tt>n_chains</tt>.

    \hline
    \section model_sect EOS Model

    Some EOS models are already provided. New models (i.e. new
    children of the \ref bamr::model class) must perform several tasks

    - The function \ref bamr::model::compute_eos() should use the
    parameters in the \ref bamr::entry argument to compute the EOS and
    store it in the object returned by \ref
    o2scl::nstar_cold::get_eos_results().

    - The energy density should be stored in a column named
    <tt>ed</tt> and the pressure in <tt>pr</tt> with the correct units
    set for each column (currently only <tt>1/fm^4</tt> is supported).

    - If \ref bamr::bamr_class::baryon_density is true and the EOS
    model did not already compute the baryon density in a column named
    <tt>"nb"</tt>, then \ref bamr::model::compute_eos() should return
    one baryon density and energy density in \ref
    bamr::model::baryon_density_point().

    - If the model provides the symmetry energy and its density
    derivative, it should be stored as constants named <tt>"S"</tt>
    and <tt>"L"</tt> in the table (in \f$ 1/\mathrm{fm} \f$ ).

    - Causality is automatically checked in bamr::compute_star(), but
    the \ref bamr::model::compute_eos() function should check that the
    pressure is not decreasing.

    - Finally, it is recommended to set the interpolation type in the
    \ref o2scl::table_units object to linear interpolation.

    \hline
    \section func_stack_sect Partial Function Call Stack

    The top-level functions in the call stack are:
    - \ref bamr::bamr_class::run()
      - \ref bamr::bamr_class::setup_cli()
      - Command <tt>"model"</tt>: \ref bamr::bamr_class::set_model()
      - Command <tt>"add-data"</tt>: \ref bamr::bamr_class::add_data()
      - Command <tt>"first-point"</tt>: 
        \ref bamr::bamr_class::set_first_point()
      - Command <tt>"mcmc"</tt>: \ref bamr::bamr_class::mcmc()
        - \ref bamr::bamr_class::mcmc_init()
        - \ref bamr::bamr_class::load_mc()
        - \ref bamr::bamr_class::init_grids_table()
          - \ref bamr::bamr_class::table_names_units()
        - Run initial point:
          - \ref bamr::bamr_class::compute_weight() (see below)
          - \ref bamr::bamr_class::add_measurement()
            - \ref bamr::bamr_class::fill_line()
          - \ref bamr::bamr_class::output_best()
        - Main MCMC loop: 
          - If at least one source: \ref bamr::bamr_class::select_mass()
          - \ref bamr::bamr_class::compute_weight() (see below)
          - \ref bamr::bamr_class::make_step()
          - \ref bamr::bamr_class::add_measurement()
            - \ref bamr::bamr_class::fill_line()
          - \ref bamr::bamr_class::output_best()
          - \ref bamr::bamr_class::update_files()
            - If first file update: \ref bamr::bamr_class::first_update()
      - Done with <tt>"mcmc"</tt> command. 

    The operation of \ref bamr::bamr_class::compute_weight() can
    be summarized with:
    - \ref bamr::bamr_class::compute_weight()
      - \ref bamr::bamr_class::compute_star()
        - If the model has an EOS: 
          - \ref bamr::model::compute_eos() to compute the EOS
          - Check pressure is increasing everywhere
          - Compute baryon density if necessary
          - Call \ref bamr::bamr_class::prepare_eos()
          - Compute crust if necessary
          - \ref o2scl::tov_solve::mvsr() to compute the mass-radius curve
          - Check maximum mass
          - If some masses are too large: \ref bamr::bamr_class::select_mass() 
        - Otherwise if there's no EOS: \ref bamr::model::compute_mr()
        - Test for causality

    \hline
    \section changelog_sect Recent Change Log

    April 2015: Added process.cpp and created new functions \ref
    bamr::model::setup_params and \ref bamr::model::remove_params() .
    Added several new models.

    \hline
    \section ack_sect Acknowledgements

    I would like to thank Paulo Bedaque, Ed Brown, Farrukh Fattoyev,
    Stefano Gandolfi, Jim Lattimer, and Will Newton for their
    collaboration on these projects.

    \hline
    \section ref_sect Bibliography

    Some of the references which contain links should direct you to
    the work referred to using its DOI identifer.
    
    \anchor Bedaque15sv Bedaque15sv:
    <a href="http://dx.doi.org/10.1103/PhysRevLett.114.031103">
    P. Bedaque and A.W. Steiner</a>,
    Phys. Rev. Lett. \b 114 (2015).

    \anchor Lattimer14co Lattimer14co:
    <a href="http://dx.doi.org/10.1140/epja/i2014-14040-y">
    J.M. Lattimer and A.W. Steiner</a>,
    Eur. Phys. J. A \b 50 (2014) 40.

    \anchor Lattimer14ns Lattimer14ns:
    <a href="http://dx.doi.org/10.1088/0004-637X/784/2/123">
    J.M. Lattimer and A.W. Steiner</a>,
    Astrophys. J. \b 784 (2014) 123.

    \anchor Steiner10te Steiner10te:
    <a href="http://dx.doi.org/10.1088/0004-637X/722/1/33">
    A.W. Steiner, J.M. Lattimer, E.F. Brown</a>,
    Astrophys. J. \b 722 (2010) 33.

    \anchor Steiner12cn Steiner12cn:
    <a href="http://dx.doi.org/10.1103/PhysRevLett.108.081102">
    A.W. Steiner and S. Gandolfi</a>,
    Phys. Rev. Lett. \b 108 (2012) 081102.

    \anchor Steiner13tn Steiner13tn:
    <a href="http://dx.doi.org/10.1088/2041-8205/765/1/L5">
    A.W. Steiner, J.M. Lattimer, E.F. Brown</a>,
    Astrophys. J. Lett. \b 765 (2013) 5.

    \anchor Steiner15un Steiner15un:
    <a href="http://dx.doi.org/10.1103/PhysRevC.91.015804">
    A.W. Steiner, S. Gandolfi, F.J. Fattoyev, and W.G. Newton</a>,
    Phys. Rev. C \b 91 (2015) 015804.

    \page license_page Licensing
   
    All code is licensed under version 3 of the GPL as provided in the
    files \c COPYING and in \c doc/gpl_license.txt.

    \verbinclude gpl_license.txt

    This documentation is provided under the GNU Free Documentation
    License, as given below and provided in \c
    doc/fdl_license.txt. 
    
    \verbinclude fdl_license.txt
*/
