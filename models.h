/*
  -------------------------------------------------------------------
  
  Copyright (C) 2012-2017, Andrew W. Steiner
  
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
/** \file models.h
    \brief Definition of EOS models
*/
#ifndef MODELS_H
#define MODELS_H

#include <iostream>

#ifndef BAMR_NO_MPI
#include <mpi.h>
#endif

#include <o2scl/nstar_cold.h>
#include <o2scl/eos_had_schematic.h>
#include <o2scl/root_brent_gsl.h>
#include <o2scl/cli.h>
#include <o2scl/prob_dens_func.h>
#include <o2scl/table3d.h>
#include <o2scl/hdf_io.h>
#include <o2scl/mcmc_para.h>

#ifdef BAMR_READLINE
#include <o2scl/cli_readline.h>
#else
#include <o2scl/cli.h>
#endif

#include "ns_data.h"
#include "settings.h"
#include "model_data.h"
#include "nstar_cold2.h"

namespace bamr {
  
  /** \brief Base class for an EOS parameterization
   */
  class model {
    
  public:
    
    /// Random number generator
    o2scl::rng_gsl gr;
  
    /// The fiducial baryon density
    double nb_n1;
    
    /// The fiducial energy density
    double nb_e1;

  public:

    /// \name Return codes for each point
    //@{
    static const int ix_success=0;
    static const int ix_mr_outside=2;
    static const int ix_r_outside=3;
    static const int ix_press_dec=4;
    static const int ix_nb_problem=5;
    static const int ix_nb_problem2=6;
    static const int ix_crust_unstable=7;
    static const int ix_mvsr_failed=8;
    static const int ix_tov_failure=9;
    static const int ix_small_max=10;
    static const int ix_tov_conv=11;
    static const int ix_mvsr_table=12;
    static const int ix_acausal=13;
    static const int ix_acausal_mr=14;
    static const int ix_param_mismatch=15;
    static const int ix_neg_pressure=16;
    static const int ix_no_eos_table=17;
    static const int ix_eos_solve_failed=18;
    static const int ix_trans_invalid=19;
    static const int ix_SL_invalid=20;
    //@}

    /// Number of parameters (EOS parameters plus mass of each source)
    size_t n_eos_params;
    
    /** \brief TOV solver
	
	The value of \ref o2scl::nstar_cold::nb_start is set to
	0.01 by the constructor
    */
    nstar_cold2 cns;

    /// TOV solver
    o2scl::tov_solve ts;
    
    /// True if the model has an EOS
    bool has_eos;

    /// Schwarzchild radius (set in constructor)
    double schwarz_km;

    /// True if the model provides S and L
    bool has_esym;

    /// \name Grids
    //@{
    o2scl::uniform_grid<double> nb_grid;
    o2scl::uniform_grid<double> e_grid;
    o2scl::uniform_grid<double> m_grid;
    //@}

    /// Lower limit for baryon density of core-crust transition
    double nt_low;

    /// Upper limit for baryon density of core-crust transition
    double nt_high;

    /// Gaussians for core-crust transition
    o2scl::prob_dens_gaussian nt_a, nt_b, nt_c;
    
    /// EOS interpolation object for TOV solver
    o2scl::eos_tov_interp teos;

    /// Settings object
    settings &set;

    /// Mass-radius data
    ns_data &nsd;

    model(settings &s, ns_data &n);

    virtual ~model() {}

    /** \brief Compute the EOS corresponding to parameters in 
	\c e and put output in \c tab_eos
    */
    virtual void compute_eos(const ubvector &pars, int &success,
			     std::ofstream &scr_out, model_data &dat) {
      return;
    }

    virtual void compute_mr(const ubvector &pars, int &success,
			    std::ofstream &scr_out, model_data &dat) {
      return;
    }

    /** \brief Tabulate EOS and then use in cold_nstar
     */
    virtual void compute_star(const ubvector &pars, std::ofstream &scr_out, 
			      int &success, model_data &dat);
    
    /** \brief Compute the EOS corresponding to parameters in 
	\c e and put output in \c tab_eos
    */
    virtual int compute_point(const ubvector &pars, std::ofstream &scr_out, 
			      double &weight, model_data &dat);

    /** \brief Specify the initial point
     */
    virtual void initial_point(ubvector &pars) {
      for(size_t i=0;i<nsd.n_sources;i++) {
	pars[i+n_eos_params]=nsd.init_mass_fracs[i];
      }
      return;
    }

    /** \brief Set parameter information [pure virtual]
     */
    virtual void get_param_info(std::vector<std::string> &names,
				std::vector<std::string> &units,
				ubvector &low, ubvector &high) {
      for(size_t i=0;i<nsd.n_sources;i++) {
	names.push_back("mf_"+nsd.source_names[i]);
	units.push_back("");
	low[i+n_eos_params]=0.0;
	high[i+n_eos_params]=1.0;
      }
      return;
    }

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
    
    /** \brief Copy model parameters */
    virtual void copy_params(model &m) {
      return;
    }
    //@}
    
  };

  /** \brief Two polytropes (8 parameters) from Steiner10te

      \verbatim embed:rst
      Based on the model from [Steiner10te]_. The original limits on
      the parameters are maintained here. This model is referred to as
      Model A in [Steiner13tn]_ and was also used in [Lattimer14ns]_
      (where it was the "Base" model) and in [Lattimer14co]_ .
      \endverbatim

      The EOS from \ref o2scl::eos_had_schematic is used for the EOS
      near the saturation density. The first parameter is \ref
      o2scl::eos_had_base::comp (<tt>comp</tt>), the second is \ref
      o2scl::eos_had_base::kprime (<tt>kprime</tt>), the third is used
      to fix the sum (<tt>esym</tt>) of \ref
      o2scl::eos_had_schematic::a and \ref
      o2scl::eos_had_schematic::b, and the fourth parameter is \ref
      o2scl::eos_had_schematic::gamma (<tt>gamma</tt>). The value of
      \ref o2scl::eos_had_schematic::a defaults to \f$
      17.0~\mathrm{MeV}/(\hbar c) \f$, and can be changed by setting
      the parameter named <tt>kin_sym</tt> at run time. This EOS is
      used up to the transition energy density specified by the fifth
      parameter (<tt>trans1</tt>). The first polytrope is used with an
      index specified by the sixth parameter (<tt>index1</tt>), up to
      an energy density specified by the seventh parameter
      (<tt>trans2</tt>). Finally, the second polytrope is used with an
      index specified by the eighth parameter (<tt>index2</tt>). 

      For a polytrope \f$ P = K \varepsilon^{1+1/n} \f$
      beginning at a pressure of \f$ P_1 \f$, an energy
      density of \f$ \varepsilon_1 \f$ and a baryon density 
      of \f$ n_{B,1} \f$, the baryon density along the polytrope
      is 
      \f[
      n_B = n_{B,1} \left(\frac{\varepsilon}{\varepsilon_1}\right)^{1+n} 
      \left(\frac{\varepsilon_1+P_1}{\varepsilon+P}\right)^{n} \, .
      \f]
      Similarly, the chemical potential is
      \f[
      \mu_B = \mu_{B,1} \left(1 + \frac{P_1}{\varepsilon_1}\right)^{1+n}
      \left(1 + \frac{P}{\varepsilon}\right)^{-(1+n)} \, .
      \f]
      The expression for the 
      baryon density can be inverted to determine \f$ \varepsilon(n_B) \f$
      \f[
      \varepsilon(n_B) = \left[ \left(\frac{n_{B,1}}
      {n_B \varepsilon_1} \right)^{1/n}
      \left(1+\frac{P_1}{\varepsilon_1}\right)-K\right]^{-n} \, .
      \f]
      Sometimes the baryon susceptibility is also useful 
      \f[
      \frac{d \mu_B}{d n_B} = \left(1+1/n\right)
      \left( \frac{P}{\varepsilon}\right)
      \left( \frac{\mu_B}{n_B}\right) \, .
      \f]
  */
  class two_polytropes : public model {

  protected:

    /// Parameter for kinetic part of symmetry energy
    o2scl::cli::parameter_double p_kin_sym;

    /// Neutron for \ref se
    o2scl::fermion neut;

    /// Proton for \ref se
    o2scl::fermion prot;
    
  public:

    /// Low-density EOS
    o2scl::eos_had_schematic se;

    /// \name Functions for model parameters fixed during the MCMC run
    //@{
    /** \brief Setup model parameters */
    virtual void setup_params(o2scl::cli &cl);

    /** \brief Remove model parameters */
    virtual void remove_params(o2scl::cli &cl);
    
    /** \brief Copy model parameters */
    virtual void copy_params(model &m);
    //@}

    /// Create a model object
    two_polytropes(settings &s, ns_data &n);

    virtual ~two_polytropes() {}

    /** \brief Set parameter information [pure virtual]
     */
    virtual void get_param_info(std::vector<std::string> &names,
				std::vector<std::string> &units,
				ubvector &low, ubvector &high);

    /** \brief Compute the EOS corresponding to parameters in 
	\c e and put output in \c tab_eos
    */
    virtual void compute_eos(const ubvector &e, int &success,
			     std::ofstream &scr_out, model_data &dat);

    /** \brief Function to compute the initial guess
     */
    virtual void initial_point(ubvector &e);

  };

  /** \brief Alternate polytropes from \ref Steiner13tn (8 parameters) 

      Referred to as Model B in \ref Steiner13tn. 

      This model is just as in \ref two_polytropes, but in terms of
      the exponents instead of the polytropic indices. The lower limit
      on 'exp1' is 1.5, as in \ref Steiner13tn, but softer EOSs could
      be allowed by setting this to zero. This would not change the
      the final results in \ref Steiner13tn, because the lowest
      pressure EOSs came from \ref bamr::fixed_pressure anyway.

      For a polytrope \f$ P = K \varepsilon^{\Gamma} \f$
      beginning at a pressure of \f$ P_1 \f$, an energy
      density of \f$ \varepsilon_1 \f$ and a baryon density 
      of \f$ n_{B,1} \f$, the baryon density along the polytrope
      is 
      \f[
      n_B = n_{B,1} \left(\frac{\varepsilon}{\varepsilon_1}
      \right)^{\Gamma/(\Gamma-1)} \left(\frac{\varepsilon+P}
      {\varepsilon_1+P_1}\right)^{1/(1-\Gamma)}
      \f]
  */
  class alt_polytropes : public two_polytropes {

  public:

  alt_polytropes(settings &s, ns_data &n) : two_polytropes(s,n) {
    }
    
    virtual ~alt_polytropes() {}

    /** \brief Set parameter information [pure virtual]
     */
    virtual void get_param_info(std::vector<std::string> &names,
				std::vector<std::string> &units,
				ubvector &low, ubvector &high);

    /** \brief Compute the EOS corresponding to parameters in 
	\c e and put output in \c tab_eos
    */
    virtual void compute_eos(const ubvector &e, int &success,
			     std::ofstream &scr_out, model_data &dat);

    /** \brief Function to compute the initial guess
     */
    virtual void initial_point(ubvector &e);
  
  };

  /** \brief Fix pressure on a grid of energy densities 
      from \ref Steiner13tn (8 parameters)
    
      This model is referred to as Model C in \ref Steiner13tn and was
      also used in \ref Lattimer14ns (where it was the model labeled
      "Exo").

      This model is computed as in \ref two_polytropes, but instead of
      using polytropes at high densities, pressures are linearly
      interpolated on a fixed grid of energy densities. The schematic
      EOS (\ref o2scl::eos_had_schematic) is used up to an energy
      density of \f$ 1~\mathrm{fm^{-4}} \f$. The last four parameters
      are pressures named <tt>pres1</tt> through <tt>pres4</tt>. Then
      the line segments are defined by the points
      \f[
      P(2~\mathrm{fm}^{-4}) - P(1~\mathrm{fm}^{-4}) = \mathrm{pres1};
      \quad
      P(3~\mathrm{fm}^{-4}) - P(2~\mathrm{fm}^{-4}) = \mathrm{pres2};
      \quad
      P(5~\mathrm{fm}^{-4}) - P(3~\mathrm{fm}^{-4}) = \mathrm{pres3};
      \quad
      P(7~\mathrm{fm}^{-4}) - P(5~\mathrm{fm}^{-4}) = \mathrm{pres4}
      \f]
      The final line segment is extrapolated up to 
      \f$ \varepsilon = 10~\mathrm{fm^{-4}} \f$

      For a linear EOS, \f$ P = P_1 + c_s^2
      (\varepsilon-\varepsilon_1) \f$ , beginning at a pressure of \f$
      P_1 \f$ , an energy density of \f$ \varepsilon_1 \f$ and a
      baryon density of \f$ n_{B,1} \f$, the baryon density is
      \f[
      n_B = n_{B,1} \left\{ \frac{\left[\varepsilon+
      P_1+c_s^2(\varepsilon-\varepsilon_1)\right]}
      {\varepsilon_1+P_1} \right\}^{1/(1+c_s^2)}
      \f]

  */
  class fixed_pressure : public two_polytropes {

  public:

  fixed_pressure(settings &s, ns_data &n) : two_polytropes(s,n) {
    }
    
    virtual ~fixed_pressure() {}

    /** \brief Set parameter information [pure virtual]
     */
    virtual void get_param_info(std::vector<std::string> &names,
				std::vector<std::string> &units,
				ubvector &low, ubvector &high);

    /** \brief Compute the EOS corresponding to parameters in 
	\c e and put output in \c tab_eos
    */
    virtual void compute_eos(const ubvector &e, int &success,
			     std::ofstream &scr_out, model_data &dat);

    /** \brief Function to compute the initial guess
     */
    virtual void initial_point(ubvector &e);

  };

  /** \brief Generic quark model from \ref Steiner13tn (9 parameters)

      Referred to as Model D in \ref Steiner13tn. 

      This model uses \ref o2scl::eos_had_schematic near saturation,
      a polytrope (with a uniform prior in the exponent like
      \ref alt_polytropes) and then a generic quark matter EOS
      at high densities. 

      Alford et al. 2005 parameterizes quark matter with
      \f[
      P = \frac{3 b_4}{4 \pi^2} \mu^4 - \frac{3 b_2}{4 \pi^2} \mu^2 -B 
      \f]
      where \f$ \mu \f$ is the quark chemical potential. QCD corrections 
      can be parameterized by expressing \f$ b_4 \equiv 1-c \f$ , 
      and values of \f$ c \f$ up to 0.4 (or maybe even larger) are
      reasonable (see discussion after Eq. 4 in Alford et al. (2005)).
      Note that, in charge-neutral matter in beta equilibrium, 
      \f$ \sum_{i=u,d,s,e} n_i \mu_i = \mu_B n_B = \mu n_Q \f$.
      where \f$ \mu_B \f$ and \f$ n_B \f$ are the baryon chemical
      potential and baryon density and \f$ n_Q \f$ is the number
      density of quarks.

      The parameter \f$ b_2 = m_s^2 - 4 \Delta^2 \f$ for CFL quark
      matter, and can thus be positive or negative. A largest possible
      range might be somewhere between \f$ (400~\mathrm{MeV})^2 \f$,
      which corresponds to the situation where the gap is zero and the
      strange quarks receive significant contributions from chiral
      symmetry breaking, to \f$ (150~\mathrm{MeV})^2-4
      (200~\mathrm{MeV})^2 \f$ which corresponds to a bare strange
      quark with a large gap. In units of \f$ \mathrm{fm}^{-1} \f$ ,
      this corresponds to a range of about \f$ -3.5 \f$ to \f$
      4~\mathrm{fm}^{-2} \f$ . In Alford et al. (2010), they choose a
      significantly smaller range, from \f$ -1 \f$ to \f$
      1~\mathrm{fm}^{-2} \f$.
      
      Simplifying the parameterization to 
      \f[
      P = a_4 \mu^4 +a_2 \mu^2 - B 
      \f]
      gives the following ranges
      \f[
      a_4 = 0.045~\mathrm{to}~0.08
      \f]
      and 
      \f[
      a_2 = -0.3~\mathrm{to}~0.3~\mathrm{fm}^{-2}
      \f]
      for the "largest possible range" described above or
      \f[
      a_2 = -0.08~\mathrm{to}~0.08~\mathrm{fm}^{-2}
      \f]
      for the range used by Alford et al. (2010). 
      
      The energy density is
      \f[
      \varepsilon = B + a_2 \mu^2 + 3 a_4 \mu^4
      \f]
    
      Note that 
      \f{eqnarray*}
      \frac{dP}{d \mu} &=& 2 a_2 \mu + 4 a_4 \mu^3 = n_Q \nonumber \\ 
      \frac{d\varepsilon}{d \mu} &=& 2 a_2 \mu + 12 a_4 \mu^3
      \f}
  */
  class generic_quarks : public two_polytropes {
  
  public:
  
  generic_quarks(settings &s, ns_data &n) : two_polytropes(s,n) {
    }
    
    virtual ~generic_quarks() {}

    /** \brief Set parameter information [pure virtual]
     */
    virtual void get_param_info(std::vector<std::string> &names,
				std::vector<std::string> &units,
				ubvector &low, ubvector &high);

    /** \brief Compute the EOS corresponding to parameters in 
	\c e and put output in \c tab_eos
    */
    virtual void compute_eos(const ubvector &e, int &success,
			     std::ofstream &scr_out, model_data &dat);

    /** \brief Function to compute the initial guess
     */
    virtual void initial_point(ubvector &e);

  };

  /** \brief A strange quark star model from \ref Steiner13tn (4 parameters)

      Referred to as Model E in \ref Steiner13tn. 
  */
  class quark_star : public model {
  
  public:

    /** \brief Setup new parameters */
    virtual void setup_params(o2scl::cli &cl) {
      return;
    }

    /** \brief Remove model-specific parameters */
    virtual void remove_params(o2scl::cli &cl) {
      return;
    }

    /// The bag constant
    double B;

    /** \brief The paramter controlling non-perturbative corrections 
	to \f$ \mu^4 \f$
    */
    double c;

    /// The gap
    double Delta;

    /// The strange quark mass
    double ms;

    /// The solver to find the chemical potential for zero pressure
    o2scl::mroot_hybrids<> gmh;
    
    /// An alternative root finder
    o2scl::root_brent_gsl<> grb;
    
  quark_star(settings &s, ns_data &n) : model(s,n) {
    }

    virtual ~quark_star() {}
  
    /// Compute the pressure as a function of the chemical potential
    int pressure(size_t nv, const ubvector &x, ubvector &y);

    /// Compute the pressure as a function of the chemical potential
    double pressure2(double mu);

    /** \brief Set parameter information [pure virtual]
     */
    virtual void get_param_info(std::vector<std::string> &names,
				std::vector<std::string> &units,
				ubvector &low, ubvector &high);

    /** \brief Compute the EOS corresponding to parameters in 
	\c e and put output in \c tab_eos
    */
    virtual void compute_eos(const ubvector &e, int &success,
			     std::ofstream &scr_out, model_data &dat);

    /** \brief Function to compute the initial guess
     */
    virtual void initial_point(ubvector &e);

  };

  /** \brief Use QMC computations of neutron matter from
      \ref Steiner12cn (7 parameters)
	
      \ref Steiner12cn used a parameterization for neutron matter
      which is designed to fit results from quantum Monte Carlo (QMC)
      simulations in \ref Gandolfi12mm . The parameterization is
      \f[
      E_{\mathrm{neut}} = a \left( \frac{n_B}{n_0} \right)^{\alpha}
      + b \left( \frac{n_B}{n_0} \right)^{\beta}
      \f]
      where \f$ E_{\mathrm{neut}} \f$ is the energy per particle in
      neutron matter, \f$ n_B \f$ is the baryon number density, and
      \f$ n_0 \equiv 0.16~\mathrm{fm}^{-3} \f$ is the saturation
      density. The parameter ranges are
      \f{eqnarray*}
      a &=& 13 \pm 0.3~\mathrm{MeV} \nonumber \\
      \alpha &=& 0.50 \pm 0.02 \nonumber \\
      b &=& 3 \pm 2~\mathrm{MeV} \nonumber \\
      \beta &=& 2.3 \pm 0.2 \, .
      \f}
      At high density polytropes are used in a way similar to that in
      \ref bamr::two_polytropes. The transition between neutron matter
      and the first polytrope is at a baryon density specified in \ref
      rho_trans. The remaining 3 parameters are <tt>index1</tt>,
      <tt>trans1</tt>, and <tt>index2</tt>. In \ref Steiner12cn, the
      polytrope indices are between 0.2 and 2.0. The upper limit on
      polytropic indices has since been changed from 2.0 to 4.0. The
      transition between the first and second polytrope at the energy
      density in <tt>trans1</tt> which is between 2.0 and 8.0 \f$
      \mathrm{fm}^{-4} \f$. 

      \comment
      Note that since the QMC model provides an EOS for neutron
      matter at low densities, the crust EOS is taken from 
      the QMC results as well, ignoring the modification in 
      the EOS from nuclei. 
      2/2/16 - This is wrong. A crust is used (as stated in 
      the paper. The crust EOS is set for the eos_tov_interp
      object in bamr.cpp). 
      \endcomment
  */
  class qmc_neut : public model {

  public:
  
    qmc_neut(settings &s, ns_data &n);
    
    virtual ~qmc_neut();
    
    /// Saturation density in \f$ \mathrm{fm}^{-3} \f$
    double rho0;

    /// Transition density (default 0.48)
    double rho_trans;

    /// Ratio interpolation object
    o2scl::interp_vec<> si;

    /// Ratio error interpolation object
    o2scl::interp_vec<> si_err;
  
    /// \name Interpolation objects
    //@{
    ubvector ed_corr;
    ubvector pres_corr;
    ubvector pres_err;
    //@}

    /// Gaussian distribution for proton correction factor
    o2scl::prob_dens_gaussian pdg;

    /** \brief Set parameter information [pure virtual]
     */
    virtual void get_param_info(std::vector<std::string> &names,
				std::vector<std::string> &units,
				ubvector &low, ubvector &high);
    
    /** \brief Compute the EOS corresponding to parameters in 
        \c e and put output in \c tab_eos
    */
    virtual void compute_eos(const ubvector &e, int &success,
			     std::ofstream &scr_out, model_data &dat);

    /** \brief Function to compute the initial guess
     */
    virtual void initial_point(ubvector &e);
  };
  
  /** \brief QMC + three polytropes created for \ref Steiner15un
      (9 parameters)

      This model was also used in \ref Fryer15tf, \ref Nattila15eo,
      and \ref Steiner16ns .
      
      For neutron-rich matter near the saturation density, this class
      uses the QMC parameterization from \ref Steiner12cn as in \ref
      qmc_neut. The parameter ranges for for \f$ a \f$ and \f$ \alpha
      \f$ are expanded and \f$ b \f$ and \f$ \beta \f$ are recast in
      terms of \f$ S \f$ and \f$ L \f$.
      \f{eqnarray*}
      a &=& 12.5~\mathrm{to}~13.5~\mathrm{MeV} \nonumber \\
      \alpha &=& 0.47~\mathrm{to}~0.53 \nonumber \\
      S &=& 29.5~\mathrm{to}~36.1~\mathrm{MeV}\nonumber \\
      L &=& 30~\mathrm{to}~70~\mathrm{MeV}
      \f}
      The correlation between \f$ S \f$ and \f$ L \f$ defined
      by 
      \f[
      L < \left(\frac{9.17}{\mathrm{MeV}}\right) S - 266~\mathrm{MeV} 
      \quad \mathrm{and} \quad
      L > \left(\frac{14.3}{\mathrm{MeV}}\right) S - 379~\mathrm{MeV}
      \f]
      from \ref Lattimer14co is enforced. Alternatively, 
      expressing these constraints in \f$ (S,L) \f$ space, 
      are between the line through (29,0) and (35,55)
      and the line through (26.5,0) and (33.5,100) .
      
      Three polytropes are added at high density similar to \ref
      bamr::two_polytropes and \ref bamr::qmc_neut based on five
      parameters <tt>index1</tt>, <tt>trans1</tt>, <tt>index2</tt>,
      <tt>trans2</tt>, and <tt>index3</tt>. The transition between
      neutron matter and the first polytrope is at a baryon density
      specified in \ref rho_trans. The transition between the first
      and second polytrope is specified in <tt>trans1</tt>, and the
      transition between the second and third polytrope is specified
      in <tt>trans2</tt>. The polytropic indices are allowed to be
      between 0.2 and 8.0 and the transition densities are allowed to
      be between 0.75 and 8.0 \f$ \mathrm{fm}^{-4} \f$. 

      \comment
      Note that since the QMC model provides an EOS for neutron
      matter at low densities, the crust EOS is taken from 
      the QMC results as well, ignoring the modification in 
      the EOS from nuclei. 
      2/2/16 - This is wrong. A crust is used (as stated in 
      the paper. The crust EOS is set for the eos_tov_interp
      object in bamr.cpp). 
      \endcomment
  */
  class qmc_threep : public model {

  public:
  
    qmc_threep(settings &s, ns_data &n);
    
    virtual ~qmc_threep();
    
    /// Saturation density in \f$ \mathrm{fm}^{-3} \f$
    double rho0;

    /// Transition density (default 0.16, different than \ref bamr::qmc_neut)
    double rho_trans;

    /** \brief Set parameter information [pure virtual]
     */
    virtual void get_param_info(std::vector<std::string> &names,
				std::vector<std::string> &units,
				ubvector &low, ubvector &high);
    
    /** \brief Compute the EOS corresponding to parameters in 
	\c e and put output in \c tab_eos
    */
    virtual void compute_eos(const ubvector &e, int &success,
			     std::ofstream &scr_out, model_data &dat);

    /** \brief Function to compute the initial guess
     */
    virtual void initial_point(ubvector &e);
  
  };

  /** \brief QMC + line segments model created for \ref Steiner15un
      (8 parameters)

      This model was also used in \ref Fryer15tf, \ref Nattila15eo,
      and \ref Steiner16ns .

      This EOS model is similar to \ref bamr::qmc_threep, except that
      the high-density EOS is a set of line-segments, similar to \ref
      bamr::fixed_pressure. The transition between neutron matter
      from the QMC parameterization and the first line segment is
      set to a baryon density of \ref nb_trans . The energy density
      at this transition density is referred to as <tt>ed_trans</tt>,
      and the corresponding pressure is <tt>pr_trans</tt>.
      The four high-density parameters <tt>pres1</tt> through
      <tt>pres4</tt> are then defined by
      \f[
      P(\mathrm{ed1}) - \mathrm{pr\_trans} = \mathrm{pres1};
      \quad
      P(\mathrm{ed2}) - P(\mathrm{ed1}) = \mathrm{pres2};
      \quad
      P(\mathrm{ed3}) - P(\mathrm{ed2}) = \mathrm{pres3};
      \quad
      P(\mathrm{ed4}) - P(\mathrm{ed3}) = \mathrm{pres4}
      \f]
      where the energy density grid is set by the class members
      <tt>ed1</tt>, <tt>ed2</tt>, <tt>ed3</tt>, and <tt>ed4</tt>. The
      lower limits on parameters <tt>pres1</tt> through <tt>pres4</tt>
      are all zero. The upper limit on <tt>pres1</tt> is \f$
      0.3~\mathrm{fm}^{-4} \f$. The upper limits on the remaining
      pressure parameters are set so that the EOS is not acausal (even
      though causality is separately double-checked by the code in
      bamr.cpp anyway). 
      
      The limits on the high-density EOS
      parameters are the same as those in \ref bamr::fixed_pressure.

      \comment
      Note that since the QMC model provides an EOS for neutron
      matter at low densities, the crust EOS is taken from 
      the QMC results as well, ignoring the modification in 
      the EOS from nuclei. 
      2/2/16 - This is wrong. A crust is used (as stated in 
      the paper. The crust EOS is set for the eos_tov_interp
      object in bamr.cpp). 
      \endcomment
  */
  class qmc_fixp : public model {

  public:
  
    qmc_fixp(settings &s, ns_data &n);
    
    virtual ~qmc_fixp();

    /// \name The energy densities which define the grid
    //@{
    double ed1;
    double ed2;
    double ed3;
    double ed4;
    //@}
    
    /// Saturation density in \f$ \mathrm{fm}^{-3} \f$
    double nb0;

    /** \brief Transition baryon density (default 0.16, different 
	than \ref bamr::qmc_neut)
    */
    double nb_trans;

    /** \brief Set parameter information [pure virtual]
     */
    virtual void get_param_info(std::vector<std::string> &names,
				std::vector<std::string> &units,
				ubvector &low, ubvector &high);
    
    /** \brief Compute the EOS corresponding to parameters in 
	\c e and put output in \c tab_eos
    */
    virtual void compute_eos(const ubvector &e, int &success,
			     std::ofstream &scr_out, model_data &dat);

    /** \brief Function to compute the initial guess
     */
    virtual void initial_point(ubvector &e);
  
  };
  
  /** \brief QMC plus two line segments with arbitrary energy densities
      (8 parameters)

      \comment
      Note that since the QMC model provides an EOS for neutron
      matter at low densities, the crust EOS is taken from 
      the QMC results as well, ignoring the modification in 
      the EOS from nuclei. 
      2/2/16 - This is wrong. A crust is used (as stated in 
      the paper. The crust EOS is set for the eos_tov_interp
      object in bamr.cpp). 
      \endcomment
  */
  class qmc_twolines : public model {

  public:
  
    qmc_twolines(settings &s, ns_data &n);
    
    virtual ~qmc_twolines();

    /// Saturation density in \f$ \mathrm{fm}^{-3} \f$
    double nb0;

    /// Transition density (default 0.16, different than \ref bamr::qmc_neut)
    double nb_trans;

    /** \brief Set parameter information [pure virtual]
     */
    virtual void get_param_info(std::vector<std::string> &names,
				std::vector<std::string> &units,
				ubvector &low, ubvector &high);
    
    /** \brief Compute the EOS corresponding to parameters in 
	\c e and put output in \c tab_eos
    */
    virtual void compute_eos(const ubvector &e, int &success,
			     std::ofstream &scr_out, model_data &dat);

    /** \brief Function to compute the initial guess
     */
    virtual void initial_point(ubvector &e);
  
  };


}

#endif
