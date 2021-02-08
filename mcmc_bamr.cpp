/*
  -------------------------------------------------------------------
  
  Copyright (C) 2012-2020, Andrew W. Steiner
  
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
#include "mcmc_bamr.h"

#include <o2scl/vector.h>
#include <o2scl/hdf_io.h>

using namespace std;
using namespace o2scl;
// For I/O with HDF files
using namespace o2scl_hdf;
// For pi, pi^2, etc.
using namespace o2scl_const;
using namespace bamr;

mcmc_bamr::mcmc_bamr() {
  model_type="";
  set=std::make_shared<settings>();
  nsd=std::make_shared<ns_data>();

  bc_arr.resize(1);
  bc_arr[0]=new bamr_class;
  bc_arr[0]->set=set;
  bc_arr[0]->nsd=nsd;
}

int mcmc_bamr::train_emulator(std::string train_filename,
                              std::vector<std::string> &param_names) {
  
  Py_Initialize();
  PyRun_SimpleString("import sys");
  PyRun_SimpleString("sys.path.append('./')");

  // AWS: This next function is deprecated, so I'm not sure what
  // to do here
  //PyEval_InitThreads();
  //Py_DECREF(PyImport_ImportModule("threading"));

  // Import python module
  emulator_module=PyImport_ImportModule("emulator");
  if (emulator_module==0) {
    O2SCL_ERR2("Failed to import module in ",
               "mcmc_bamr::train_emulator().",o2scl::exc_einval);
  }

  // Copy parameter names to a python object. 
  train_param_names=PyList_New(param_names.size());
  for(size_t i=0;i<param_names.size();i++) {
    PyList_SetItem(train_param_names,i, 
		   PyUnicode_FromString(param_names[i].c_str()));
  }

  // Python class object
  emulator_class = PyObject_GetAttrString(emulator_module,"gp_emulator");
  if (emulator_class==0) {
    O2SCL_ERR2("Failed to import class in ",
               "mcmc_bamr::train_emulator().",o2scl::exc_einval);
  }

  // Create an instance of the gp_emulator class
  if(PyCallable_Check(emulator_class)) {
    emulator_instance=PyObject_CallObject(emulator_class,0);
  }

  // Set the number of sources
  emu_n_sources=PyLong_FromSize_t(nsd->n_sources);

  // Python arguments for the gp_emulator::train() function
  train_args=PyTuple_Pack(4,PyUnicode_FromString(train_filename.c_str()),
                          train_param_names,train_param_names,
                          emu_n_sources);
  
  // Call python training function with previously created arguments
  cout << "Calling train method." << endl;
  train_method=PyObject_GetAttrString(emulator_instance,"train");
  if (PyCallable_Check(train_method)) {
    PyObject *train_res=PyObject_CallObject(train_method,train_args);
    if (train_res==0) {
      O2SCL_ERR("Training method failed.",o2scl::exc_einval);
    }
  }
  cout << "Done with train method." << endl;

  // Get prediction function for later use
  predict_method=PyObject_GetAttrString(emulator_instance,"predict");
  
  return 0;
}

int mcmc_bamr::threads(std::vector<std::string> &sv, bool itive_com) {
  
  if (sv.size()==1) {
    cerr << "Number of threads not specified in 'threads'." << endl;
    return 1;							       
  }

  if (model_type.length()>0) {
    cerr << "Threads must be set before model." << endl;
    return 2;
  }
  
  size_t n_threads_old=n_threads;
  for(size_t i=0;i<n_threads_old;i++) {
    delete bc_arr[i];
  }
  n_threads=o2scl::stoszt(sv[1]);
  
  bc_arr.resize(n_threads);
  for(size_t i=0;i<n_threads;i++) {
    bc_arr[i]=new bamr_class;
    bc_arr[i]->set=set;
    bc_arr[i]->nsd=nsd;
    bc_arr[i]->n_threads=n_threads;
  }
  
  return 0;
}
  
void mcmc_bamr::file_header(o2scl_hdf::hdf_file &hf) {

  mcmc_para_cli::file_header(hf);
  
  model &m=*(bc_arr[0]->mod);
  
  hf.sets_vec("source_names",nsd->source_names);
  hf.sets_vec("source_fnames",nsd->source_fnames);
  hf.sets_vec("slice_names",nsd->slice_names);

  hf.set_szt("grid_size",set->grid_size);
  hf.set_szt("n_sources",nsd->n_sources);
  hf.sets("model",model_type);
  hf.setd("min_mass",set->min_mass);
  hf.setd("exit_mass",set->exit_mass);
  hf.setd("min_max_mass",set->min_max_mass);
  hf.setd("input_dist_thresh",set->input_dist_thresh);
  hf.seti("use_crust",set->use_crust);
  hf.seti("baryon_density",set->baryon_density);
  hf.seti("debug_load",set->debug_load);
  hf.seti("debug_eos",set->debug_eos);
  hf.seti("debug_star",set->debug_star);
  hf.seti("inc_baryon_mass",set->inc_baryon_mass);
  hf.seti("addl_quants",set->addl_quants);
  hf.setd("nb_low",set->nb_low);
  hf.setd("nb_high",set->nb_high);
  hf.setd("e_low",set->e_low);
  hf.setd("e_high",set->e_high);
  hf.setd("m_low",set->m_low);
  hf.setd("m_high",set->m_high);

  hdf_output(hf,m.nb_grid,"nb_grid");
  hdf_output(hf,m.e_grid,"e_grid");
  hdf_output(hf,m.m_grid,"m_grid");

  return;
}

int mcmc_bamr::mcmc_init() {

  if (this->verbose>=2) {
    std::cout << "(rank " << this->mpi_rank
	      << ") Start mcmc_bamr::mcmc_init()." << std::endl;
  }

  if (bc_arr.size()<1) {
    O2SCL_ERR("Object bc_arr invalid.",o2scl::exc_esanity);
  }
  model &m=*(bc_arr[0]->mod);
  
  // This ensures enough space for all the
  // default return values in models.h
  this->ret_value_counts.resize(this->n_threads);
  for(size_t it=0;it<this->n_threads;it++) {
    this->ret_value_counts[it].resize(21);
  }

  // Copy parameter values to all of the model objects
  for(size_t i=1;i<bc_arr.size();i++) {
    model &m2=*(bc_arr[i]->mod);
    m.copy_params(m2);
  }
  
  mcmc_para_cli::mcmc_init();

  // -----------------------------------------------------------
  // Make sure the settings are consistent

  // Does inc_baryon_mass also need baryon_density?
  if (set->inc_baryon_mass && !set->baryon_density) {
    scr_out << "Cannot use inc_baryon_mass=true with "
	    << "baryon_density=false." << endl;
    return exc_efailed;
  }
  if (set->compute_cthick && (!set->baryon_density || !set->use_crust)) {
    scr_out << "Cannot use compute_cthick=true with "
	    << "baryon_density=false or use_crust=false." << endl;
    return exc_efailed;
  }
  if (set->crust_from_L && (!m.has_esym || !set->use_crust ||
			    !set->baryon_density)) {
    scr_out << "crust_from_L: " << set->crust_from_L << std::endl;
    scr_out << "has_esym: " << m.has_esym << std::endl;
    scr_out << "use_crust: " << set->use_crust << std::endl;
    scr_out << "baryon_density: " << set->baryon_density << std::endl;
    scr_out << "Cannot use crust_from_L=true with a model which does not "
	    << "provide S and L\nor with use_crust=false or with "
	    << "baryon_density=false." << endl;
    return exc_efailed;
  }
  if (set->addl_quants && !set->inc_baryon_mass) {
    scr_out << "Cannot do additional quantities without including "
	    << "baryon mass." << endl;
    return exc_efailed;
  }

  if(set->apply_emu == false){

    // -----------------------------------------------------------
    // Add columns to table

    for(size_t i=0;i<nsd->n_sources;i++) {
      this->table->new_column(((std::string)"wgt_")+nsd->source_names[i]);
      if (!set->norm_max) {
        this->table->set_unit(((std::string)"wgt_")+nsd->source_names[i],
                              "1/km/Msun");
      }
    }
  
    // It is important here that all of these columns which store values
    // over a grid are either always positive or always negative,
    // because the code reports zero in the fill_line() function for
    // values beyond the end of the EOS or the M-R curve. 
    for(size_t i=0;i<nsd->n_sources;i++) {
      this->table->new_column(((std::string)"Rns_")+nsd->source_names[i]);
      this->table->set_unit(((std::string)"Rns_")+nsd->source_names[i],
                            "km");
    }
  
    for(size_t i=0;i<nsd->n_sources;i++) {
      this->table->new_column(((std::string)"Mns_")+nsd->source_names[i]);
      this->table->set_unit(((std::string)"Mns_")+nsd->source_names[i],
                            "Msun");
    }
  
    if (m.has_eos) {
      for(int i=0;i<set->grid_size;i++) {
        this->table->new_column(((string)"P_")+o2scl::itos(i));
        this->table->set_unit(((string)"P_")+o2scl::itos(i),
                              "1/fm^4");
      }
      for(int i=0;i<set->grid_size;i++) {
        this->table->new_column(((string)"cs2_")+o2scl::itos(i));
      }
    }
  
    for(int i=0;i<set->grid_size;i++) {
      this->table->new_column(((string)"R_")+o2scl::itos(i));
      this->table->set_unit(((string)"R_")+o2scl::itos(i),
                            "km");
      if (m.has_eos) {
        this->table->new_column(((string)"PM_")+o2scl::itos(i));
        this->table->set_unit(((string)"PM_")+o2scl::itos(i),
                              "1/fm^4");
      }
    }
    if (m.has_eos) {
      if (set->baryon_density) {
        for(int i=0;i<set->grid_size;i++) {
          this->table->new_column(((string)"Pnb_")+o2scl::itos(i));
          this->table->set_unit(((string)"Pnb_")+o2scl::itos(i),
                                "1/fm^4");
          this->table->new_column(((string)"EoA_")+o2scl::itos(i));
          this->table->set_unit(((string)"EoA_")+o2scl::itos(i),
                                "MeV");
        }
      }
      if (m.has_esym) {
        this->table->new_column("S");
        this->table->set_unit("S","1/fm");
        this->table->new_column("L");
        this->table->set_unit("L","1/fm");
      }
      this->table->new_column("R_max");
      this->table->set_unit("R_max","km");
      this->table->new_column("M_max");
      this->table->set_unit("M_max","Msun");
      this->table->new_column("P_max");
      this->table->set_unit("P_max","1/fm^4");
      this->table->new_column("e_max");
      this->table->set_unit("e_max","1/fm^4");
      if (set->baryon_density) {
        this->table->new_column("nb_max");
        this->table->set_unit("nb_max","1/fm^3");
      }
      for(size_t i=0;i<nsd->n_sources;i++) {
        this->table->new_column(((string)"ce_")+nsd->source_names[i]);
        this->table->set_unit(((string)"ce_")+nsd->source_names[i],
                              "1/fm^4");
      }
      if (set->baryon_density) {
        for(size_t i=0;i<nsd->n_sources;i++) {
          this->table->new_column(((string)"cnb_")+nsd->source_names[i]);
          this->table->set_unit(((string)"cnb_")+nsd->source_names[i],
                                "1/fm^3");
        }
        this->table->new_column("gm_nb1");
        this->table->set_unit("gm_nb1","Msun");
        this->table->new_column("r_nb1");
        this->table->set_unit("r_nb1","km");
        this->table->new_column("gm_nb2");
        this->table->set_unit("gm_nb2","Msun");
        this->table->new_column("r_nb2");
        this->table->set_unit("r_nb2","km");
        this->table->new_column("gm_nb3");
        this->table->set_unit("gm_nb3","Msun");
        this->table->new_column("r_nb3");
        this->table->set_unit("r_nb3","km");
        this->table->new_column("gm_nb4");
        this->table->set_unit("gm_nb4","Msun");
        this->table->new_column("r_nb4");
        this->table->set_unit("r_nb4","km");
        this->table->new_column("gm_nb5");
        this->table->set_unit("gm_nb5","Msun");
        this->table->new_column("r_nb5");
        this->table->set_unit("r_nb5","km");
      }
      if (set->compute_cthick) {
        this->table->new_column("nt");
        this->table->set_unit("nt","1/fm^3");
        this->table->new_column("Pt");
        this->table->set_unit("Pt","1/fm^4");
        for(int i=0;i<set->grid_size;i++) {
          this->table->new_column(((string)"CT_")+o2scl::itos(i));
          this->table->set_unit(((string)"CT_")+o2scl::itos(i),"km");
        }
      }
    }
    if (set->addl_quants) {
      for(int i=0;i<set->grid_size;i++) {
        this->table->new_column(((string)"MB_")+o2scl::itos(i));
        this->table->set_unit(((string)"MB_")+o2scl::itos(i),"Msun");
        this->table->new_column(((string)"BE_")+o2scl::itos(i));
        this->table->set_unit(((string)"BE_")+o2scl::itos(i),"Msun");

        this->table->new_column(((string)"I_")+o2scl::itos(i));
        this->table->set_unit(((string)"I_")+o2scl::itos(i),
                              "Msun*km^2");
        this->table->new_column(((string)"I_bar_")+o2scl::itos(i));
      
        this->table->new_column(((string)"Lambda_bar_")+o2scl::itos(i));
      }
    }

    if (nsd->source_fnames_alt.size()>0) {
      for(size_t i=0;i<nsd->n_sources;i++) {
        this->table->new_column(((std::string)"atm_")+o2scl::szttos(i));
      }
    }

    if (model_type==((string)"qmc_threep_ligo") ||
        model_type==((string)"tews_threep_ligo") ||
        model_type==((string)"tews_fixp_ligo") ||
        model_type==((string)"qmc_fixp_ligo")) {
      this->table->new_column("M_chirp");
      this->table->set_unit("M_chirp","Msun");
      this->table->new_column("m1");
      this->table->set_unit("m1","Msun");
      this->table->new_column("m2");
      this->table->set_unit("m2","Msun");
      this->table->new_column("R1");
      this->table->set_unit("R1","km");
      this->table->new_column("R2");
      this->table->set_unit("R2","km");
      this->table->new_column("I1");
      this->table->set_unit("I1","Msun*km^2");
      this->table->new_column("I2");
      this->table->set_unit("I2","Msun*km^2");
      this->table->new_column("I_bar1");
      this->table->new_column("I_bar2");
      this->table->new_column("Lambda1");
      this->table->new_column("Lambda2");
      this->table->new_column("Lambdat");
      this->table->new_column("del_Lambdat");    
      this->table->new_column("Lambda_rat");
      this->table->new_column("q6");
      this->table->new_column("Lambda_s");
      this->table->new_column("Lambda_a");
      this->table->new_column("Lambda_a_YY");
      this->table->new_column("C1");
      this->table->new_column("C2");
      this->table->new_column("tews_prob");
      this->table->new_column("ligo_prob");
      this->table->new_column("eta");
    }
  }
  
  // -----------------------------------------------------------
  // Make grids

  for(size_t i=0;i<n_threads;i++) {
    bc_arr[i]->mod->nb_grid=uniform_grid_end<double>
      (set->nb_low,set->nb_high,set->grid_size-1);
    bc_arr[i]->mod->e_grid=uniform_grid_end<double>
      (set->e_low,set->e_high,set->grid_size-1);
    bc_arr[i]->mod->m_grid=uniform_grid_end<double>
      (set->m_low,set->m_high,set->grid_size-1);
  }

  // -----------------------------------------------------------
  // Load data

  nsd->load_mc(this->scr_out,mpi_size,mpi_rank,set);

  // -----------------------------------------------------------
  // Setup filters
  
  for(size_t i=0;i<n_threads;i++) {
    bamr_class &bc=dynamic_cast<bamr_class &>(*(bc_arr[i]));
    bc.setup_filters();
  }

  // Read FFT cache
  
  if (set->cached_intsc) {
    for(size_t i=0;i<n_threads;i++) {
      bamr_class &bc=dynamic_cast<bamr_class &>(*(bc_arr[i]));
      hdf_file hfx;
      for(size_t ii=0;ii<nsd->n_sources;ii++) {
	string fname=((string)"data/cache/tg_")+szttos(ii)+"_0";
	hfx.open(fname);
	hdf_input(hfx,bc.fft_data[ii*2],"tg");
	hfx.close();
	fname=((string)"data/cache/tg_")+szttos(ii)+"_1";
	hfx.open(fname);
	hdf_input(hfx,bc.fft_data[ii*2+1],"tg");
	hfx.close();
      }
    }
  }

  if (model_type==((string)"qmc_threep_ligo") ||
      model_type==((string)"tews_threep_ligo") ||
      model_type==((string)"tews_fixp_ligo") ||
      model_type==((string)"qmc_fixp_ligo")) {
    hdf_file hfx;
    for(size_t i=0;i<n_threads;i++) {
      bamr_class &bc=dynamic_cast<bamr_class &>(*(bc_arr[i]));
      hfx.open("data/ligo/ligo_tg3_v4.o2");
      std::string name;
      hdf_input(hfx,bc.ligo_data_table,name);
      hfx.close();
    }
  }
  
  if (this->verbose>=2) {
    std::cout << "(rank " << this->mpi_rank
	      << ") End mcmc_bamr::mcmc_init()." << std::endl;
  }

  return 0;
}

int mcmc_bamr::set_model(std::vector<std::string> &sv, bool itive_com) {
  
  // We cannot use scr_out here because it isn't set until the call
  // to mcmc().
  if (sv.size()<2) {
    cerr << "Model name not given." << endl;
    return exc_efailed;
  }
  if (model_type==sv[1]) {
    cerr << "Model already set to " << sv[1] << endl;
    return 0;
  }
  if (model_type.length()>0) {
    bc_arr[0]->mod->remove_params(cl);
  }
  if (sv[1]==((string)"twop")) {
    for(size_t i=0;i<n_threads;i++) {
      std::shared_ptr<model> mnew(new two_polytropes(set,nsd));
      bc_arr[i]->mod=mnew;
      bc_arr[i]->model_type=sv[1];
    }
  } else if (sv[1]==((string)"altp")) {
    for(size_t i=0;i<n_threads;i++) {
      std::shared_ptr<model> mnew(new alt_polytropes(set,nsd));
      bc_arr[i]->mod=mnew;
      bc_arr[i]->model_type=sv[1];
    }
  } else if (sv[1]==((string)"fixp")) {
    for(size_t i=0;i<n_threads;i++) {
      std::shared_ptr<model> mnew(new fixed_pressure(set,nsd));
      bc_arr[i]->mod=mnew;
      bc_arr[i]->model_type=sv[1];
    }
  } else if (sv[1]==((string)"qstar")) {
    for(size_t i=0;i<n_threads;i++) {
      std::shared_ptr<model> mnew(new quark_star(set,nsd));
      bc_arr[i]->mod=mnew;
      bc_arr[i]->model_type=sv[1];
    }
  } else if (sv[1]==((string)"genq")) {
    for(size_t i=0;i<n_threads;i++) {
      std::shared_ptr<model> mnew(new generic_quarks(set,nsd));
      bc_arr[i]->mod=mnew;
      bc_arr[i]->model_type=sv[1];
    }
  } else if (sv[1]==((string)"qmc")) {
    for(size_t i=0;i<n_threads;i++) {
      std::shared_ptr<model> mnew(new qmc_neut(set,nsd));
      bc_arr[i]->mod=mnew;
      bc_arr[i]->model_type=sv[1];
    }
  } else if (sv[1]==((string)"qmc_threep")) {
    for(size_t i=0;i<n_threads;i++) {
      std::shared_ptr<model> mnew(new qmc_threep(set,nsd));
      bc_arr[i]->mod=mnew;
      bc_arr[i]->model_type=sv[1];
    }
  } else if (sv[1]==((string)"qmc_fixp")) {
    for(size_t i=0;i<n_threads;i++) {
      std::shared_ptr<model> mnew(new qmc_fixp(set,nsd));
      bc_arr[i]->mod=mnew;
      bc_arr[i]->model_type=sv[1];
    }
  } else if (sv[1]==((string)"qmc_twolines")) {
    for(size_t i=0;i<n_threads;i++) {
      std::shared_ptr<model> mnew(new qmc_twolines(set,nsd));
      bc_arr[i]->mod=mnew;
      bc_arr[i]->model_type=sv[1];
    }
  } else if (sv[1]==((string)"tews_threep_ligo")) {
    for(size_t i=0;i<n_threads;i++) {
      std::shared_ptr<model> mnew(new tews_threep_ligo(set,nsd));
      bc_arr[i]->mod=mnew;
      bc_arr[i]->model_type=sv[1];
    }
  } else if (sv[1]==((string)"tews_fixp_ligo")) {
    for(size_t i=0;i<n_threads;i++) {
      std::shared_ptr<model> mnew(new tews_fixp_ligo(set,nsd));
      bc_arr[i]->mod=mnew;
      bc_arr[i]->model_type=sv[1];
    }
  } else {
    cerr << "Model unknown." << endl;
    return exc_efailed;
  }
  model_type=sv[1];
  bc_arr[0]->mod->setup_params(cl);
  return 0;
}

int mcmc_bamr::initial_point_last(std::vector<std::string> &sv,
				  bool itive_com) {

  if (sv.size()<2) {
    cerr << "Need a filename for initial_point_last()." << endl;
    return 1;
  }

  if (model_type.length()<2) {
    cerr << "Model not specified in initial_point_last()." << endl;
    return 2;
  }
      
  model &m=*(bc_arr[0]->mod);
  size_t np=m.n_eos_params+nsd->n_sources;
  
  string fname=sv[1];
  size_t pos=fname.find("<rank>");
  if (pos!=std::string::npos) {
    fname.replace(pos,6,o2scl::itos(mpi_rank));
  }
  this->initial_points_file_last(fname,np);
  
  return 0;
}

int mcmc_bamr::initial_point_best(std::vector<std::string> &sv,
				  bool itive_com) {
  
  if (sv.size()<2) {
    cerr << "Need a filename for initial_point_best()." << endl;
    return 1;
  }
      
  if (model_type.length()<2) {
    cerr << "Model not specified in initial_point_best()." << endl;
    return 2;
  }
  
  model &m=*(bc_arr[0]->mod);
  size_t np=m.n_eos_params+nsd->n_sources;
  
  string fname=sv[1];
  size_t pos=fname.find("<rank>");
  if (pos!=std::string::npos) {
    fname.replace(pos,6,o2scl::itos(mpi_rank));
  }
  this->initial_points_file_best(fname,np);
  
  return 0;
}

int mcmc_bamr::read_prev_results_mb(std::vector<std::string> &sv,
				    bool itive_com) {

  O2SCL_ERR("Not implemented yet.",o2scl::exc_eunimpl);
  
  if (sv.size()<2) {
    cerr << "Need a filename for read_prev_results_mb()." << endl;
    return 1;
  }

  if (model_type.length()<2) {
    cerr << "Model not specified in read_prev_results_mb()." << endl;
    return 2;
  }
  
  model &m=*(bc_arr[0]->mod);
  size_t np=m.n_eos_params+nsd->n_sources;
  
  // Ensure that multiple threads aren't reading from the 
  // filesystem at the same time
#ifdef BAMR_MPI
  int tag=0, buffer=0;
  if (mpi_size>1 && mpi_rank>=1) {
    MPI_Recv(&buffer,1,MPI_INT,mpi_rank-1,
	     tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
  }
#endif
  
  string fname=sv[1];
  size_t pos=fname.find("<rank>");
  if (pos!=std::string::npos) {
    fname.replace(pos,6,o2scl::itos(mpi_rank));
  }
  cout << "Rank " << mpi_rank
       << " is reading previous results from " << fname << " ." << endl;
  hdf_file hf;
  hf.open(fname);
  mcmc_para_table::read_prev_results(hf,np);
  hf.close();
  
#ifdef BAMR_MPI
  if (mpi_size>1 && mpi_rank<mpi_size-1) {
    MPI_Send(&buffer,1,MPI_INT,mpi_rank+1,
	     tag,MPI_COMM_WORLD);
  }
#endif

  return 0;
}

int mcmc_bamr::mcmc_func(std::vector<std::string> &sv, bool itive_com) {

  if (model_type.length()==0) {
    cerr << "Model not set in 'mcmc' command." << endl;
    return 1;
  }
  
  std::vector<std::string> names;
  std::vector<std::string> units;

  ubvector low;
  ubvector high;
  // Get upper and lower parameter limits and also the column names
  // and units for the data table (which also automatically includes
  // nuisance variables for the data points). The other columns and
  // units are specified in mcmc_init() function manually using a call
  // to table::new_column().
  bc_arr[0]->mod->get_param_info(names,units,low,high);

  if (set->apply_intsc) {

    // Ugly hack to increase the size of the 'low' and 'high' vectors
    ubvector low2(low.size()+nsd->n_sources);
    ubvector high2(low.size()+nsd->n_sources);
    vector_copy(low.size(),low,low2);
    vector_copy(high.size(),high,high2);

    for(size_t i=0;i<nsd->n_sources;i++) {
      names.push_back(((string)"log10_is_")+nsd->source_names[i]);
      units.push_back("");
      low2[i+low.size()]=-2.0;
      high2[i+high.size()]=2.0;
    }

    // Ugly hack, part 2
    low.resize(low2.size());
    high.resize(high2.size());
    vector_copy(low.size(),low2,low);
    vector_copy(high.size(),high2,high);
    
  }

  if (set->apply_emu) {
    
    // Ugly hack to increase the size of the 'low' and 'high' vectors
    ubvector low2(low.size()+nsd->n_sources);
    ubvector high2(low.size()+nsd->n_sources);
    vector_copy(low.size(),low,low2);
    vector_copy(high.size(),high,high2);
    
    for(size_t i=0;i<nsd->n_sources;i++) {
      names.push_back(((string)"atm_")+o2scl::szttos(i));
      units.push_back("");
      low2[i+low.size()]=0.0;
      high2[i+high.size()]=1.0;
    }
    
    // Ugly hack, part 2
    low.resize(low2.size());
    high.resize(high2.size());
    vector_copy(low.size(),low2,low);
    vector_copy(high.size(),high2,high);
    
  }
  
  
  set_names_units(names,units);
  
  // Set initial points if they have not already been set by the
  // user
  if (this->initial_points.size()==0) {
    // Get the parameter initial values for this model 
    ubvector init(names.size());
    bc_arr[0]->mod->initial_point(init);

    if (set->apply_intsc) {
      for(size_t i=0;i<nsd->n_sources;i++) {
	init[i+bc_arr[0]->mod->n_eos_params+nsd->n_sources]=-0.5;
      }
    }
    
    // AWS: 3/20/18: I changed this part, because we want the MCMC class
    // to be able to expect that different entries in the initial_points
    // array
    this->initial_points.clear();
    this->initial_points.push_back(init);
  }
  
  if (model_type==((string)"tews_threep_ligo")) {
    names[10]="q";
  }
  
  if (model_type==((string)"tews_fixp_ligo")) {
    names[9]="q";
  }
  
  vector<bamr::point_funct> pfa(n_threads);
  vector<bamr::fill_funct> ffa(n_threads);
  for(size_t i=0;i<n_threads;i++) {
    pfa[i]=std::bind
      (std::mem_fn<int(const ubvector &,ofstream &,double &,model_data &)>
       (&bamr_class::compute_point),bc_arr[i],std::placeholders::_2,
       std::ref(scr_out),std::placeholders::_3,std::placeholders::_4);
    ffa[i]=std::bind
      (std::mem_fn<int(const ubvector &,double,vector<double> &,
		       model_data &)>
       (&bamr_class::fill),bc_arr[i],std::placeholders::_1,
       std::placeholders::_2,std::placeholders::_3,std::placeholders::_4);
  }
  
  if (set->apply_emu) {
    cout << "Training emulator." << endl;
    
    // Train the emulator
    int pinfo=train_emulator(set->emu_train,names);
    if (pinfo!=0) {
      O2SCL_ERR2("Training emulator failed in ",
                 "mcmc_bamr::mcmc_func().",o2scl::exc_efailed);
    }

    // Copy trained method to bint classes
    for(size_t i=0;i<n_threads;i++){
      bamr_class &bc=dynamic_cast<bamr_class &>(*(bc_arr[i]));

      // copy pyobject to bint class
      bc.emulator_module=emulator_module;
      bc.emulator_class=emulator_class;
      bc.emulator_instance=emulator_instance;
      bc.train_method=train_method;
      bc.predict_method=predict_method;
      bc.train_param_names=train_param_names;
      bc.emu_n_sources=emu_n_sources;
    }

    // Delete unnecessary PyObjects
    Py_DECREF(emulator_module);
    Py_DECREF(emulator_instance);
    Py_DECREF(emulator_class);
  }

  // Perform the MCMC simulation
  this->mcmc_fill(names.size(),low,high,pfa,ffa);
  
  if (set->apply_emu) {
    Py_Finalize();
  }
  
  return 0;
}

int mcmc_bamr::add_data(std::vector<std::string> &sv, bool itive_com) {
  nsd->add_data(sv,itive_com);
  return 0;
}

int mcmc_bamr::add_data_alt(std::vector<std::string> &sv, bool itive_com) {
  nsd->add_data_alt(sv,itive_com);
  return 0;
}

void mcmc_bamr::setup_cli_mb() {
  
  mcmc_para_cli::setup_cli(cl);

  set->setup_cli(cl);
  
  // ---------------------------------------
  // Set options
    
  static const int nopt=8;
  comm_option_s options[nopt]=
    {
     {'m',"mcmc","Perform the Markov Chain Monte Carlo simulation.",
      0,0,"",((std::string)"This is the main part of ")+
      "the code which performs the simulation. Make sure to set the "+
      "model first using the 'model' command first.",
      new o2scl::comm_option_mfptr<mcmc_bamr>(this,&mcmc_bamr::mcmc_func),
      o2scl::cli::comm_option_both},
     {'o',"model","Choose model.",
      1,1,"<model name>",((string)"Choose the EOS parameterization model. ")+
      "Possible values are 'twop', 'altp', 'fixp', 'genq', 'qstar', "+
      "'qmc', 'qmc_threep' ,'qmc_fixp', and 'qmc_twolines'. A "+
      "model must be chosen before a MCMC run.",
      new comm_option_mfptr<mcmc_bamr>(this,&mcmc_bamr::set_model),
      cli::comm_option_both},
     {0,"threads","Specify number of OpenMP threads",
      1,1,"<number>","",
      new comm_option_mfptr<mcmc_bamr>(this,&mcmc_bamr::threads),
      cli::comm_option_both},
     {'a',"add-data","Add data source to the list.",
      4,5,"<name> <file> <slice> <initial mass> [obj name]",
      ((string)"Specify data as a table3d object in a HDF5 file. ")+
      "The string <name> is the name used, <file> is the filename, "+
      "<slice> is the name of the slice in the table3d object, "+
      "<initial mass> is the initial mass for the first point, and "+
      "[obj name] is the optional name of table3d object in <file>. "+
      "If [obj name] is not specified, then the first table3d object "+
      "is used.",new comm_option_mfptr<mcmc_bamr>
      (this,&mcmc_bamr::add_data),
      cli::comm_option_both},
     {0,"add-data-alt","Add data source to the list.",
      5,6,"<name> <file> <alt file> <slice> <initial mass> [obj name]",
      ((string)"Specify data as a table3d object in two HDF5 files. ")+
      "The string <name> is the name used, <file> and <alt file> are "+
      "the filenames, "+
      "<slice> is the name of the slice in the table3d object, "+
      "<initial mass> is the initial mass for the first point, and "+
      "[obj name] is the optional name of table3d object in <file>. "+
      "If [obj name] is not specified, then the first table3d object "+
      "is used.",new comm_option_mfptr<mcmc_bamr>
      (this,&mcmc_bamr::add_data_alt),
      cli::comm_option_both},
     {0,"initial-point-last","Set initial point from file.",1,1,
      "<filename>","Long. desc.",
      new o2scl::comm_option_mfptr<mcmc_bamr>
      (this,&mcmc_bamr::initial_point_last),
      o2scl::cli::comm_option_both},
     {0,"initial-point-best","Set initial point from file.",1,1,
      "<filename>","Long. desc.",
      new o2scl::comm_option_mfptr<mcmc_bamr>
      (this,&mcmc_bamr::initial_point_best),
      o2scl::cli::comm_option_both},
     {0,"read-prev-results","Read previous results from file (unfinished).",
      1,1,"<filename>","Long. desc.",
      new o2scl::comm_option_mfptr<mcmc_bamr>
      (this,&mcmc_bamr::read_prev_results_mb),
      o2scl::cli::comm_option_both}
    };
  cl.set_comm_option_vec(nopt,options);

  // --------------------------------------------------------
  
  return;
}

