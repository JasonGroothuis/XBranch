-- Copyright 2003 Regents of the University of California

-- SETI_BOINC is free software; you can redistribute it and/or modify it under
-- the terms of the GNU General Public License as published by the Free
-- Software Foundation; either version 2, or (at your option) any later
-- version.

-- SETI_BOINC is distributed in the hope that it will be useful, but WITHOUT
-- ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
-- FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
-- more details.

-- You should have received a copy of the GNU General Public License along
-- with SETI_BOINC; see the file COPYING.  If not, write to the Free Software
-- Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.

-- In addition, as a special exception, the Regents of the University of
-- California give permission to link the code of this program with libraries
-- that provide specific optimized fast Fourier transform (FFT) functions and
-- distribute a linked executable.  You must obey the GNU General Public 
-- License in all respects for all of the code used other than the FFT library
-- itself.  Any modification required to support these libraries must be
-- distributed in source code form.  If you modify this file, you may extend 
-- this exception to your version of the file, but you are not obligated to 
-- do so. If you do not wish to do so, delete this exception statement from 
-- your version.


create database sah2b@sci_master_tcp in sah2dbs; -- informix


create row type coordinate_t 
(
   time float,
   ra float,
   dec float
);



create row type chirp_parameter_t
  (
     chirp_limit smallfloat,
     fft_len_flags integer 		-- bitfield
  );



create row type subband_description_t
  (
     number 	 integer,
     center 	 float,
     base 	 float,
     sample_rate float
  );



create row type data_description_t
  (
     start_ra	float,
     start_dec	float,
     end_ra	float,
     end_dec	float,
     true_angle_range	smallfloat,
     time_recorded varchar(255),
     time_recorded_jd float,
     nsamples	integer,
     coords	list(coordinate_t not null) 
  );


create table receiver_config
  (
    id serial primary key, 
    s4_id integer unique,
    name varchar(255) unique,
    beam_width smallfloat,     -- degrees 
    center_freq float,		-- MHz	
    latitude float,
    longitude float,
    elevation float,
    diameter smallfloat,
    az_orientation float,
    az_corr_coeff  list(float not null), 
    zen_corr_coeff list(float not null),
    array_az_ellipse float default 0 not null,
    array_za_ellipse float default 0 not null,
    array_angle float default 0 not null
  );



create table recorder_config
  (
     id serial primary key,
     name char(64),
     bits_per_sample integer,
     sample_rate float,
     beams integer,
     version smallfloat unique
  );



create table splitter_config
  (
    id serial primary key,
    version smallfloat,
    data_type char(64),
    fft_len integer,
    ifft_len integer,
    filter char(64),
    window char(64), 
    samples_per_wu integer default 1048576 not null,             
    highpass smallfloat default 0 not null,             
    blanker_filter char(64) default "none" not null
  );




create table analysis_config
  (
     id serial primary key,
     spike_thresh smallfloat,
     spikes_per_spectrum integer,
     gauss_null_chi_sq_thresh smallfloat,
     gauss_chi_sq_thresh smallfloat,
     gauss_power_thresh smallfloat,
     gauss_peak_power_thresh smallfloat,
     gauss_pot_length integer,
     pulse_thresh smallfloat,
     pulse_display_thresh smallfloat,
     pulse_max integer,
     pulse_min integer,
     pulse_fft_max integer,
     pulse_pot_length integer,
     triplet_thresh smallfloat,
     triplet_max integer,
     triplet_min integer,
     triplet_pot_length integer,
     pot_overlap_factor smallfloat,
     pot_t_offset smallfloat,
     pot_min_slew smallfloat,      
     pot_max_slew smallfloat,      
     chirp_resolution float,
     analysis_fft_lengths integer,	-- bitfield
     bsmooth_boxcar_length integer,
     bsmooth_chunk_size integer,
     chirps list(chirp_parameter_t not null),   
     pulse_beams smallfloat,
     max_signals integer,
     max_spikes integer,
     max_gaussians integer,
     max_pulses integer,
     max_triplets integer,
     keyuniq integer,
     credit_rate smallfloat
  );


create table science_config 
  (
    id  serial primary key,                  
    active integer not null,               
    qpix_scheme char(16) not null,       
    qpix_nside integer not null,          
    fpix_width float not null,          
    sky_disc_radius float not null,     
    epoch integer not null,               
    bary_freq_window integer not null,    
    nonbary_freq_window integer not null,
    gauss_obs_duration float not null,   
    gauss_obs_sep_time float not null,  
    pulse_obs_duration float not null,  
    pulse_obs_sep_time float not null,  
    min_app_version float not null    
  );


create row type candidate_t 
  (
    type integer not null,                  
    id int8 not null, 
    num_obs integer not null,              
    score float not null,
    is_rfi integer not null         -- bitfield          
  );                            


create table meta_candidate 
  (
    id serial8 not null,            
    version integer not null,        
    best_barycentric_score float not null,
    best_nonbarycentric_score float not null,
    meta_score float not null,
    rfi_clean integer,     
    candidates list(candidate_t not null)
  );


create table  multiplet 
  (
    id serial primary key,
    version integer not null,
    type integer not null,
    mean_ra float not null,
    mean_decl float not null,
    ra_rms float not null,
    decl_rms float not null,
    mean_frequency float not null,
    frequency_rms float not null,
    mean_chirp float not null,
    chirp_rms float not null,
    mean_period float not null,
    period_rms float not null,
    mean_snr float not null,
    snr_rms float not null,
    mean_threshold float not null,
    threshold_rms float not null,
    score float not null,
    num_detections integer not null,
    signal_ids list(int8 not null)
  );

create table  star 
  (
    id integer not null,
    object_type char(1) not null,
    catalog_name varchar(64),
    catalog_number integer,
    object_name varchar(64),
    ra smallfloat not null,
    decl smallfloat not null,
    qpix integer,
    v_mag smallfloat,
    b_minus_v smallfloat,
    parallax smallfloat,
    stellar_type varchar(32),
    score smallfloat      
  );
                             

create table tape 
  (
    id serial primary key,
    name char(20) not null ,
    start_time float not null ,
    last_block_time float not null ,
    last_block_done integer not null ,
    missed integer not null ,
    tape_quality integer,
    beam integer default 0 not null
--	,
 --   unique (name,beam) constraint uniq_namebeam
  );



create table settings (
    id serial primary key,
    active integer,
    recorder_cfg integer references recorder_config,
    splitter_cfg integer references splitter_config,
    analysis_cfg integer references analysis_config,
    receiver_cfg integer references receiver_config
);

    


create table workunit_grp 
  (
    id serial primary key,
    tape_info integer not null ,			--references tape
    name char(64) not null ,
    data_desc data_description_t,
    receiver_cfg integer references receiver_config,
    recorder_cfg integer references recorder_config,
    splitter_cfg integer references splitter_config,
    analysis_cfg integer references analysis_config,
    sb_id integer
  );



create table workunit_header 
  (
    id serial8 primary key,
    name char(64) not null ,
    group_info integer not null ,   -- references workunit_grp
    subband_desc subband_description_t,
    sb_id int8
  )
    fragment by expression
      (mod(id,4)=0) in other_dbs001,
      (mod(id,4)=1) in other_dbs002,
      (mod(id,4)=2) in other_dbs003,
      (mod(id,4)=3) in other_dbs004
      extent size 209714 next size 204714;

create synonym workunit for workunit_header;



create table result 
  (
    id serial8 primary key,
    boinc_result int8 not null,
    wuid int8 not null ,		-- references workunit_header
    received float not null,
    hostid integer not null,
    versionid integer not null,
    return_code integer not null,
    overflow smallint not null,
    reserved integer not null,
    sb_id int8
  )
    fragment by expression
      (mod(id,4)=0) in other_dbs001, 
      (mod(id,4)=1) in other_dbs002, 
      (mod(id,4)=2) in other_dbs003, 
      (mod(id,4)=3) in other_dbs004 
      extent size 209714 next size 204714;

      

create table triplet 
  (
    id serial8 not null ,
    result_id int8,			-- references result
    peak_power smallfloat not null ,
    mean_power smallfloat not null ,
    time float not null ,
    ra smallfloat not null ,
    decl smallfloat not null ,
    q_pix int8 not null ,
    freq float not null ,
    detection_freq float not null,
    barycentric_freq float not null,
    fft_len integer not null ,
    chirp_rate smallfloat not null ,
    rfi_checked smallint,
    rfi_found smallint,
    reserved integer,
    period smallfloat not null 
  )
    fragment by expression
      (mod(id,4)=0) in other_dbs001,
      (mod(id,4)=1) in other_dbs002,
      (mod(id,4)=2) in other_dbs003,
      (mod(id,4)=3) in other_dbs004
      extent size 209714 next size 204714;


      
create table gaussian 
  (
    id serial8 not null ,
    result_id int8,			-- references result
    peak_power smallfloat not null ,
    mean_power smallfloat not null ,
    time float not null ,
    ra smallfloat not null ,
    decl smallfloat not null ,
    q_pix int8 not null ,
    freq float not null ,
    detection_freq float not null,
    barycentric_freq float not null,
    fft_len integer not null ,
    chirp_rate smallfloat not null ,
    rfi_checked smallint,
    rfi_found smallint,
    reserved integer,
    sigma smallfloat not null ,
    chisqr smallfloat not null ,
    null_chisqr smallfloat not null ,
    score smallfloat not null ,
    max_power smallfloat,
    pot byte                        -- binary
  ) 
    fragment by expression
      (mod(id,4)=0) in other_dbs001,
      (mod(id,4)=1) in other_dbs002,
      (mod(id,4)=2) in other_dbs003,
      (mod(id,4)=3) in other_dbs004
      extent size 209714 next size 204714;


      

create table pulse 
  (
    id serial8 not null ,
    result_id int8,			-- references result
    peak_power smallfloat not null ,
    mean_power smallfloat not null ,
    time float not null ,
    ra smallfloat not null ,
    decl smallfloat not null ,
    q_pix int8 not null ,
    freq float not null ,
    detection_freq float not null,
    barycentric_freq float not null,
    fft_len integer not null ,
    chirp_rate smallfloat not null ,
    rfi_checked smallint,
    rfi_found smallint,
    reserved integer,
    period smallfloat not null ,
    snr smallfloat not null ,
    thresh smallfloat not null ,
    score smallfloat not null ,
    len_prof smallint not null ,
    pot byte                   -- binary
  )
    fragment by expression
      (mod(id,4)=0) in other_dbs001,
      (mod(id,4)=1) in other_dbs002,
      (mod(id,4)=2) in other_dbs003,
      (mod(id,4)=3) in other_dbs004
      extent size 209714 next size 204714;


      

create table sah_pointing 
  (
    time_id integer not null ,
    time float not null ,
    ra float not null ,
    dec float not null ,
    q_pix integer not null,
    angle_range float not null ,
    bad smallint
  );


create table sky_map
  (
        npix                    int8,           -- the primary search key
        qpix                    int,            -- for fast spatial maps
        fpix                    int,            -- for fast frequency maps
        spike_max_id            int8 ,
        gaussian_max_id         int8,
        pulse_max_id            int8,
        triplet_max_id          int8,
        spike_count             int,
        gaussian_count          int,
        pulse_count             int,
        triplet_count           int,
        new_data                smallint,       -- a boolean
        score			float
   );

create table hotpix
   (
        id                      int,            -- qpix
        last_hit_time           int
   );


create table spike 
  (
    id serial8 not null ,
    result_id int8,			-- references result
    peak_power smallfloat not null ,
    mean_power smallfloat not null ,
    time float not null ,
    ra smallfloat not null ,
    decl smallfloat not null ,
    q_pix int8 not null ,
    freq float not null ,
    detection_freq float not null,
    barycentric_freq float not null,
    fft_len integer not null ,
    chirp_rate smallfloat not null ,
    rfi_checked smallint,
    rfi_found smallint,
    reserved integer
  )
    fragment by expression
      (mod(id,5)=0) in spike_dbs001, 
      (mod(id,5)=1) in spike_dbs002, 
      (mod(id,5)=2) in spike_dbs003, 
      (mod(id,5)=3) in spike_dbs004, 
      (mod(id,5)=4) in spike_dbs005
      extent size 2097140 next size 204714;



create table classic_versions
  (
   id serial not null,
   ver_major integer not null,
   ver_minor integer not null,
   platformid integer not null,
   comment char(254),
   filename char(254),
   md5_cksum char(254),
   sum_cksum char(254),
   cksum_cksum char(254),
   file_cksum integer not null
  );



create table classic_active_versions
  (
   id integer,
   versionid integer,
   ver_major integer,
   ver_minor integer
  );



create table classic_active_versionids
  (
   id integer,
   versionid  integer
  );


      
create index spike_res on spike(result_id);


create index gaussian_res on gaussian(result_id);


create index pulse_res on pulse(result_id);


create index triplet_res on triplet(result_id);


create index result_wu on result(wuid);


create index workunit_wu_grp on workunit(group_info);



create index wugrp_tapenum on workunit(tape_info);
alter table workunit_grp add constraint (foreign key 
    (tape_info) references tape (id));


create index wu_grpnum on workunit(group_info);
alter table workunit add constraint (foreign key (group_info) 
    references workunit_grp (id));


create index res_wuid on result(wuid);
alter table result add constraint (foreign key (wuid) 
    references workunit_header (id)),
    add constraint unique(boinc_result);


create index trip_res on triplet(result_id);
alter table triplet add constraint (foreign key (result_id) 
    references result (id));


crete index gauss-res on gaussian(result_id);
alter table gaussian add constraint (foreign key (result_id) 
    references result (id));


create index pulse_res on pulse(result_id);
alter table pulse add constraint (foreign key (result_id) 
    references result (id));


create index spike_res on spike(result_id);
alter table spike add constraint (foreign key (result_id) 
    references result (id) constraint result_spike);

create unique index namebeam on tape (name, beam);
    

