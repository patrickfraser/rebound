/**
 * @file 	problem.c
 * @brief 	Example problem: shearing sheet.
 * @author 	Hanno Rein <hanno@hanno-rein.de>
 * @detail 	This example simulates a small patch of Saturn's
 * Rings in shearing sheet coordinates. If you have OpenGL enabled, 
 * you'll see one copy of the computational domain. Press `g` to see
 * the ghost boxes which are used to calculate gravity and collisions.
 * Particle properties resemble those found in Saturn's rings. 
 *
 *
 * @section 	LICENSE
 * Copyright (c) 2011 Hanno Rein, Shangfei Liu
 *
 * This file is part of rebound.
 *
 * rebound is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * rebound is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with rebound.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include "main.h"
#include "particle.h"
#include "boundaries.h"
#include "output.h"
#include "communication_mpi.h"
#include "tree.h"
#include "tools.h"
#include "display.h"

extern double OMEGA;
extern double minimum_collision_velocity;
double moon_radius = 3*4.961;

extern double (*coefficient_of_restitution_for_velocity)(double); 
double coefficient_of_restitution_bridges(double v); 

extern double opening_angle2;

void problem_init(int argc, char* argv[]){
	// Setup constants
#ifdef GRAVITY_TREE
	opening_angle2	= .5;					// This determines the precission of the tree code gravity calculation.
#endif // GRAVITY_TREE
	OMEGA 				= 1.22e-4;      	// 1/s
	G 				= 6.67428e-11;		// N / (1e-5 kg)^2 m^2
	softening 			= 0.1;			// m
	dt 				= 1e-3*2.*M_PI/OMEGA;	// s
#ifdef OPENGL							// Delete the next two lines if you want a face on view.
	display_rotate_z		= 20;			// Rotate the box by 20 around the z axis, then 
	display_rotate_x		= 60;			// rotate the box by 60 degrees around the x axis	
#ifdef LIBPNG
	system("mkdir png");
#endif // LIBPNG
#endif // OPENGL
	// This example uses two root boxes in the x and y direction. 
	// Although not necessary in this case, it allows for the parallelization using MPI. 
	// See Rein & Liu for a description of what a root box is in this context.
	root_nx = 6; root_ny = 6; root_nz = 1;			
	nghostx = 0; nghosty = 2; nghostz = 0; 			// Use two ghost rings
	double surfacedensity 		= 283.0; 			// kg/m^2
	double particle_density		= 200;			// kg/m^3
	double particle_radius_min 	= 1;			// m
	double particle_radius_max 	= 4;			// m
	double particle_radius_slope 	= -3;	
	boxsize 			= 400;			// m
	if (argc>1){						// Try to read boxsize from command line
		boxsize = atof(argv[1]);
	}
	init_box();
	
	// Initial conditions
	printf("Toomre wavelength: %f\n",4.*M_PI*M_PI*surfacedensity/OMEGA/OMEGA*G);
	// Use Bridges et al coefficient of restitution.
	coefficient_of_restitution_for_velocity = coefficient_of_restitution_bridges;
	// When two particles collide and the relative velocity is zero, the might sink into each other in the next time step.
	// By adding a small repulsive velocity to each collision, we prevent this from happening.
	minimum_collision_velocity = particle_radius_min*OMEGA*0.001;  // small fraction of the shear accross a particle


	// Hanno: Add all ring paricles
	double total_mass = surfacedensity*boxsize_x*boxsize_y/5.;
	double mass = 0;
	while(mass<total_mass){
		struct particle pt;
		pt.x 		= tools_uniform(boxsize_x/10.,boxsize_x*(3./10.));
		pt.y 		= tools_uniform(-boxsize_y/2.,boxsize_y/2.);
		pt.z 		= tools_normal(1.);			 // m
		pt.vx 		= 0;
		pt.vy 		= -1.5*(pt.x)*OMEGA;
		pt.vz 		= 0;
		pt.ax 		= 0;
		pt.ay 		= 0;
		pt.az 		= 0;
		double radius 	= tools_powerlaw(particle_radius_min,particle_radius_max,particle_radius_slope);
	#ifndef COLLISIONS_NONE
		pt.r 		= radius;			  // m
	#endif
		double		particle_mass = particle_density*4./3.*M_PI*radius*radius*radius;
		pt.m 		= particle_mass; 	// kg
		particles_add(pt);
		mass += particle_mass;
	}
	//-Moi: Add Daphnis
	struct particle ptd;
	ptd.x           = 0;
	ptd.y 		= 0;
	ptd.z 		= 0;		       // m
	ptd.vx 		= 0;
	ptd.vy 		= 0;
	ptd.vz 		= 0;
	ptd.ax 		= 0;
	ptd.ay 		= 0;
	ptd.az 		= 0;
	/////////////////////double radius 	= 8;
	//#ifndef COLLISIONS_NONE
	/////////////////////ptd.r 		= radius;	       // -m
	//#endif
	//double	       particle_mass = particle_density*4./3.*M_PI*radius*radius*radius;
	//double		particle_mass = 7.7e13;  //actual
	double          real2sim_ratio = 670.4; // M_Daphnis/Hill_radius in kg/m^3 
	double          hill_r = 0.231*boxsize_x/10.; // fraction of separation distance * radius of simulated Keeler gap    in m 
	//double		particle_mass = real2sim_ratio*hill_r*hill_r*hill_r;    //incorrectly filles entire hill radius with mass 
	double          daphnis_density = 340.0; // kg/m^3
	// for reference, moon_radius = pow(particle_mass/(4*M_PI*daphnis_density),0.333);
	//	double particle_mass = 8.262e6; //kg
	double sat_mass = 568.3e24; //kg
	double sep_dis = 136505500.0; //m - Separation distance (Saturn,daphnis)
	double particle_mass = 1.5*(hill_r/sep_dis)*(hill_r/sep_dis)*(hill_r/sep_dis)*3*sat_mass;
	ptd.m 		= particle_mass; 	// kg
	ptd.r = moon_radius;
	particles_add(ptd);
	
	// end of moi
}

// This example is using a custom velocity dependend coefficient of restitution
double coefficient_of_restitution_bridges(double v){
	// assumes v in units of [m/s]
	double eps = 0.32*pow(fabs(v)*100.,-0.234);
	if (eps>1) eps=1;
	if (eps<0) eps=0;
	return eps;
}

void problem_inloop(){
  for(int i=0; i<N; i++){
    if (particles[i].r == moon_radius){
      particles[i].x = 0;
      particles[i].y = 0;
      particles[i].z = 0;
    }
}

}

void problem_output(){
#ifdef LIBPNG
	if (output_check(1e-3*2.*M_PI/OMEGA)){
		output_png("png/");
	}
#endif //LIBPNG
	if (output_check(1e-3*2.*M_PI/OMEGA)){
	        output_timing();
		output_append_velocity_dispersion("veldisp.txt");
		output_binary_positions("binary_positions.bin");
		output_binary("binary_structs.bin");
	}
	if (output_check(0.05*M_PI/OMEGA)){
	  output_ascii("position.txt");
	  // output_orbits("orbital_parameters.txt");
	}
	if (output_check(2*M_PI/OMEGA)){
	  output_append_velocity_dispersion("veldisp_orb.txt");
	}

}


void problem_finish(){
}
