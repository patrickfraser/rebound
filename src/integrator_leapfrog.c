/**
 * @file 	integrator.c
 * @brief 	Leap-frog integration scheme.
 * @author 	Hanno Rein <hanno@hanno-rein.de>
 * @detail	This file implements the leap-frog integration scheme.  
 * This scheme is second order accurate, symplectic and well suited for 
 * non-rotating coordinate systems. Note that the scheme is formally only
 * first order accurate when velocity dependent forces are present.
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
#include "particle.h"
#include "main.h"
#include "gravity.h"
#include "tools.h"
#include "boundaries.h"

// These variables have no effect for leapfrog.
int integrator_force_is_velocitydependent 	= 1;
double integrator_epsilon 			= 0;
double integrator_min_dt 			= 0;

double integrator_megno_Ys;
double integrator_megno_Yss;
double integrator_megno_cov_Yt;	// covariance of <Y> and t
double integrator_megno_var_t;  // variance of t 
double integrator_megno_mean_t; // mean of t
double integrator_megno_mean_Y; // mean of Y
long   integrator_megno_n; 	// number of covariance updates
void integrator_megno_init(double delta){
	int _N_megno = N;
	integrator_megno_Ys = 0.;
	integrator_megno_Yss = 0.;
	integrator_megno_cov_Yt = 0.;
	integrator_megno_var_t = 0.;
	integrator_megno_n = 0;
	integrator_megno_mean_Y = 0;
	integrator_megno_mean_t = 0;
        for (int i=0;i<_N_megno;i++){ 
                struct particle megno = {
			.m  = particles[i].m,
			.x  = delta*tools_normal(1.),
			.y  = delta*tools_normal(1.),
			.z  = delta*tools_normal(1.),
			.vx = delta*tools_normal(1.),
			.vy = delta*tools_normal(1.),
			.vz = delta*tools_normal(1.) };
                particles_add(megno);
        }
	N_megno = _N_megno;
}
double integrator_megno(){ // Returns the MEGNO <Y>
	if (t==0.) return 0.;
	return integrator_megno_Yss/t;
}
double integrator_lyapunov(){ // Returns the largest Lyapunov characteristic number (LCN), or maximal Lyapunov exponent
	if (t==0.) return 0.;
	return integrator_megno_cov_Yt/integrator_megno_var_t;
}
double integrator_megno_deltad_delta2(){
        double deltad = 0;
        double delta2 = 0;
        for (int i=N-N_megno;i<N;i++){
                deltad += particles[i].vx * particles[i].x; 
                deltad += particles[i].vy * particles[i].y; 
                deltad += particles[i].vz * particles[i].z; 
                deltad += particles[i].ax * particles[i].vx; 
                deltad += particles[i].ay * particles[i].vy; 
                deltad += particles[i].az * particles[i].vz; 
                delta2 += particles[i].x  * particles[i].x; 
                delta2 += particles[i].y  * particles[i].y;
                delta2 += particles[i].z  * particles[i].z;
                delta2 += particles[i].vx * particles[i].vx; 
                delta2 += particles[i].vy * particles[i].vy;
                delta2 += particles[i].vz * particles[i].vz;
        }
        return deltad/delta2;
}
void integrator_megno_calculate_acceleration(){
#pragma omp parallel for schedule(guided)
	for (int i=N-N_megno; i<N; i++){
	for (int j=N-N_megno; j<N; j++){
		if (i==j) continue;
		const double dx = particles[i-N/2].x - particles[j-N/2].x;
		const double dy = particles[i-N/2].y - particles[j-N/2].y;
		const double dz = particles[i-N/2].z - particles[j-N/2].z;
		const double r = sqrt(dx*dx + dy*dy + dz*dz + softening*softening);
		const double r3inv = 1./(r*r*r);
		const double r5inv = 3./(r*r*r*r*r);
		const double ddx = particles[i].x - particles[j].x;
		const double ddy = particles[i].y - particles[j].y;
		const double ddz = particles[i].z - particles[j].z;
		const double Gm = G * particles[j].m;
		
		// Variational equations
		particles[i].ax += Gm * (
			+ ddx * ( dx*dx*r5inv - r3inv )
			+ ddy * ( dx*dy*r5inv )
			+ ddz * ( dx*dz*r5inv )
			);

		particles[i].ay += Gm * (
			+ ddx * ( dy*dx*r5inv )
			+ ddy * ( dy*dy*r5inv - r3inv )
			+ ddz * ( dy*dz*r5inv )
			);

		particles[i].az += Gm * (
			+ ddx * ( dz*dx*r5inv )
			+ ddy * ( dz*dy*r5inv )
			+ ddz * ( dz*dz*r5inv - r3inv )
			);
	}
	}
}

// Leapfrog integrator (Drift-Kick-Drift)
// for non-rotating frame.
void integrator_part1(){
#pragma omp parallel for schedule(guided)
	for (int i=0;i<N;i++){
		particles[i].x  += 0.5* dt * particles[i].vx;
		particles[i].y  += 0.5* dt * particles[i].vy;
		particles[i].z  += 0.5* dt * particles[i].vz;
	}
	t+=dt/2.;
}
void integrator_part2(){
	integrator_megno_calculate_acceleration();
#pragma omp parallel for schedule(guided)
	for (int i=0;i<N;i++){
		particles[i].vx += dt * particles[i].ax;
		particles[i].vy += dt * particles[i].ay;
		particles[i].vz += dt * particles[i].az;
		particles[i].x  += 0.5* dt * particles[i].vx;
		particles[i].y  += 0.5* dt * particles[i].vy;
		particles[i].z  += 0.5* dt * particles[i].vz;
	}
	t+=dt/2.;
	
	if (N_megno){
		double integrator_megno_thisdt = 2.* t * integrator_megno_deltad_delta2();
		// Calculate running Y(t)
		integrator_megno_Ys += dt*integrator_megno_thisdt;
		double Y = integrator_megno_Ys/t;
		// Calculate averge <Y> 
		integrator_megno_Yss += Y * dt;
		// Update covariance of (Y,t) and variance of t
		integrator_megno_n++;
		double _d_t = t - integrator_megno_mean_t;
		integrator_megno_mean_t += _d_t/(double)integrator_megno_n;
		double _d_Y = integrator_megno() - integrator_megno_mean_Y;
		integrator_megno_mean_Y += _d_Y/(double)integrator_megno_n;
		integrator_megno_cov_Yt += ((double)integrator_megno_n-1.)/(double)integrator_megno_n 
						*(t-integrator_megno_mean_t)
						*(integrator_megno()-integrator_megno_mean_Y);
		integrator_megno_var_t  += ((double)integrator_megno_n-1.)/(double)integrator_megno_n 
						*(t-integrator_megno_mean_t)
						*(t-integrator_megno_mean_t);
	}
}
	

