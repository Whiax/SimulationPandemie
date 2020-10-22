#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Import
# =============================================================================
from collections import defaultdict, OrderedDict
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import scipy.spatial as spatial
from os.path import exists
import numpy as np
import argparse
import sys
import os

# =============================================================================
# Argparse some parameters
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("-seed", "--seed", default=0, type=int)
parser.add_argument("-type", "--type", default='default', type=str)
parser.add_argument("-distanciation", "--distanciation", default=16, type=float)
parser.add_argument("-n_population", "--n_population", default=1, type=int)
parser.add_argument("-transfer_rate", "--transfer_rate", default=0.0, type=float)
parser.add_argument("-transfer_proportion", "--transfer_proportion", default=0.01, type=float)
parser.add_argument("-curfew", "--curfew", default=24, type=int)
parser.add_argument("-confined", "--confined", default=0.0, type=float)
args = parser.parse_args() #"--seed 20 --distanciation 0.0".split(' ')
seed = args.seed

#define simulation name
simulation_name = f'fix4_{args.type}_{seed}_{args.distanciation}_{args.transfer_rate}_{args.curfew}_{args.confined}'
if exists(f'simulation/{simulation_name}/logs.pydict'):
    sys.exit()
print('Simulation name:', simulation_name)

# =============================================================================
# Simulation parameters - Starting hypotheses
# -----------------------------------------------------------------------------
# n_ is an absolute number
# p_ is a probability/proportion with 1 = 100%
# =============================================================================
#primary parameters
n_hours = 2 * 30 * 24 #2 months
n_population = args.n_population #number of densed population / cities
n_persons_per_population = 1000
p_contaminated_per_population = 0.01 #init proportion of contaminated people
distance_to_pcontamination = OrderedDict({0.1:0.95, 0.5:0.9, 1:0.7, 2:0.6, 5:0.3}) #probability is applied each hour
starting_distanciation = args.distanciation #meters | "density"
default_movement = 2 #meters per hour
contamination_duration = 14 * 24 #hours
delay_before_infectious = 3 * 24 #hours
death_possibility_delay = 9*24 #hours
p_lethality = 0.006
p_lethality_per_hour = 0.006/(contamination_duration-death_possibility_delay)
wake_up_time = 8
sleep_up_time = 24
curfew = False if args.curfew == 24 else args.curfew
sleep_up_time = sleep_up_time if not curfew else curfew
moving_time = list(range(wake_up_time, sleep_up_time))
p_confined  = args.confined #proportion of people not moving each day
probability_population_transfer = args.transfer_rate 
proportion_population_transfer = args.transfer_proportion

#secondary parameters
move_before_start = 50
init_delta_xy = 3  #+/- n meters initializing the population
mean_hours_since_contamination = 3*24; 
std_hours_since_contamination = 2*24

#non-parameters
np.random.seed(seed)
VULNERABLE = 0
IMMUNIZED = -1
DEAD = -2
plt.ioff()
colors = ['black', 'limegreen', 'dodgerblue', 'tomato']
simulation_dir = f'simulation/{simulation_name}'

#check
assert death_possibility_delay < contamination_duration
assert sleep_up_time > wake_up_time
assert VULNERABLE > IMMUNIZED > DEAD
assert not (p_confined > 0 and probability_population_transfer > 0), 'currently not compatible'
if not exists(simulation_dir):
    os.mkdir(simulation_dir)



# =============================================================================
# Generate populations
# -----------------------------------------------------------------------------
# generate populations in a grid pattern, each person is separated by starting_distanciation
# =============================================================================
populations_yx = []
for i_pop in range(0, n_population):
    border = int(np.sqrt(n_persons_per_population))+1
    xpos = [int(i/border) for i in list(range(0,n_persons_per_population))]
    ypos = list(range(0, border)) * border
    ypos = ypos[0:n_persons_per_population]
    population = np.array((ypos, xpos), dtype=np.float)
    population *= starting_distanciation
    for i in range(0,move_before_start):
        population += np.random.uniform(-init_delta_xy,init_delta_xy,population.shape)
    populations_yx.append(population)

#contaminate p_contaminated_per_population
contaminations = np.random.uniform(0,1,(n_population, n_persons_per_population)) < p_contaminated_per_population 
contaminations = list(np.array(contaminations, dtype=float))

#put random numbers on days since contamination 
for c in contaminations:
    contaminated = c!=0
    c[contaminated] = np.random.normal(mean_hours_since_contamination, std_hours_since_contamination, c[contaminated].shape)
    c[contaminated] = np.clip(c[contaminated], 0, np.inf)
        


# =============================================================================
# Main Loop - Move populations and contaminate
# =============================================================================
#init
hour = 0
epsilon = 0.01
stats = {}
stats_per_pop = {}
stats_per_day_per_pop = {}
confined = {}
#f = figure(num=None, figsize=(6, 6), dpi=80, facecolor='w', edgecolor='k')
#for each hour
for hour in range(0, n_hours):
    
    #process infections
    for c in contaminations:
        #forward contaminations by 1 hour
        c[c>VULNERABLE] += 1
        #people above 14 days are protected (100% for now)
        c[c>contamination_duration] = IMMUNIZED
    
    
    #each 24 hours, some people stay confined under lockdown
    if hour % 24 == 0 and p_confined:
        for i_population, population in enumerate(populations_yx):
            confined[i_population] = np.array(np.random.binomial(1, p_confined, population.shape[1]), dtype=bool)
        
    #for each population
    i_population = 0
    for i_population, population in enumerate(populations_yx):
        
        #create ingame and alive filters
        population = populations_yx[i_population]
        contamination = contaminations[i_population]
        ingame_mask = contaminations[i_population]>=VULNERABLE
        alive_mask = contaminations[i_population]>=IMMUNIZED
            
        #each person move default_movement on average
        if hour % 24 in moving_time:
            delta_xy = np.random.uniform(0,default_movement*2,population.shape)
            delta_xy *= np.random.choice([-1,1],size=delta_xy.shape)
            if p_confined:
                population[:,(~confined[i_population])&alive_mask] += delta_xy[:,(~confined[i_population])&alive_mask]
            else:
                population[:,alive_mask] += delta_xy[:,alive_mask]
        
        #if two persons are close enough, apply contamination probability if not protected
        # - get positions of infected people
        infectious_mask = contamination>delay_before_infectious
        infectious = infectious_mask.nonzero()[0]
        
        #if there are still vulnerable people, play
        if sum(ingame_mask) > 0:
            
            # - create a "grid" for finding infectious people
            resolution = int(max(distance_to_pcontamination.keys()) / 2) + 1
            x1y1x2y2grid = (population[1,ingame_mask].min(), population[0,ingame_mask].min(), 
                            population[1,ingame_mask].max(), population[0,ingame_mask].max())
            x1, y1, x2, y2 = [int(a) for a in x1y1x2y2grid]
            w = x2-x1+2
            h = y2-y1+2
            w //= resolution
            h //= resolution
            
            population_t = population.transpose()
            contaminated_yx = population_t[infectious, :]
            yx_to_ids = defaultdict(list)
            for i, indiv in enumerate(contaminated_yx):
                indiv = indiv-(x1y1x2y2grid[1],x1y1x2y2grid[0])
                indiv = (indiv // resolution).astype(int)
                yx_to_ids[(indiv[0], indiv[1])] += [infectious[i]]
            
            # - try to contaminate all unprotected uninfected people 
            contamiable_mask = contamination==VULNERABLE
            contaminable_nz = contamiable_mask.nonzero()[0]
            contaminable_yx = population_t[contamiable_mask,:]
            # if there are contaminated people:
            if len(contaminated_yx) > 0:
                # for each contaminable individual
                for i_indiv, indiv in enumerate(contaminable_yx):
                    assert contamination[contamiable_mask][i_indiv] == 0
                    indiv = indiv-(x1y1x2y2grid[1],x1y1x2y2grid[0])
                    indiv = (indiv // resolution).astype(int)
                    #look around him
                    for dy in range(-1,2):
                        for dx in range(-1,2):
                            #if a contaminated person is  here
                            yx = (indiv[0]+dy, indiv[1]+dx)
                            if yx in yx_to_ids:
                                #compute distances (not very efficient)
                                ids = yx_to_ids[yx]
                                contaminated_yx_local = population_t[ids,:]
                                distances = spatial.distance.cdist(contaminated_yx_local, np.expand_dims(contaminable_yx[i_indiv],0))
                                #for each distance with a contaminated person 
                                for d in distances:
                                    #for each distance that could contaminate
                                    for d_to_cont in distance_to_pcontamination.keys():
                                        #if we could be contaminated based on these distances, try to contaminate
                                        if d <= d_to_cont:
                                            p = distance_to_pcontamination[d_to_cont]
                                            contamination[contaminable_nz[i_indiv]] = np.random.binomial(1, p) * epsilon
                                            break #only try the closest distance then skip to next
                                    #stop if already contaminated
                                    if contamination[i_indiv]:
                                        break
                            #stop if already contaminated
                            if contamination[i_indiv]:
                                break
                        #stop if already contaminated
                        if contamination[i_indiv]:
                            break
            
            #kill people
            killable_people = (contamination>death_possibility_delay).nonzero()[0]
            kill = (np.random.binomial(1, p_lethality_per_hour, len(killable_people))).astype(bool)
            contamination[killable_people[kill]] = DEAD
            contaminations[i_population] = contamination
        
        #plot population
#        if hour % 1 == 0:
#            mask = np.zeros(contamination.shape, dtype=int)
#            mask += contamination > DEAD
#            mask += contamination > IMMUNIZED
#            mask += contamination > VULNERABLE
#            for i in range(0,4):
#                plt.scatter(population[0,mask==i], population[1,mask==i], c=colors[i], s=25)
##            plt.title(f'Jour {hour//24} - Heure {hour%24}')
##            plt.savefig(f'{simulation_dir}/{i_population}_{hour}_{hour//24}.png')
#            plt.show()
##            plt.clf()
        
        
        #log each day for that population
        if hour % 24 == 0:
            n_kill = (contaminations[i_population] == DEAD).sum()
            n_contaminated = (contaminations[i_population] > 0).sum()
            n_recovered = (contaminations[i_population] == IMMUNIZED).sum()
            n_population = population.shape[1]
            stats['day'] = hour//24
            stats['hour'] = hour
            stats['kill'] = n_kill
            stats['contaminated'] = n_contaminated
            stats['recovered'] = n_recovered
            stats['populations_length'] = population.shape[1]
            stats_per_pop[i_population] = stats.copy()
            print("day:", hour//24, "- pop:", i_population, ' - total:',n_population ,' - kill:', n_kill, ' - contaminated:', n_contaminated, ' - recovered:', n_recovered)
       
    #log each day for all populations
    if hour % 24 == 0: 
        stats_per_day_per_pop[hour//24] = stats_per_pop.copy()
    
    
    
    #movement between populations
    if len(populations_yx) > 1 and probability_population_transfer and hour % 24 in moving_time:
        for i in range(len(populations_yx)):
            if np.random.binomial(1, probability_population_transfer):
                #choose a start and end population
                weights = [p.shape[1] for p in populations_yx]
                population_start = np.random.choice(list(range(0, len(populations_yx))), p=[w/sum(weights) for w in weights])
                weights[population_start] = 0
                population_end = np.random.choice(list(range(0, len(populations_yx))), p=[w/sum(weights) for w in weights])
                
                #remove the persons from the 1st population
#                selected_person = np.random.randint(0, populations_yx[population_start].shape[1])
                len_pop = populations_yx[population_start].shape[1]
                selected_persons = np.random.choice(len_pop, int(proportion_population_transfer*len_pop), replace=False)
                selected_persons_status = contaminations[population_start][selected_persons]
                populations_yx[population_start] = np.delete(populations_yx[population_start], selected_persons, 1)
                contaminations[population_start] = np.delete(contaminations[population_start], selected_persons, 0)
                
                #make these persons join other persons in the second population
                len_pop = populations_yx[population_end].shape[1]
                selected_persons = np.random.choice(len_pop, len(selected_persons), replace=False)
                selected_persons = populations_yx[population_end][:,selected_persons] #np.expand_dims(populations_yx[population_end][:,selected_persons], 1) 
                populations_yx[population_end] = np.hstack((populations_yx[population_end], selected_persons ) )
                contaminations[population_end] = np.hstack((contaminations[population_end], selected_persons_status ) )
                


##log logs
print('logging...')
log_file = f'{simulation_dir}/logs.pydict'
open(log_file, 'w+').write(str(stats_per_day_per_pop))
d = eval(open(log_file).read())


