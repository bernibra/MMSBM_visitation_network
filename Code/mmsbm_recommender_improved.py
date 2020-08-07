#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Mixed membership stochastic block model (Godoy-Lorite et al. 2016)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#----------------------------------------------------------------------
# PURPOSE:
# This method attempt to predict missing links within
# a network based on observed links.
#---------------------------------------------------------------------
# In our case we attempt to predict the missing link between a place
# and a visitor based on the choices users made without categorising
# them in a defined group
#-----------------------------------------------------------------------
# NOTE: [Antonia's code]
# 1) Script modified to speed up code on python
# 2) Order of vectors not the same as in Antonia's code
# 3) Some modifications are still required to run the prediction.dat bit
#-----------------------------------------------------------------------

#importing the different modules
import sys
# import numpy as np (short form)
#import _numpypy as np
import numpy as np
from math import *
import copy
import random
import csv

def read_files(training, zeros_as_null):

    user_dict = {}
    place_dict = {}
    visits = {}
    linksr = []

    file = open(training,"r")
    for line in file:
        about = line.strip().split("\t")
        if int(about[2])!=0 or not zeros_as_null:
            try:
                x = user_dict[about[0]][0]
                user_dict[about[0]][1] += 1
            except KeyError:
                #x = len(user_dict)
                x = int(about[0])-1
                user_dict[about[0]] = [x, 1]

            try:
                y = place_dict[about[1]][0]
                place_dict[about[1]][1] += 1
            except KeyError:
                #y = len(place_dict)
                y = int(about[1])-1
                place_dict[about[1]] = [y, 1]

            try:
                v = visits[about[2]]
            except KeyError:
                v = len(visits)
                visits[about[2]] = v

            linksr.append([x, y, v])
    file.close()

    return user_dict, place_dict, linksr, visits

def calc_likelihood(linksr, theta, eta, L, K, pr):
    Like = 0.
    for n, m, ra in linksr:
        D = 0.
        for l in range(L):
            for k in range(K):
                D = D+theta[n][k]*eta[m][l]*pr[ra][k][l]
        for l in range(L):
            for k in range(K):
                Like = Like+(theta[n][k]*eta[m][l]*pr[ra][k][l])*log(D)/D
    return Like

def sampling(c, ofolder, linksr, nsampling, iterations, K, L, R,
             user_dict, n_users, users_denom, place_dict, n_places, places_denom,
             verbose, study_likelyhood, alloutput):

    if verbose:
        sys.stderr.write(" ".join(['sampling',str(c),'\n']))

    #theta = vector containing the different groups to which each user belongs to
    theta = np.random.rand(n_users,K) / users_denom[:,np.newaxis]

    #eta = vector containing the different groups to which each place belongs to
    eta = np.random.rand(n_places,L) / places_denom[:,np.newaxis]

    # 3d matrix containing random probabilities of ratings across user-group and place-group combos
    # NOTE: I have changed the structure of this and related variables!!!
    pr = np.random.rand(R, K, L)

    # normalize the probabilities across ratings
    # should divide by: sum of all ratings corresponding to a group-group combo
    pr = pr / pr.sum(axis=0)

    # create empty containers for the calculations that are made during each iteration
    ntheta = np.zeros((n_users, K))
    neta = np.zeros((n_places, L))
    npr = np.zeros((R, K, L))
    Like=[]

    ################################################################################

    for g in range(iterations):
        if verbose:
            sys.stderr.write(" ".join(['iteration',str(g),'\n']))
        
        if study_likelyhood:
            Like+=[calc_likelihood(linksr, theta, eta, L, K, pr)]

        # update the parameters using each observed 'rating'
        for n, m, ra in linksr:

            # calculate the sum of all mixtures for rating ra by
            # multiplying rating probabilities rowwise by the user group
            # membership probabilities and columnwise by the place group
            # membership probabilities

            D = (pr[ra].T * theta[n]).T * eta[m]

            # normalize these values
            a = D / D.sum()

            # update the new (n) parameter estimates
            npr[ra] = npr[ra] + a
            ntheta[n] = ntheta[n] + a.sum(axis=1)
            neta[m] = neta[m] + a.sum(axis=0)

        # normalize the users' membership probabilities across groups
        ntheta = ntheta / users_denom[:,np.newaxis]

        # normalize the places' membership probabilities across groups
        neta = neta / places_denom[:,np.newaxis]

        # normalize the probabilities across ratings
        npr = npr / npr.sum(axis=0)

        # create copies of previous values and zero'd estimates as placeholders
        theta = copy.deepcopy(ntheta)
        eta = copy.deepcopy(neta)
        pr = copy.deepcopy(npr)

        # restart arrays
        ntheta = ntheta*0
        neta = neta*0
        npr = npr*0

    # calculate the likelihood given the probabilities
    if study_likelyhood:
        Like += [calc_likelihood(linksr, theta, eta, L, K, pr)]
    else:
        Like = calc_likelihood(linksr, theta, eta, L, K, pr)
    
    if verbose:
        print Like[-1]
        
    
    #inv_user = {v[0]: k for k, v in user_dict.iteritems()}
    #id_user = np.asarray([int(inv_user[x]) for x in np.sort(np.asarray(inv_user.keys()))])
    #inv_place = {v[0]: k for k, v in place_dict.iteritems()}
    #id_place = np.asarray([int(inv_place[x]) for x in np.sort(np.asarray(inv_place.keys()))])
    #theta_=theta[id_user,:]
    #eta_=eta[id_place,:]
    
    if alloutput:
        np.save(ofolder + "_".join([str(c), str(K), str(L)]) + "_theta", theta)
        np.save(ofolder + "_".join([str(c), str(K), str(L)]) + "_eta", eta)
        np.save(ofolder + "_".join([str(c), str(K), str(L)]) + "_pr", pr)
        np.save(ofolder + "_".join([str(c), str(K), str(L)]) + "_like", np.asarray(Like))
        
    return Like

def run_sampling(training,
             ofolder="../../results/test/",
             K=10,
             L=10,
             nsampling=0,
             iterations=200,
             zeros_as_null=False,
             verbose=False,
             study_likelyhood=True,
             alloutput=True):

    if not zeros_as_null:
        sys.stderr.write("\nCareful! Zeros in your data will represent a type of interaction and will be used in the sampling.\n\n")

    user_dict, place_dict, linksr, visits = read_files(training, zeros_as_null)

    n_users = len(user_dict)
    n_places = len(place_dict)
    R = len(visits)

    users_denom = np.asarray(user_dict.values())
    users_denom = users_denom[users_denom[:,0].argsort(),1]
    places_denom = np.asarray(place_dict.values())
    places_denom = places_denom[places_denom[:,0].argsort(),1]
    
    lkl = sampling(nsampling, ofolder, linksr, nsampling, iterations, K, L, R, user_dict, n_users, users_denom, place_dict, n_places, places_denom, verbose, study_likelyhood, alloutput)
    
    return lkl



