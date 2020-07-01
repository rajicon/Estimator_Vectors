#!/usr/bin/env cython
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# coding: utf-8
#
# By Raj Patel, but heavily based on (built out of) Gensim's word2vec_inner.pyx
# Based on code by Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import cython
import numpy as np
cimport numpy as np

from libc.math cimport exp
from libc.math cimport log
from libc.string cimport memset

# scipy <= 0.15
try:
    from scipy.linalg.blas import fblas
except ImportError:
    # in scipy > 0.15, fblas function has been removed
    import scipy.linalg.blas as fblas

REAL = np.float32

DEF MAX_SENTENCE_LEN = 10000
DEF MAX_CONTEXT_SIZE = 30  #Raj:  We need a max because size decided at compile time, methods will control this

cdef scopy_ptr scopy=<scopy_ptr>PyCObject_AsVoidPtr(fblas.scopy._cpointer)  # y = x
cdef saxpy_ptr saxpy=<saxpy_ptr>PyCObject_AsVoidPtr(fblas.saxpy._cpointer)  # y += alpha * x
cdef sdot_ptr sdot=<sdot_ptr>PyCObject_AsVoidPtr(fblas.sdot._cpointer)  # float = dot(x, y)
cdef dsdot_ptr dsdot=<dsdot_ptr>PyCObject_AsVoidPtr(fblas.sdot._cpointer)  # double = dot(x, y)
cdef snrm2_ptr snrm2=<snrm2_ptr>PyCObject_AsVoidPtr(fblas.snrm2._cpointer)  # sqrt(x^2)
cdef sscal_ptr sscal=<sscal_ptr>PyCObject_AsVoidPtr(fblas.sscal._cpointer) # x = alpha * x

DEF EXP_TABLE_SIZE = 1000
DEF MAX_EXP = 6



cdef REAL_t[EXP_TABLE_SIZE] EXP_TABLE
cdef REAL_t[EXP_TABLE_SIZE] LOG_TABLE

cdef int ONE = 1
cdef REAL_t ONEF = <REAL_t>1.0

# for when fblas.sdot returns a double
cdef REAL_t our_dot_double(const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil:
    return <REAL_t>dsdot(N, X, incX, Y, incY)

# for when fblas.sdot returns a float
cdef REAL_t our_dot_float(const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil:
    return <REAL_t>sdot(N, X, incX, Y, incY)

# for when no blas available
cdef REAL_t our_dot_noblas(const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil:
    # not a true full dot()-implementation: just enough for our cases
    cdef int i
    cdef REAL_t a
    a = <REAL_t>0.0
    for i from 0 <= i < N[0] by 1:
        a += X[i] * Y[i]
    return a

# for when no blas available
cdef void our_saxpy_noblas(const int *N, const float *alpha, const float *X, const int *incX, float *Y, const int *incY) nogil:
    cdef int i
    for i from 0 <= i < N[0] by 1:
        Y[i * (incY[0])] = (alpha[0]) * X[i * (incX[0])] + Y[i * (incY[0])] #Raj, this is the gradient update

#Raj: I think I can eventually remove this
cdef void fast_sentence_sg_hs(
    const np.uint32_t *word_point, const np.uint8_t *word_code, const int codelen,
    REAL_t *syn0, REAL_t *syn1, const int size,
    const np.uint32_t word2_index, const REAL_t alpha, REAL_t *work, REAL_t *word_locks,
    const int _compute_loss, REAL_t *_running_training_loss_param) nogil:

    cdef long long a, b
    cdef long long row1 = word2_index * size, row2, sgn
    cdef REAL_t f, g, f_dot, lprob

    memset(work, 0, size * cython.sizeof(REAL_t))
    for b in range(codelen):
        row2 = word_point[b] * size
        f_dot = our_dot(&size, &syn0[row1], &ONE, &syn1[row2], &ONE)
        if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (1 - word_code[b] - f) * alpha

        if _compute_loss == 1:
            sgn = (-1)**word_code[b]  # ch function: 0-> 1, 1 -> -1
            lprob = -1*sgn*f_dot
            if lprob <= -MAX_EXP or lprob >= MAX_EXP:
                continue
            lprob = LOG_TABLE[<int>((lprob + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
            _running_training_loss_param[0] = _running_training_loss_param[0] - lprob

        our_saxpy(&size, &g, &syn1[row2], &ONE, work, &ONE)
        our_saxpy(&size, &g, &syn0[row1], &ONE, &syn1[row2], &ONE)

    our_saxpy(&size, &word_locks[word2_index], work, &ONE, &syn0[row1], &ONE)


# to support random draws from negative-sampling cum_table
cdef inline unsigned long long bisect_left(np.uint32_t *a, unsigned long long x, unsigned long long lo, unsigned long long hi) nogil:
    cdef unsigned long long mid
    while hi > lo:
        mid = (lo + hi) >> 1
        if a[mid] >= x:
            hi = mid
        else:
            lo = mid + 1
    return lo

# this quick & dirty RNG apparently matches Java's (non-Secure)Random
# note this function side-effects next_random to set up the next number
cdef inline unsigned long long random_int32(unsigned long long *next_random) nogil:
    cdef unsigned long long this_random = next_random[0] >> 16
    next_random[0] = (next_random[0] * <unsigned long long>25214903917ULL + 11) & 281474976710655ULL
    return this_random

# (cc word) (word sub)
cdef unsigned long long fast_sentence_sg_neg(
    const int negative, np.uint32_t *cum_table, unsigned long long cum_table_len,
    REAL_t *syn3cc, REAL_t *syn1neg, REAL_t *syn2subword, const int size, const np.uint32_t word_index,
    const np.uint32_t word2_index, const REAL_t alpha, REAL_t *work, REAL_t *word_work, REAL_t *neu1, REAL_t *sub_neu,
    unsigned long long next_random, REAL_t *word_locks,
    const int _compute_loss, REAL_t *_running_training_loss_param, np.uint32_t* cc_set, int* index_subword_breakdown, int subword_list_len, int sub_padding_num, int *skipped_pairs) nogil:


    #syn3cc: context clue   #neu1 holds sum
    #syn1neg: word
    #syn2subword:  subword  #sub_neu holds sum

    cdef long long a
    cdef long long row1 = word2_index * size, row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, label, f_dot, log_e_f_dot, cc_s_dot, f_cc_s, g_cc_s, w_s_dot, f_w_s, g_w_s
    cdef np.uint32_t target_index
    cdef int d, m, qqqq, q, yyyy, sub_ind

    memset(work, 0, size * cython.sizeof(REAL_t))  #I think work is for storing gradient updates for syn3cc (with all negs/target word)
                                                   #in our case, work will store all gradients for each context clue I guess?
                                                   #I think memset is analoguous to np.zeros, although I'm not sure

    memset(word_work, 0, size * cython.sizeof(REAL_t)) #for LHS word vector

    #Raj: it would appear that row 1 is matched with word2 and row 2 is matched with word1/neg samples for some reason
    #Raj:  EXP_TABLE is precomputed sigmoid, LOG_TABLE is log sigmoid


    #Raj calculate sum of context clues instead of syn3cc[row1], use neu1
    #based on how cbow does it
    memset(neu1, 0, size * cython.sizeof(REAL_t))
    #count = <REAL_t>0.0
    for m in range(0, MAX_CONTEXT_SIZE):
        #count += ONEF
        if cc_set[m] != -1:
            our_saxpy(&size, &ONEF, &syn3cc[cc_set[m] * size], &ONE, neu1, &ONE)  #our alpha here is 1, so just adds each context clue
    
    for d in range(negative+1):
        if d == 0:
            target_index = word_index
            label = ONEF
        else:
            target_index = bisect_left(cum_table, (next_random >> 16) % cum_table[cum_table_len-1], 0, cum_table_len)
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            #if target_index == word_index:
            #    continue
                
            #if target_index == word2_index: #Raj: added by me, to prevent neg sample being picked that matches word context clues are trying to estimate.
            #    continue
            
            '''
            while target_index == word_index or target_index == word2_index:
                target_index = bisect_left(cum_table, (next_random >> 16) % cum_table[cum_table_len-1], 0, cum_table_len)
                next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            '''
            
            label = <REAL_t>0.0


        #calculate subword of row2/neg sample
        memset(sub_neu, 0, size * cython.sizeof(REAL_t))  #reset subword sum
        for q in range(0, subword_list_len):
            sub_ind = index_subword_breakdown[(target_index * subword_list_len)+q]
            if sub_ind != sub_padding_num:
                our_saxpy(&size, &ONEF, &syn2subword[sub_ind * size], &ONE, sub_neu, &ONE)  #our alpha here is 1, so just adds each context clue
            else:
                break

        row2 = target_index * size
        
        
        #cc and word -------------------------------------------------------------
        
        f_dot = our_dot(&size, neu1, &ONE, &syn1neg[row2], &ONE)   
        if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
        
            skipped_pairs[0] = skipped_pairs[0] + 1
            continue
            
        f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]  #this is basically applying sigmoid function but using EXP_TABLE for fast calculations
        g = (label - f) * alpha   #so this is label - sigmoid(u*v) * lr
        #------------------------------------------------------------------------
        
        
        ''' #remove
        #cc and sub -------------------------------------------------------------
        cc_s_dot = our_dot(&size, neu1, &ONE, sub_neu, &ONE)   
        if cc_s_dot <= -MAX_EXP or cc_s_dot >= MAX_EXP:
            continue
        f_cc_s = EXP_TABLE[<int>((cc_s_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]  #this is basically applying sigmoid function but using EXP_TABLE for fast calculations
        g_cc_s = (label - f_cc_s) * alpha   #so this is label - sigmoid(u*v) * lr
        #------------------------------------------------------------------------
        ''' # /remove

        #word and sub -------------------------------------------------------------
        
        w_s_dot = our_dot(&size, &syn1neg[row1], &ONE, sub_neu, &ONE)   
        if w_s_dot <= -MAX_EXP or w_s_dot >= MAX_EXP:
            skipped_pairs[0] = skipped_pairs[0] + 1
            continue
        f_w_s = EXP_TABLE[<int>((w_s_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]  #this is basically applying sigmoid function but using EXP_TABLE for fast calculations
        g_w_s = (label - f_w_s) * alpha   #so this is label - sigmoid(u*v) * lr
        #------------------------------------------------------------------------
        

        if _compute_loss == 1:
            f_dot = (f_dot if d == 0  else -f_dot)
            if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                continue
            log_e_f_dot = LOG_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
            _running_training_loss_param[0] = _running_training_loss_param[0] - log_e_f_dot


        
        our_saxpy(&size, &g, &syn1neg[row2], &ONE, work, &ONE)  #LHS cc word
        
        #our_saxpy(&size, &g_cc_s, sub_neu, &ONE, work, &ONE)  #LHS cc sub
        our_saxpy(&size, &g_w_s, sub_neu, &ONE, word_work, &ONE)  #LHS word sub 
        

        #RHS do cc word LAST so we aren't changing syn1neg first 

        #RHS cc sub  &&    word sub   --------------------------------------------
        for yyyy in range(0, subword_list_len):
            sub_ind = index_subword_breakdown[(target_index * subword_list_len)+yyyy]
            if sub_ind != sub_padding_num:
                #our_saxpy(&size, &g_cc_s, neu1, &ONE, &syn2subword[sub_ind * size], &ONE)  #RHS cc sub
                our_saxpy(&size, &g_w_s, &syn1neg[row1], &ONE, &syn2subword[sub_ind * size], &ONE)  #RHS word sub
                
            else:
                break
        #----------------------------------------------------------------------------
        

        #RHS cc word ------------------------------------------
        our_saxpy(&size, &g, neu1, &ONE, &syn1neg[row2], &ONE)
        #------------------------------------------------------
        

    
    for qqqq in range(0, MAX_CONTEXT_SIZE):
        if cc_set[qqqq] != -1:
            our_saxpy(&size, &word_locks[word2_index], work, &ONE, &syn3cc[cc_set[qqqq] * size], &ONE)  #updates syn3cc(context clues?)  #LHS cc word and cc sub
    
    
    our_saxpy(&size, &word_locks[word2_index], word_work, &ONE, &syn1neg[row1], &ONE) #LHS word sub
    
    #with gil:
    #    print('got here')
    return next_random
    
    
#2 pair, average instead of sum----------------------------------------------------------------------------------------------------------------------------------
cdef unsigned long long fast_sentence_sg_neg_ALL_2_PAIRS_MEAN(
    const int negative, np.uint32_t *cum_table, unsigned long long cum_table_len,
    REAL_t *syn3cc, REAL_t *syn1neg, REAL_t *syn2subword, const int size, const np.uint32_t word_index,
    const np.uint32_t word2_index, const REAL_t alpha, REAL_t *work, REAL_t *word_work, REAL_t *neu1, REAL_t *sub_neu,
    unsigned long long next_random, REAL_t *word_locks,
    const int _compute_loss, REAL_t *_running_training_loss_param, np.uint32_t* cc_set, int* index_subword_breakdown, int subword_list_len, int sub_padding_num, int *skipped_pairs) nogil:


    #syn3cc: context clue   #neu1 holds sum
    #syn1neg: word
    #syn2subword:  subword  #sub_neu holds sum

    cdef long long a
    cdef long long row1 = word2_index * size, row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, label, f_dot, log_e_f_dot, cc_s_dot, f_cc_s, g_cc_s, w_s_dot, f_w_s, g_w_s, cc_count, sub_count, cc_inv_count = 1.0, sub_inv_count = 1.0
    cdef np.uint32_t target_index
    cdef int d, m, qqqq, q, yyyy, sub_ind

    memset(work, 0, size * cython.sizeof(REAL_t))  #I think work is for storing gradient updates for syn3cc(with all negs/target word)
                                                   #in our case, work will store all gradients for each context clue I guess?
                                                   #I think memset is analoguous to np.zeros, although I'm not sure

    memset(word_work, 0, size * cython.sizeof(REAL_t)) #for LHS word vector

    #Raj: it would appear that row 1 is matched with word2 and row 2 is matched with word1/neg samples for some reason
    #Raj:  EXP_TABLE is precomputed sigmoid, LOG_TABLE is log sigmoid


    #Raj calculate sum of context clues instead of syn3cc[row1], use neu1
    #based on how cbow does it
    memset(neu1, 0, size * cython.sizeof(REAL_t))
    cc_count = <REAL_t>0.0
    for m in range(0, MAX_CONTEXT_SIZE):
        
        if cc_set[m] != -1:
            cc_count += ONEF
            our_saxpy(&size, &ONEF, &syn3cc[cc_set[m] * size], &ONE, neu1, &ONE)  #our alpha here is 1, so just adds each context clue
    
    #take avg instead of sum    
    if cc_count > (<REAL_t>0.5):
        cc_inv_count = ONEF/cc_count
 
    sscal(&size, &cc_inv_count, neu1, &ONE)
    
    for d in range(negative+1):
        if d == 0:
            target_index = word_index
            label = ONEF
        else:
            target_index = bisect_left(cum_table, (next_random >> 16) % cum_table[cum_table_len-1], 0, cum_table_len)
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            #if target_index == word_index:
            #    continue
                
            #if target_index == word2_index: #Raj: added by me, to prevent neg sample being picked that matches word context clues are trying to estimate.
            #    continue
            
            '''
            while target_index == word_index or target_index == word2_index:
                target_index = bisect_left(cum_table, (next_random >> 16) % cum_table[cum_table_len-1], 0, cum_table_len)
                next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            '''
            
            label = <REAL_t>0.0


        #calculate subword of row2/neg sample
        memset(sub_neu, 0, size * cython.sizeof(REAL_t))  #reset subword sum
        sub_count = <REAL_t>0.0
        sub_inv_count = 1.0
        for q in range(0, subword_list_len):
            sub_ind = index_subword_breakdown[(target_index * subword_list_len)+q]
            if sub_ind != sub_padding_num:
                our_saxpy(&size, &ONEF, &syn2subword[sub_ind * size], &ONE, sub_neu, &ONE)  #our alpha here is 1, so just adds each context clue
                sub_count += ONEF
            else:
                break
        
        #take avg instead of sum        
        if sub_count > (<REAL_t>0.5):
            sub_inv_count = ONEF/sub_count
 
        sscal(&size, &sub_inv_count, sub_neu, &ONE)
                
                

        row2 = target_index * size
        
        
        #cc and word -------------------------------------------------------------
        
        f_dot = our_dot(&size, neu1, &ONE, &syn1neg[row2], &ONE)   
        if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:

            skipped_pairs[0] = skipped_pairs[0] + 1

            continue
            
        f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]  #this is basically applying sigmoid function but using EXP_TABLE for fast calculations
        g = (label - f) * alpha   #so this is label - sigmoid(u*v) * lr
        #------------------------------------------------------------------------
        
        
        '''
        #cc and sub -------------------------------------------------------------
        cc_s_dot = our_dot(&size, neu1, &ONE, sub_neu, &ONE)   
        if cc_s_dot <= -MAX_EXP or cc_s_dot >= MAX_EXP:

            skipped_pairs[0] = skipped_pairs[0] + 1

           
            continue

        f_cc_s = EXP_TABLE[<int>((cc_s_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]  #this is basically applying sigmoid function but using EXP_TABLE for fast calculations
        g_cc_s = (label - f_cc_s) * alpha   #so this is label - sigmoid(u*v) * lr
        #------------------------------------------------------------------------
        '''

        #word and sub -------------------------------------------------------------
        
        w_s_dot = our_dot(&size, &syn1neg[row1], &ONE, sub_neu, &ONE)   
        if w_s_dot <= -MAX_EXP or w_s_dot >= MAX_EXP:
            skipped_pairs[0] = skipped_pairs[0] + 1

            continue
        f_w_s = EXP_TABLE[<int>((w_s_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]  #this is basically applying sigmoid function but using EXP_TABLE for fast calculations
        g_w_s = (label - f_w_s) * alpha   #so this is label - sigmoid(u*v) * lr
        #------------------------------------------------------------------------
        

        if _compute_loss == 1:
            f_dot = (f_dot if d == 0  else -f_dot)
            if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                continue
            log_e_f_dot = LOG_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
            _running_training_loss_param[0] = _running_training_loss_param[0] - log_e_f_dot


        
        our_saxpy(&size, &g, &syn1neg[row2], &ONE, work, &ONE)  #LHS cc word
        
        #our_saxpy(&size, &g_cc_s, sub_neu, &ONE, work, &ONE)  #LHS cc sub
        our_saxpy(&size, &g_w_s, sub_neu, &ONE, word_work, &ONE)  #LHS word sub 
        

        #RHS do cc word LAST so we aren't changing syn1neg first 

        #NEED TO SCALE SUBWORDS BECAUSE OF AVERAGE! However, we don't want to change neu1 or syn1neg, so instead we scale g_cc_s and g_w_s? ***
        #g_cc_s = g_cc_s * sub_inv_count
        g_w_s = g_w_s * sub_inv_count
        
        #RHS cc sub  &&    word sub   --------------------------------------------
        for yyyy in range(0, subword_list_len):
            sub_ind = index_subword_breakdown[(target_index * subword_list_len)+yyyy]
            if sub_ind != sub_padding_num:
                
                
                #our_saxpy(&size, &g_cc_s, neu1, &ONE, &syn2subword[sub_ind * size], &ONE)  #RHS cc sub
                our_saxpy(&size, &g_w_s, &syn1neg[row1], &ONE, &syn2subword[sub_ind * size], &ONE)  #RHS word sub
                
            else:
                break
        #----------------------------------------------------------------------------
        

        #RHS cc word ------------------------------------------
        our_saxpy(&size, &g, neu1, &ONE, &syn1neg[row2], &ONE)
        #------------------------------------------------------
        

    
    sscal(&size, &cc_inv_count, work, &ONE)  #need to scale derivative by average I think?  ***
    
    for qqqq in range(0, MAX_CONTEXT_SIZE):
        if cc_set[qqqq] != -1:
            our_saxpy(&size, &word_locks[word2_index], work, &ONE, &syn3cc[cc_set[qqqq] * size], &ONE)  #updates syn3cc(context clues?)  #LHS cc word and cc sub
    
    
    our_saxpy(&size, &word_locks[word2_index], word_work, &ONE, &syn1neg[row1], &ONE) #LHS word sub
    
    #with gil:
    #    print('got here')
    return next_random


#--------------------------------------------------------------------------------------------------------------------------------------------------------------



#2 pair but faster by subwords on left side
# (cc word) (sub word)
#----------------------------------------------------------------------------------------------------------------------------------
cdef unsigned long long FASTER_2_PAIRS_MEAN(
    const int negative, np.uint32_t *cum_table, unsigned long long cum_table_len,
    REAL_t *syn3cc, REAL_t *syn1neg, REAL_t *syn2subword, const int size, const np.uint32_t word_index,
    const np.uint32_t word2_index, const REAL_t alpha, REAL_t *work, REAL_t *sub_work, REAL_t *neu1, REAL_t *sub_neu,
    unsigned long long next_random, REAL_t *word_locks,
    const int _compute_loss, REAL_t *_running_training_loss_param, np.uint32_t* cc_set, int* index_subword_breakdown, int subword_list_len, int sub_padding_num, int *skipped_pairs) nogil:


    #syn3cc: context clue   #neu1 holds sum
    #syn1neg: word
    #syn2subword:  subword  #sub_neu holds sum

    cdef long long a
    cdef long long row1 = word2_index * size, row2, row3, row4
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, label, f_dot, log_e_f_dot, cc_s_dot, f_cc_s, g_cc_s, w_s_dot, f_w_s, g_w_s, cc_count, sub_count1, cc_inv_count = 1.0, sub_inv_count1 = 1.0
    cdef np.uint32_t target_index
    cdef int d, m, qqqq, q, yyyy, sub_ind, e, e2
    cdef int temp_int_cc_set_m = 0

    memset(work, 0, size * cython.sizeof(REAL_t))  #I think work is for storing gradient updates for syn3cc (with all negs/target word)
                                                   #in our case, work will store all gradients for each context clue I guess?
                                                   #I think memset is analoguous to np.zeros, although I'm not sure

    memset(sub_work, 0, size * cython.sizeof(REAL_t)) #for LHS sub vector

    #Raj: it would appear that row 1 is matched with word2 and row 2 is matched with word1/neg samples for some reason
    #Raj:  EXP_TABLE is precomputed sigmoid, LOG_TABLE is log sigmoid



    #Raj calculate sum of context clues instead of syn3cc[row1], use neu1
    #based on how cbow does it
    memset(neu1, 0, size * cython.sizeof(REAL_t))
    cc_count = <REAL_t>0.0
    for m in range(0, MAX_CONTEXT_SIZE):
        temp_int_cc_set_m = cc_set[m]
        if temp_int_cc_set_m != -1:
            cc_count += ONEF
            

            row3 = temp_int_cc_set_m * size

            our_saxpy(&size, &ONEF, &syn3cc[row3], &ONE, neu1, &ONE)  #our alpha here is 1, so just adds each context clue
            #our_saxpy(&size, &ONEF, &syn3cc[temp_int_cc_set_m * size], &ONE, neu1, &ONE)  #our alpha here is 1, so just adds each context clue
            #our_saxpy(&size, &ONEF, &syn3cc[cc_set[m] * size], &ONE, neu1, &ONE)  #our alpha here is 1, so just adds each context clue
    
    #take avg instead of sum    
    if cc_count > (<REAL_t>0.5):
        cc_inv_count = ONEF/cc_count
 
    sscal(&size, &cc_inv_count, neu1, &ONE)
    
   
    #Raj calculate subwords instead of syn3cc[row1]
    memset(sub_neu, 0, size * cython.sizeof(REAL_t))  #reset subword sum


    sub_count1 = <REAL_t>0.0
    sub_inv_count1 = 1.0
    for e in range(0, subword_list_len):

        sub_ind = index_subword_breakdown[(word2_index * subword_list_len)+e]

        if sub_ind != sub_padding_num:

            our_saxpy(&size, &ONEF, &syn2subword[sub_ind * size], &ONE, sub_neu, &ONE)  #our alpha here is 1, so just adds each context clue
            sub_count1 += ONEF
        
            
        else:
            break
            
            

   
    #take avg instead of sum        
    if sub_count1 > (<REAL_t>0.5):
        sub_inv_count1 = ONEF/sub_count1



    sscal(&size, &sub_inv_count1, sub_neu, &ONE)
   
    for d in range(negative+1):
        if d == 0:
            target_index = word_index
            label = ONEF
        else:
            target_index = bisect_left(cum_table, (next_random >> 16) % cum_table[cum_table_len-1], 0, cum_table_len)
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            #if target_index == word_index:
            #    continue
                
            #if target_index == word2_index: #Raj: added by me, to prevent neg sample being picked that matches word context clues are trying to estimate.
            #    continue
            
            
            #while target_index == word_index or target_index == word2_index:
            #    target_index = bisect_left(cum_table, (next_random >> 16) % cum_table[cum_table_len-1], 0, cum_table_len)
            #    next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            
            
            label = <REAL_t>0.0


        #calculate subword of row2/neg sample

              
                

        row2 = target_index * size
        
        
        #cc and word -------------------------------------------------------------
        
        f_dot = our_dot(&size, neu1, &ONE, &syn1neg[row2], &ONE)   
        if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
            skipped_pairs[0] = skipped_pairs[0] + 1
            continue


        f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]  #this is basically applying sigmoid function but using EXP_TABLE for fast calculations
        g = (label - f) * alpha   #so this is label - sigmoid(u*v) * lr
        #------------------------------------------------------------------------
      
        
        #sub and word -------------------------------------------------------------
        
        w_s_dot = our_dot(&size, sub_neu, &ONE, &syn1neg[row2], &ONE)   
        if w_s_dot <= -MAX_EXP or w_s_dot >= MAX_EXP:
            skipped_pairs[0] = skipped_pairs[0] + 1
            continue
        

        
        f_w_s = EXP_TABLE[<int>((w_s_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]  #this is basically applying sigmoid function but using EXP_TABLE for fast calculations
        g_w_s = (label - f_w_s) * alpha   #so this is label - sigmoid(u*v) * lr
        #------------------------------------------------------------------------
         

        if _compute_loss == 1:
            f_dot = (f_dot if d == 0  else -f_dot)
            if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                continue
            log_e_f_dot = LOG_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
            _running_training_loss_param[0] = _running_training_loss_param[0] - log_e_f_dot

   
         
        our_saxpy(&size, &g, &syn1neg[row2], &ONE, work, &ONE)  #LHS cc word
      
        our_saxpy(&size, &g_w_s, &syn1neg[row2], &ONE, sub_work, &ONE)  #LHS sub word
           

        #RHS do cc word LAST so we aren't changing syn1neg first 


           

        
        #RHS cc word and sub word ------------------------------------------
        our_saxpy(&size, &g, neu1, &ONE, &syn1neg[row2], &ONE)
        our_saxpy(&size, &g_w_s, sub_neu, &ONE, &syn1neg[row2], &ONE)
        #------------------------------------------------------
         

    
    sscal(&size, &cc_inv_count, work, &ONE)  #need to scale derivative by average I think?  ***
    sscal(&size, &sub_inv_count1, sub_work, &ONE)  #need to scale derivative by average I think?  ***

    for qqqq in range(0, MAX_CONTEXT_SIZE):
        temp_int_cc_set_m = cc_set[qqqq]
        if temp_int_cc_set_m != -1:


            row4 = temp_int_cc_set_m * size
            our_saxpy(&size, &ONEF, work, &ONE, &syn3cc[row4], &ONE)  #updates syn3cc(context clues?)  #LHS cc word and cc sub

            #our_saxpy(&size, &ONEF, work, &ONE, &syn3cc[temp_int_cc_set_m * size], &ONE)  #updates syn3cc(context clues?)  #LHS cc word and cc sub
            #our_saxpy(&size, &word_locks[word2_index], work, &ONE, &syn3cc[cc_set[qqqq] * size], &ONE)  #updates syn3cc (context clues?)  #LHS cc word and cc sub
    
              
    for e2 in range(0, subword_list_len):
        sub_ind = index_subword_breakdown[(word2_index * subword_list_len)+e2]  
        if sub_ind != sub_padding_num:
            our_saxpy(&size, &ONEF, sub_work, &ONE, &syn2subword[sub_ind * size], &ONE)  #our alpha here is 1, so just adds each context clue

        else:
            break

    return next_random

    
#(cc, word) (sub, word) (cc + sub, word)
#----------------------------------------------------------------------------------------------------------------------------------
cdef unsigned long long FASTER_3_PAIRS_MEAN(
    const int negative, np.uint32_t *cum_table, unsigned long long cum_table_len,
    REAL_t *syn3cc, REAL_t *syn1neg, REAL_t *syn2subword, const int size, const np.uint32_t word_index,
    const np.uint32_t word2_index, const REAL_t alpha, REAL_t *work, REAL_t *sub_work, REAL_t *neu1, REAL_t *sub_neu, REAL_t *cc_sub_sum1,
    unsigned long long next_random, REAL_t *word_locks,
    const int _compute_loss, REAL_t *_running_training_loss_param, np.uint32_t* cc_set, int* index_subword_breakdown, int subword_list_len, int sub_padding_num, int *skipped_pairs) nogil:

    
    #syn3cc: context clue   #neu1 holds sum
    #syn1neg: word
    #syn2subword:  subword  #sub_neu holds sum

    cdef long long a
    cdef long long row1 = word2_index * size, row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, label, f_dot, log_e_f_dot, cc_s_dot, f_cc_s, g_cc_s, w_s_dot, f_w_s, g_w_s,  cc_s_w_dot, f_cc_s_w, g_cc_s_w, g_cc_s_w_2_scaled, cc_count, sub_count1, cc_inv_count = 1.0, sub_inv_count1 = 1.0
    cdef np.uint32_t target_index
    cdef int d, m, qqqq, q, yyyy, sub_ind, e, e2

    
    cdef REAL_t  two_divisor = 1.0/2.0    
    
    memset(work, 0, size * cython.sizeof(REAL_t))  #I think work is for storing gradient updates for syn3cc(with all negs/target word)
                                                   #in our case, work will store all gradients for each context clue I guess?
                                                   #I think memset is analoguous to np.zeros, although I'm not sure

    memset(sub_work, 0, size * cython.sizeof(REAL_t)) #for LHS sub vector

    #Raj: it would appear that row 1 is matched with word2 and row 2 is matched with word1/neg samples for some reason
    #Raj:  EXP_TABLE is precomputed sigmoid, LOG_TABLE is log sigmoid


    #Raj calculate sum of context clues instead of syn3cc[row1], use neu1
    #based on how cbow does it
    memset(neu1, 0, size * cython.sizeof(REAL_t))
    cc_count = <REAL_t>0.0
    for m in range(0, MAX_CONTEXT_SIZE):
        
        if cc_set[m] != -1:
            cc_count += ONEF
            our_saxpy(&size, &ONEF, &syn3cc[cc_set[m] * size], &ONE, neu1, &ONE)  #our alpha here is 1, so just adds each context clue
    
    #take avg instead of sum    
    if cc_count > (<REAL_t>0.5):
        cc_inv_count = ONEF/cc_count
 
    sscal(&size, &cc_inv_count, neu1, &ONE)
    
    
    #Raj calculate subwords instead of syn3cc[row1]
    memset(sub_neu, 0, size * cython.sizeof(REAL_t))  #reset subword sum
    sub_count1 = <REAL_t>0.0
    sub_inv_count1 = 1.0
    for e in range(0, subword_list_len):
        sub_ind = index_subword_breakdown[(word2_index * subword_list_len)+e]
        if sub_ind != sub_padding_num:
            our_saxpy(&size, &ONEF, &syn2subword[sub_ind * size], &ONE, sub_neu, &ONE)  #our alpha here is 1, so just adds each context clue
            sub_count1 += ONEF
        else:
            break
    
    #take avg instead of sum        
    if sub_count1 > (<REAL_t>0.5):
        sub_inv_count1 = ONEF/sub_count1

    sscal(&size, &sub_inv_count1, sub_neu, &ONE)

    memset(cc_sub_sum1, 0, size * cython.sizeof(REAL_t))
    
    our_saxpy(&size, &ONEF, neu1, &ONE, cc_sub_sum1, &ONE)
    our_saxpy(&size, &ONEF, sub_neu, &ONE, cc_sub_sum1, &ONE)
    
    #divide by 2
    sscal(&size, &two_divisor, cc_sub_sum1, &ONE)    
    
    
    for d in range(negative+1):
        if d == 0:
            target_index = word_index
            label = ONEF
        else:
            target_index = bisect_left(cum_table, (next_random >> 16) % cum_table[cum_table_len-1], 0, cum_table_len)
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            #if target_index == word_index:
            #    continue
                
            #if target_index == word2_index: #Raj: added by me, to prevent neg sample being picked that matches word context clues are trying to estimate.
            #    continue
            
            '''
            while target_index == word_index or target_index == word2_index:
                target_index = bisect_left(cum_table, (next_random >> 16) % cum_table[cum_table_len-1], 0, cum_table_len)
                next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            '''
            
            label = <REAL_t>0.0


        #calculate subword of row2/neg sample

                
                

        row2 = target_index * size
        
        
        #cc and word -------------------------------------------------------------
        
        f_dot = our_dot(&size, neu1, &ONE, &syn1neg[row2], &ONE)   
        #if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
            #skipped_pairs[0] = skipped_pairs[0] + 1
            #continue
            
        if f_dot <= -MAX_EXP:
            f_dot = -MAX_EXP
        if f_dot >= MAX_EXP:
            f_dot = MAX_EXP
            
        f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]  #this is basically applying sigmoid function but using EXP_TABLE for fast calculations
        g = (label - f) * alpha   #so this is label - sigmoid(u*v) * lr
        #------------------------------------------------------------------------
        
        
        #sub and word -------------------------------------------------------------
        
        w_s_dot = our_dot(&size, sub_neu, &ONE, &syn1neg[row2], &ONE)   
        #if w_s_dot <= -MAX_EXP or w_s_dot >= MAX_EXP:
            #skipped_pairs[0] = skipped_pairs[0] + 1
            #continue
        
        if w_s_dot <= -MAX_EXP:
            w_s_dot = -MAX_EXP
        if w_s_dot >= MAX_EXP:        
            w_s_dot = MAX_EXP
            
        f_w_s = EXP_TABLE[<int>((w_s_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]  #this is basically applying sigmoid function but using EXP_TABLE for fast calculations
        g_w_s = (label - f_w_s) * alpha   #so this is label - sigmoid(u*v) * lr
        #------------------------------------------------------------------------
        
        #cc+sub and word -------------------------------------------------------------
        
        cc_s_w_dot = our_dot(&size, cc_sub_sum1, &ONE, &syn1neg[row2], &ONE)    
        #if cc_s_w_dot <= -MAX_EXP or cc_s_w_dot >= MAX_EXP:
        #    skipped_pairs[0] = skipped_pairs[0] + 1
        #    continue
        
        if cc_s_w_dot <= -MAX_EXP:
            cc_s_w_dot = -MAX_EXP        
        if cc_s_w_dot >= MAX_EXP:
            cc_s_w_dot = MAX_EXP        
        
        
        f_cc_s_w = EXP_TABLE[<int>((cc_s_w_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]  #this is basically applying sigmoid function but using EXP_TABLE for fast calculations
        g_cc_s_w = (label - f_cc_s_w) * alpha   #so this is label - sigmoid(u*v) * lr
        #------------------------------------------------------------------------
        

        if _compute_loss == 1:
            f_dot = (f_dot if d == 0  else -f_dot)
            if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                continue
            log_e_f_dot = LOG_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
            _running_training_loss_param[0] = _running_training_loss_param[0] - log_e_f_dot
        
        our_saxpy(&size, &g, &syn1neg[row2], &ONE, work, &ONE)  #LHS cc word
        
        our_saxpy(&size, &g_w_s, &syn1neg[row2], &ONE, sub_work, &ONE)  #LHS sub word
        
        #3rd pair (cc + sub, word), need to scale by 2!
        
        g_cc_s_w_2_scaled = g_cc_s_w * two_divisor        

        our_saxpy(&size, &g_cc_s_w_2_scaled, &syn1neg[row2], &ONE, work, &ONE)  #LHS cc word
        
        our_saxpy(&size, &g_cc_s_w_2_scaled, &syn1neg[row2], &ONE, sub_work, &ONE)  #LHS sub word
        

        
        #RHS cc word and sub word ------------------------------------------
        our_saxpy(&size, &g, neu1, &ONE, &syn1neg[row2], &ONE)
        our_saxpy(&size, &g_w_s, sub_neu, &ONE, &syn1neg[row2], &ONE)
        our_saxpy(&size, &g_cc_s_w, cc_sub_sum1, &ONE, &syn1neg[row2], &ONE)
        #------------------------------------------------------
        

    
    sscal(&size, &cc_inv_count, work, &ONE)  #need to scale derivative by average I think?  ***
    sscal(&size, &sub_inv_count1, sub_work, &ONE)  #need to scale derivative by average I think?  ***
    
    for qqqq in range(0, MAX_CONTEXT_SIZE):
        if cc_set[qqqq] != -1:
            our_saxpy(&size, &word_locks[word2_index], work, &ONE, &syn3cc[cc_set[qqqq] * size], &ONE)  #updates syn3cc(context clues?)  #LHS cc word and cc sub
            
    for e2 in range(0, subword_list_len):
        sub_ind = index_subword_breakdown[(word2_index * subword_list_len)+e2]  
        if sub_ind != sub_padding_num:
            our_saxpy(&size, &ONEF, sub_work, &ONE, &syn2subword[sub_ind * size], &ONE)  #our alpha here is 1, so just adds each context clue

        else:
            break
    
    return next_random


#------------------------------------



#---
#(cc, word) (sub, word) (cc + sub, word)
#----------------------------------------------------------------------------------------------------------------------------------
cdef unsigned long long FASTER_3_PAIRS_MEAN_WITH_SINGLE_PARAMETER(
    const int negative, np.uint32_t *cum_table, unsigned long long cum_table_len,
    REAL_t *syn3cc, REAL_t *syn1neg, REAL_t *syn2subword, const int size, const np.uint32_t word_index,
    const np.uint32_t word2_index, const REAL_t alpha, REAL_t *work, REAL_t *sub_work, REAL_t *neu1, REAL_t *sub_neu, REAL_t *cc_sub_sum1,
    unsigned long long next_random, REAL_t *word_locks,
    const int _compute_loss, REAL_t *_running_training_loss_param, np.uint32_t* cc_set, int* index_subword_breakdown, int subword_list_len, int sub_padding_num, int *skipped_pairs, REAL_t *single_parameter) nogil:

    
    #syn3cc: context clue   #neu1 holds sum
    #syn1neg: word
    #syn2subword:  subword  #sub_neu holds sum

    cdef long long a
    cdef long long row1 = word2_index * size, row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, label, f_dot, log_e_f_dot, cc_s_dot, f_cc_s, g_cc_s, w_s_dot, f_w_s, g_w_s,  cc_s_w_dot, f_cc_s_w, g_cc_s_w, g_cc_s_w_2_scaled_alph, g_cc_s_w_2_scaled_one_minus_alph, cc_count, sub_count1, cc_inv_count = 1.0, sub_inv_count1 = 1.0
    cdef np.uint32_t target_index
    cdef int d, m, qqqq, q, yyyy, sub_ind, e, e2

    cdef REAL_t sp_alph = <REAL_t>single_parameter[0]
    cdef REAL_t sp_one_minus_alph = <REAL_t>1.0 - sp_alph
    cdef REAL_t sp_alph_grad = 0.0
    
    memset(work, 0, size * cython.sizeof(REAL_t))  #I think work is for storing gradient updates for syn3cc(with all negs/target word)
                                                   #in our case, work will store all gradients for each context clue I guess?
                                                   #I think memset is analoguous to np.zeros, although I'm not sure

    memset(sub_work, 0, size * cython.sizeof(REAL_t)) #for LHS sub vector
    
    

    #Raj: it would appear that row 1 is matched with word2 and row 2 is matched with word1/neg samples for some reason
    #Raj:  EXP_TABLE is precomputed sigmoid, LOG_TABLE is log sigmoid


    #Raj calculate sum of context clues instead of syn3cc[row1], use neu1
    #based on how cbow does it
    memset(neu1, 0, size * cython.sizeof(REAL_t))
    cc_count = <REAL_t>0.0
    for m in range(0, MAX_CONTEXT_SIZE):
        
        if cc_set[m] != -1:
            cc_count += ONEF
            our_saxpy(&size, &ONEF, &syn3cc[cc_set[m] * size], &ONE, neu1, &ONE)  #our alpha here is 1, so just adds each context clue
    
    #take avg instead of sum    
    if cc_count > (<REAL_t>0.5):
        cc_inv_count = ONEF/cc_count
 
    sscal(&size, &cc_inv_count, neu1, &ONE)
    
    
    #Raj calculate subwords instead of syn3cc[row1]
    memset(sub_neu, 0, size * cython.sizeof(REAL_t))  #reset subword sum
    sub_count1 = <REAL_t>0.0
    sub_inv_count1 = 1.0
    for e in range(0, subword_list_len):
        sub_ind = index_subword_breakdown[(word2_index * subword_list_len)+e]
        if sub_ind != sub_padding_num:
            our_saxpy(&size, &ONEF, &syn2subword[sub_ind * size], &ONE, sub_neu, &ONE)  #our alpha here is 1, so just adds each context clue
            sub_count1 += ONEF
        else:
            break
    
    #take avg instead of sum        
    if sub_count1 > (<REAL_t>0.5):
        sub_inv_count1 = ONEF/sub_count1

    sscal(&size, &sub_inv_count1, sub_neu, &ONE)

    memset(cc_sub_sum1, 0, size * cython.sizeof(REAL_t))
    
    our_saxpy(&size, &sp_alph, neu1, &ONE, cc_sub_sum1, &ONE)
    our_saxpy(&size, &sp_one_minus_alph, sub_neu, &ONE, cc_sub_sum1, &ONE)
    
    #with gil:
        #printf('alph', &sp_alph)
        #print('alph',sp_alph)
        #printf('1 - alph',&sp_one_minus_alph)
        #print('1 - alph',sp_one_minus_alph)
        #print('---')
    
    
    for d in range(negative+1):
        if d == 0:
            target_index = word_index
            label = ONEF
        else:
            target_index = bisect_left(cum_table, (next_random >> 16) % cum_table[cum_table_len-1], 0, cum_table_len)
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            #if target_index == word_index:
            #    continue
                
            #if target_index == word2_index: #Raj: added by me, to prevent neg sample being picked that matches word context clues are trying to estimate.
            #    continue
            
            '''
            while target_index == word_index or target_index == word2_index:
                target_index = bisect_left(cum_table, (next_random >> 16) % cum_table[cum_table_len-1], 0, cum_table_len)
                next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            '''
            
            label = <REAL_t>0.0


        #calculate subword of row2/neg sample

                
                

        row2 = target_index * size
        
        
        #cc and word -------------------------------------------------------------
        
        f_dot = our_dot(&size, neu1, &ONE, &syn1neg[row2], &ONE)   
        #if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
            #skipped_pairs[0] = skipped_pairs[0] + 1
            #continue
            
        if f_dot <= -MAX_EXP:
            f_dot = -MAX_EXP
        if f_dot >= MAX_EXP:
            f_dot = MAX_EXP
            
        f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]  #this is basically applying sigmoid function but using EXP_TABLE for fast calculations
        g = (label - f) * alpha   #so this is label - sigmoid(u*v) * lr
        #------------------------------------------------------------------------
        
        
        #sub and word -------------------------------------------------------------
        
        w_s_dot = our_dot(&size, sub_neu, &ONE, &syn1neg[row2], &ONE)   
        #if w_s_dot <= -MAX_EXP or w_s_dot >= MAX_EXP:
            #skipped_pairs[0] = skipped_pairs[0] + 1
            #continue
        
        if w_s_dot <= -MAX_EXP:
            w_s_dot = -MAX_EXP
        if w_s_dot >= MAX_EXP:        
            w_s_dot = MAX_EXP
            
        f_w_s = EXP_TABLE[<int>((w_s_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]  #this is basically applying sigmoid function but using EXP_TABLE for fast calculations
        g_w_s = (label - f_w_s) * alpha   #so this is label - sigmoid(u*v) * lr
        #------------------------------------------------------------------------
        
        #alph*cc+(1-alph)*sub and word -------------------------------------------------------------
        
        cc_s_w_dot = our_dot(&size, cc_sub_sum1, &ONE, &syn1neg[row2], &ONE)    
        #if cc_s_w_dot <= -MAX_EXP or cc_s_w_dot >= MAX_EXP:
        #    skipped_pairs[0] = skipped_pairs[0] + 1
        #    continue
        
        if cc_s_w_dot <= -MAX_EXP:
            cc_s_w_dot = -MAX_EXP        
        if cc_s_w_dot >= MAX_EXP:
            cc_s_w_dot = MAX_EXP        
        
        
        f_cc_s_w = EXP_TABLE[<int>((cc_s_w_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]  #this is basically applying sigmoid function but using EXP_TABLE for fast calculations
        g_cc_s_w = (label - f_cc_s_w) * alpha   #so this is label - sigmoid(u*v) * lr
        #------------------------------------------------------------------------
        

        if _compute_loss == 1:
            f_dot = (f_dot if d == 0  else -f_dot)
            if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                continue
            log_e_f_dot = LOG_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
            _running_training_loss_param[0] = _running_training_loss_param[0] - log_e_f_dot
        
        our_saxpy(&size, &g, &syn1neg[row2], &ONE, work, &ONE)  #LHS cc word
        
        our_saxpy(&size, &g_w_s, &syn1neg[row2], &ONE, sub_work, &ONE)  #LHS sub word
        
        #3rd pair (cc + sub, word), need to scale by 2!
        
        g_cc_s_w_2_scaled_alph = g_cc_s_w * sp_alph    
        g_cc_s_w_2_scaled_one_minus_alph = g_cc_s_w * sp_one_minus_alph         

        our_saxpy(&size, &g_cc_s_w_2_scaled_alph, &syn1neg[row2], &ONE, work, &ONE)  #LHS cc word
        
        our_saxpy(&size, &g_cc_s_w_2_scaled_one_minus_alph, &syn1neg[row2], &ONE, sub_work, &ONE)  #LHS sub word
        
        #handle alpha!!!! in LHS
        #derive for sp_alph:     [avg_cc dot w   -    avg_sub dot w] * g_cc_s_w
        sp_alph_grad = sp_alph_grad + ( (f_dot - w_s_dot) * g_cc_s_w )    
        
        #RHS cc word and sub word ------------------------------------------
        our_saxpy(&size, &g, neu1, &ONE, &syn1neg[row2], &ONE)
        our_saxpy(&size, &g_w_s, sub_neu, &ONE, &syn1neg[row2], &ONE)
        our_saxpy(&size, &g_cc_s_w, cc_sub_sum1, &ONE, &syn1neg[row2], &ONE)
        #------------------------------------------------------
        

    
    sscal(&size, &cc_inv_count, work, &ONE)  #need to scale derivative by average I think?  ***
    sscal(&size, &sub_inv_count1, sub_work, &ONE)  #need to scale derivative by average I think?  ***
    
    for qqqq in range(0, MAX_CONTEXT_SIZE):
        if cc_set[qqqq] != -1:
            our_saxpy(&size, &word_locks[word2_index], work, &ONE, &syn3cc[cc_set[qqqq] * size], &ONE)  #updates syn3cc(context clues?)  #LHS cc word and cc sub
            
    for e2 in range(0, subword_list_len):
        sub_ind = index_subword_breakdown[(word2_index * subword_list_len)+e2]  
        if sub_ind != sub_padding_num:
            our_saxpy(&size, &ONEF, sub_work, &ONE, &syn2subword[sub_ind * size], &ONE)  #our alpha here is 1, so just adds each context clue

        else:
            break
    
    single_parameter[0] = single_parameter[0] + sp_alph_grad
    return next_random


#------------------------------------
#---



# (cc word) (word sub) (cc sub)
cdef unsigned long long fast_sentence_sg_neg_ALL_3_PAIRS(
    const int negative, np.uint32_t *cum_table, unsigned long long cum_table_len,
    REAL_t *syn3cc, REAL_t *syn1neg, REAL_t *syn2subword, const int size, const np.uint32_t word_index,
    const np.uint32_t word2_index, const REAL_t alpha, REAL_t *work, REAL_t *word_work, REAL_t *neu1, REAL_t *sub_neu,
    unsigned long long next_random, REAL_t *word_locks,
    const int _compute_loss, REAL_t *_running_training_loss_param, np.uint32_t* cc_set, int* index_subword_breakdown, int subword_list_len, int sub_padding_num, int *skipped_pairs) nogil:


    #syn3cc: context clue   #neu1 holds sum
    #syn1neg: word
    #syn2subword:  subword  #sub_neu holds sum

    cdef long long a
    cdef long long row1 = word2_index * size, row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, label, f_dot, log_e_f_dot, cc_s_dot, f_cc_s, g_cc_s, w_s_dot, f_w_s, g_w_s, cc_count, sub_count, cc_inv_count = 1.0, sub_inv_count = 1.0
    cdef np.uint32_t target_index
    cdef int d, m, qqqq, q, yyyy, sub_ind

    memset(work, 0, size * cython.sizeof(REAL_t))  #I think work is for storing gradient updates for syn3cc(with all negs/target word)
                                                   #in our case, work will store all gradients for each context clue I guess?
                                                   #I think memset is analoguous to np.zeros, although I'm not sure

    memset(word_work, 0, size * cython.sizeof(REAL_t)) #for LHS word vector

    #Raj: it would appear that row 1 is matched with word2 and row 2 is matched with word1/neg samples for some reason
    #Raj:  EXP_TABLE is precomputed sigmoid, LOG_TABLE is log sigmoid


    #Raj calculate sum of context clues instead of syn3cc[row1], use neu1
    #based on how cbow does it
    memset(neu1, 0, size * cython.sizeof(REAL_t))
    cc_count = <REAL_t>0.0
    for m in range(0, MAX_CONTEXT_SIZE):
        
        if cc_set[m] != -1:
            cc_count += ONEF
            our_saxpy(&size, &ONEF, &syn3cc[cc_set[m] * size], &ONE, neu1, &ONE)  #our alpha here is 1, so just adds each context clue
    
    #take avg instead of sum    
    if cc_count > (<REAL_t>0.5):
        cc_inv_count = ONEF/cc_count
 
    sscal(&size, &cc_inv_count, neu1, &ONE)
    
    for d in range(negative+1):
        if d == 0:
            target_index = word_index
            label = ONEF
        else:
            target_index = bisect_left(cum_table, (next_random >> 16) % cum_table[cum_table_len-1], 0, cum_table_len)
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            #if target_index == word_index:
            #    continue
                
            #if target_index == word2_index: #Raj: added by me, to prevent neg sample being picked that matches word context clues are trying to estimate.
            #    continue
            
            '''
            while target_index == word_index or target_index == word2_index:
                target_index = bisect_left(cum_table, (next_random >> 16) % cum_table[cum_table_len-1], 0, cum_table_len)
                next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            '''
            
            label = <REAL_t>0.0


        #calculate subword of row2/neg sample
        memset(sub_neu, 0, size * cython.sizeof(REAL_t))  #reset subword sum
        sub_count = <REAL_t>0.0
        sub_inv_count = 1.0
        for q in range(0, subword_list_len):
            sub_ind = index_subword_breakdown[(target_index * subword_list_len)+q]
            if sub_ind != sub_padding_num:
                our_saxpy(&size, &ONEF, &syn2subword[sub_ind * size], &ONE, sub_neu, &ONE)  #our alpha here is 1, so just adds each context clue
                sub_count += ONEF
            else:
                break
        
        #take avg instead of sum        
        if sub_count > (<REAL_t>0.5):
            sub_inv_count = ONEF/sub_count
 
        sscal(&size, &sub_inv_count, sub_neu, &ONE)
                
                

        row2 = target_index * size
        
        
        #cc and word -------------------------------------------------------------
        
        f_dot = our_dot(&size, neu1, &ONE, &syn1neg[row2], &ONE)   
        if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:

            skipped_pairs[0] = skipped_pairs[0] + 1

            continue
            
        f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]  #this is basically applying sigmoid function but using EXP_TABLE for fast calculations
        g = (label - f) * alpha   #so this is label - sigmoid(u*v) * lr
        #------------------------------------------------------------------------
        
        
        
        #cc and sub -------------------------------------------------------------
        cc_s_dot = our_dot(&size, neu1, &ONE, sub_neu, &ONE)   
        if cc_s_dot <= -MAX_EXP or cc_s_dot >= MAX_EXP:

            skipped_pairs[0] = skipped_pairs[0] + 1

           
            continue

        f_cc_s = EXP_TABLE[<int>((cc_s_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]  #this is basically applying sigmoid function but using EXP_TABLE for fast calculations
        g_cc_s = (label - f_cc_s) * alpha   #so this is label - sigmoid(u*v) * lr
        #------------------------------------------------------------------------
        

        #word and sub -------------------------------------------------------------
        
        w_s_dot = our_dot(&size, &syn1neg[row1], &ONE, sub_neu, &ONE)   
        if w_s_dot <= -MAX_EXP or w_s_dot >= MAX_EXP:
            skipped_pairs[0] = skipped_pairs[0] + 1

            continue
        f_w_s = EXP_TABLE[<int>((w_s_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]  #this is basically applying sigmoid function but using EXP_TABLE for fast calculations
        g_w_s = (label - f_w_s) * alpha   #so this is label - sigmoid(u*v) * lr
        #------------------------------------------------------------------------
        

        if _compute_loss == 1:
            f_dot = (f_dot if d == 0  else -f_dot)
            if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                continue
            log_e_f_dot = LOG_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
            _running_training_loss_param[0] = _running_training_loss_param[0] - log_e_f_dot


        
        our_saxpy(&size, &g, &syn1neg[row2], &ONE, work, &ONE)  #LHS cc word
        
        our_saxpy(&size, &g_cc_s, sub_neu, &ONE, work, &ONE)  #LHS cc sub
        our_saxpy(&size, &g_w_s, sub_neu, &ONE, word_work, &ONE)  #LHS word sub 
        

        #RHS do cc word LAST so we aren't changing syn1neg first 

        #NEED TO SCALE SUBWORDS BECAUSE OF AVERAGE! However, we don't want to change neu1 or syn1neg, so instead we scale g_cc_s and g_w_s? ***
        g_cc_s = g_cc_s * sub_inv_count
        g_w_s = g_w_s * sub_inv_count
        
        #RHS cc sub  &&    word sub   --------------------------------------------
        for yyyy in range(0, subword_list_len):
            sub_ind = index_subword_breakdown[(target_index * subword_list_len)+yyyy]
            if sub_ind != sub_padding_num:
                
                
                our_saxpy(&size, &g_cc_s, neu1, &ONE, &syn2subword[sub_ind * size], &ONE)  #RHS cc sub
                our_saxpy(&size, &g_w_s, &syn1neg[row1], &ONE, &syn2subword[sub_ind * size], &ONE)  #RHS word sub
                
            else:
                break
        #----------------------------------------------------------------------------
        

        #RHS cc word ------------------------------------------
        our_saxpy(&size, &g, neu1, &ONE, &syn1neg[row2], &ONE)
        #------------------------------------------------------
        

    
    sscal(&size, &cc_inv_count, work, &ONE)  #need to scale derivative by average I think?  ***
    
    for qqqq in range(0, MAX_CONTEXT_SIZE):
        if cc_set[qqqq] != -1:
            our_saxpy(&size, &word_locks[word2_index], work, &ONE, &syn3cc[cc_set[qqqq] * size], &ONE)  #updates syn3cc(context clues?)  #LHS cc word and cc sub
    
    
    our_saxpy(&size, &word_locks[word2_index], word_work, &ONE, &syn1neg[row1], &ONE) #LHS word sub
    

    return next_random


#----------------------------------------------------------------------------------------------
# (cc1+sub1, word2) (cc1+word1, sub2) (cc1, word2+sub2)
cdef unsigned long long fast_sentence_sg_neg_INTERNAL_SUM(
    const int negative, np.uint32_t *cum_table, unsigned long long cum_table_len,
    REAL_t *syn3cc, REAL_t *syn1neg, REAL_t *syn2subword, const int size, const np.uint32_t word_index,
    const np.uint32_t word2_index, const REAL_t alpha, REAL_t *word1_work, REAL_t *cc_work, REAL_t *sub1_work,  REAL_t *neu1, REAL_t *sub_neu,
    unsigned long long next_random, REAL_t *word_locks,
    const int _compute_loss, REAL_t *_running_training_loss_param, np.uint32_t* cc_set, int* index_subword_breakdown,
    int subword_list_len, int sub_padding_num, int *skipped_pairs, REAL_t *cc_sub_sum1, REAL_t *cc_word_sum1, REAL_t *word_sub2) nogil:


    #syn3cc: context clue   #neu1 holds sum
    #syn1neg: word
    #syn2subword:  subword  #sub_neu holds sum

    cdef long long a
    cdef long long row1 = word2_index * size, row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, label, f_dot, log_e_f_dot, cc_s_dot, f_cc_s, g_cc_s, w_s_dot, f_w_s, g_w_s, cc_count, sub_count1, sub_count2, cc_inv_count = 1.0,  g_2_scaled, g_cc_s_2_scaled, g_cc_s_scaled, g_w_s_scaled , g_w_s_scaled_2_scaled, g_w_s_2_scaled , sub_inv_count1 = 1.0, sub_inv_count2 = 1.0
    cdef np.uint32_t target_index
    cdef int d, m, qqqq, q, yyyy, sub_ind, e, iii
    cdef REAL_t sub_mult = 1.0
    cdef REAL_t  two_divisor = 1.0/2.0

    memset(word1_work, 0, size * cython.sizeof(REAL_t)) #keeps track of word1 gradient
    memset(cc_work, 0, size * cython.sizeof(REAL_t)) #keeps cc_work
    memset(sub1_work, 0, size * cython.sizeof(REAL_t)) #keeps sub1_work
    
    #Raj: it would appear that row 1 is matched with word2 and row 2 is matched with word1/neg samples for some reason
    #Raj:  EXP_TABLE is precomputed sigmoid, LOG_TABLE is log sigmoid


    #Raj calculate sum of context clues instead of syn3cc[row1], use neu1
    #based on how cbow does it
    memset(neu1, 0, size * cython.sizeof(REAL_t))
    cc_count = <REAL_t>0.0
    for m in range(0, MAX_CONTEXT_SIZE):
        
        if cc_set[m] != -1:
            cc_count += ONEF
            our_saxpy(&size, &ONEF, &syn3cc[cc_set[m] * size], &ONE, neu1, &ONE)  #our alpha here is 1, so just adds each context clue
    
    #take avg instead of sum    
    if cc_count > (<REAL_t>0.5):
        cc_inv_count = ONEF/cc_count
 
    sscal(&size, &cc_inv_count, neu1, &ONE)
    
    
    #Store cc1 + sub1 in cc_sub_sum1 ---------------------------------------------
    memset(cc_sub_sum1, 0, size * cython.sizeof(REAL_t))
    
    #add subword first
    sub_count1 = <REAL_t>0.0
    sub_inv_count1 = 1.0
    for e in range(0, subword_list_len):
        sub_ind = index_subword_breakdown[(word2_index * subword_list_len)+e]
        if sub_ind != sub_padding_num:
            our_saxpy(&size, &ONEF, &syn2subword[sub_ind * size], &ONE, cc_sub_sum1, &ONE)  #our alpha here is 1, so just adds each context clue
            sub_count1 += ONEF
        else:
            break
    
    #take avg instead of sum        
    if sub_count1 > (<REAL_t>0.5):
        sub_inv_count1 = ONEF/sub_count1

    sscal(&size, &sub_inv_count1, cc_sub_sum1, &ONE)

    #add in context clues
    our_saxpy(&size, &ONEF, neu1, &ONE, cc_sub_sum1, &ONE)

    #divide by two!

    sscal(&size, &two_divisor, cc_sub_sum1, &ONE)

    
    #----------------------------------------------------------------------------
    
    
    #Store cc1 + word1 in cc_word_sum1 ------------------------------------------
    memset(cc_word_sum1, 0, size * cython.sizeof(REAL_t))
    #add cc1
    our_saxpy(&size, &ONEF, neu1, &ONE, cc_word_sum1, &ONE)    
    #add word1    
    our_saxpy(&size, &ONEF, &syn1neg[row1], &ONE, cc_word_sum1, &ONE)

    sscal(&size, &two_divisor, cc_word_sum1, &ONE)  #divide by 2 in sum    
    #---------------------------------------------------------------------------
    
    for d in range(negative+1):
        if d == 0:
            target_index = word_index
            label = ONEF
        else:
            target_index = bisect_left(cum_table, (next_random >> 16) % cum_table[cum_table_len-1], 0, cum_table_len)
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            #if target_index == word_index:
            #    continue
                
            #if target_index == word2_index: #Raj: added by me, to prevent neg sample being picked that matches word context clues are trying to estimate.
            #    continue
            
            '''
            while target_index == word_index or target_index == word2_index:
                target_index = bisect_left(cum_table, (next_random >> 16) % cum_table[cum_table_len-1], 0, cum_table_len)
                next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            '''
            
            label = <REAL_t>0.0


        #calculate subword of row2/neg sample
        memset(sub_neu, 0, size * cython.sizeof(REAL_t))  #reset subword sum
        
        
        sub_count2 = <REAL_t>0.0
        sub_inv_count2 = 1.0
        for q in range(0, subword_list_len):
            sub_ind = index_subword_breakdown[(target_index * subword_list_len)+q]
            if sub_ind != sub_padding_num:
                our_saxpy(&size, &ONEF, &syn2subword[sub_ind * size], &ONE, sub_neu, &ONE)  #our alpha here is 1, so just adds each context clue
                sub_count2 += ONEF
            else:
                break
        
        #take avg instead of sum        
        if sub_count2 > (<REAL_t>0.5):
            sub_inv_count2 = ONEF/sub_count2
 
        sscal(&size, &sub_inv_count2, sub_neu, &ONE)
                                
        row2 = target_index * size
        
        
        #Store word2 + sub2 in word_sub2 ------------------------------------------
        memset(word_sub2, 0, size * cython.sizeof(REAL_t))
        #add word2
        our_saxpy(&size, &ONEF, &syn1neg[row2], &ONE, word_sub2, &ONE)    
        #add sub2    
        our_saxpy(&size, &ONEF, sub_neu, &ONE, word_sub2, &ONE)

        sscal(&size, &two_divisor, word_sub2, &ONE)  #divide by 2 in sum            
        #---------------------------------------------------------------------------
        
     
        # (cc1+sub1, word2) -------------------------------------------------------------            
        f_dot = our_dot(&size, cc_sub_sum1, &ONE, &syn1neg[row2], &ONE)   
        
        '''
        if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:

            skipped_pairs[0] = skipped_pairs[0] + 1

            continue
        '''
        
        if f_dot <= -MAX_EXP:
            f_dot = -MAX_EXP
        if f_dot >= MAX_EXP:
            f_dot = MAX_EXP
            
        f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]  #this is basically applying sigmoid function but using EXP_TABLE for fast calculations
        g = (label - f) * alpha   #so this is label - sigmoid(u*v) * lr
        #------------------------------------------------------------------------
        
    

        # (cc1+word1, sub2) ------------------------------------------------------------- 
        cc_s_dot = our_dot(&size, cc_word_sum1, &ONE, sub_neu, &ONE)   
        
        
        '''
        if cc_s_dot <= -MAX_EXP or cc_s_dot >= MAX_EXP:

            skipped_pairs[0] = skipped_pairs[0] + 1

           
            continue
        '''
        if cc_s_dot <= -MAX_EXP:
            cc_s_dot = -MAX_EXP

        if cc_s_dot >= MAX_EXP:
            cc_s_dot = MAX_EXP

        f_cc_s = EXP_TABLE[<int>((cc_s_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]  #this is basically applying sigmoid function but using EXP_TABLE for fast calculations
        g_cc_s = (label - f_cc_s) * alpha   #so this is label - sigmoid(u*v) * lr
        #------------------------------------------------------------------------

 
            
        # (cc1, word2+sub2)        
        w_s_dot = our_dot(&size, neu1, &ONE, word_sub2, &ONE)   
        
        '''
        if w_s_dot <= -MAX_EXP or w_s_dot >= MAX_EXP:
            skipped_pairs[0] = skipped_pairs[0] + 1

            continue
        '''
        
        if w_s_dot <= -MAX_EXP:
            w_s_dot = -MAX_EXP
        if w_s_dot >= MAX_EXP:
            w_s_dot = MAX_EXP
        
        f_w_s = EXP_TABLE[<int>((w_s_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]  #this is basically applying sigmoid function but using EXP_TABLE for fast calculations
        g_w_s = (label - f_w_s) * alpha   #so this is label - sigmoid(u*v) * lr
        #------------------------------------------------------------------------
        

        if _compute_loss == 1:
            f_dot = (f_dot if d == 0  else -f_dot)
            if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                continue
            log_e_f_dot = LOG_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
            _running_training_loss_param[0] = _running_training_loss_param[0] - log_e_f_dot


            
        g_2_scaled = g * two_divisor    #for cc1 + sub, divide gradient by 2       
        g_cc_s_2_scaled = g_cc_s * two_divisor  #for cc1 + word, divide gradient by 2


        #the cc and sub normalization will happen later! (when the work is added to cc_work, sub1_work)
        our_saxpy(&size, &g_2_scaled, &syn1neg[row2], &ONE, cc_work, &ONE)  #LHS (cc1+sub1, word2)
        our_saxpy(&size, &g_2_scaled, &syn1neg[row2], &ONE, sub1_work, &ONE)  #LHS (cc1+sub1, word2) 
        
        our_saxpy(&size, &g_cc_s_2_scaled, sub_neu, &ONE, cc_work, &ONE)  #LHS (cc1+word1, sub2)
        our_saxpy(&size, &g_cc_s_2_scaled, sub_neu, &ONE, word1_work, &ONE)  #LHS (cc1+word1, sub2) 
                
        our_saxpy(&size, &g_w_s, word_sub2, &ONE, cc_work, &ONE)  #LHS (cc1, word2+sub2) 
        
        
        
        #RHS do cc word LAST so we aren't changing syn1neg first 

        #NEED TO SCALE SUBWORDS BECAUSE OF AVERAGE! However, we don't want to change neu1 or syn1neg, so instead we scale g_cc_s and g_w_s? ***
        g_cc_s_scaled = g_cc_s * sub_inv_count2  #scale by sub 2
        g_w_s_scaled = g_w_s * sub_inv_count2    #scale by sub 2

        
        g_w_s_scaled_2_scaled = g_w_s_scaled * two_divisor #for RHS sub in sub sums ( word2 + sub2)  
        g_w_s_2_scaled = g_w_s * two_divisor  #for RHS word in word sums (word2 + sub2)
        
        #RHS cc sub  &&    word sub   --------------------------------------------
        for yyyy in range(0, subword_list_len):
            sub_ind = index_subword_breakdown[(target_index * subword_list_len)+yyyy]
            if sub_ind != sub_padding_num:
                                
                our_saxpy(&size, &g_cc_s_scaled, cc_word_sum1, &ONE, &syn2subword[sub_ind * size], &ONE)  #RHS (cc1+word1, sub2)  SUB2
                our_saxpy(&size, &g_w_s_scaled_2_scaled, neu1, &ONE, &syn2subword[sub_ind * size], &ONE)  #RHS (cc1, word2+sub2)   SUB2
                
            else:
                break
        #----------------------------------------------------------------------------
        

        #RHS  ----------------------------------------------------
        our_saxpy(&size, &g, cc_sub_sum1, &ONE, &syn1neg[row2], &ONE)  #RHS (cc1+sub1, word2)  #WORD2
        our_saxpy(&size, &g_w_s_2_scaled, neu1, &ONE, &syn1neg[row2], &ONE)  #RHS (cc1, word2+sub2)  WORD2
        #------------------------------------------------------
        

    sscal(&size, &cc_inv_count, cc_work, &ONE)  #scale cc1
    sscal(&size, &sub_inv_count1, sub1_work, &ONE)  #scale sub1
    
    
    for qqqq in range(0, MAX_CONTEXT_SIZE):
        if cc_set[qqqq] != -1:
            our_saxpy(&size, &word_locks[word2_index], cc_work, &ONE, &syn3cc[cc_set[qqqq] * size], &ONE)  #updates syn3cc(context clues?)  #LHS cc word and cc sub
            
            
    for iii in range(0, subword_list_len):
        sub_ind = index_subword_breakdown[(word2_index * subword_list_len)+iii]    
        if sub_ind != sub_padding_num:
            our_saxpy(&size, &sub_mult, sub1_work, &ONE, &syn2subword[sub_ind * size], &ONE) 

    
    our_saxpy(&size, &word_locks[word2_index], word1_work, &ONE, &syn1neg[row1], &ONE) #LHS word sub

    return next_random
#----------------------------------------------------------------------------------------------




cdef void fast_sentence_cbow_hs(
    const np.uint32_t *word_point, const np.uint8_t *word_code, int codelens[MAX_SENTENCE_LEN],
    REAL_t *neu1, REAL_t *syn3cc, REAL_t *syn1, const int size,
    const np.uint32_t indexes[MAX_SENTENCE_LEN], const REAL_t alpha, REAL_t *work,
    int i, int j, int k, int cbow_mean, REAL_t *word_locks,
    const int _compute_loss, REAL_t *_running_training_loss_param) nogil:

    cdef long long a, b
    cdef long long row2, sgn
    cdef REAL_t f, g, count, inv_count = 1.0, f_dot, lprob
    cdef int m

    memset(neu1, 0, size * cython.sizeof(REAL_t))
    count = <REAL_t>0.0
    for m in range(j, k):
        if m == i:
            continue
        else:
            count += ONEF
            our_saxpy(&size, &ONEF, &syn3cc[indexes[m] * size], &ONE, neu1, &ONE)
    if count > (<REAL_t>0.5):
        inv_count = ONEF/count
    if cbow_mean:
        sscal(&size, &inv_count, neu1, &ONE)  # (does this need BLAS-variants like saxpy?)

    memset(work, 0, size * cython.sizeof(REAL_t))
    for b in range(codelens[i]):
        row2 = word_point[b] * size
        f_dot = our_dot(&size, neu1, &ONE, &syn1[row2], &ONE)
        if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (1 - word_code[b] - f) * alpha

        if _compute_loss == 1:
            sgn = (-1)**word_code[b]  # ch function: 0-> 1, 1 -> -1
            lprob = -1*sgn*f_dot
            if lprob <= -MAX_EXP or lprob >= MAX_EXP:
                continue
            lprob = LOG_TABLE[<int>((lprob + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
            _running_training_loss_param[0] = _running_training_loss_param[0] - lprob

        our_saxpy(&size, &g, &syn1[row2], &ONE, work, &ONE)
        our_saxpy(&size, &g, neu1, &ONE, &syn1[row2], &ONE)

    if not cbow_mean:  # divide error over summed window vectors
        sscal(&size, &inv_count, work, &ONE)  # (does this need BLAS-variants like saxpy?)

    for m in range(j, k):
        if m == i:
            continue
        else:
            our_saxpy(&size, &word_locks[indexes[m]], work, &ONE, &syn3cc[indexes[m] * size], &ONE)


cdef unsigned long long fast_sentence_cbow_neg(
    const int negative, np.uint32_t *cum_table, unsigned long long cum_table_len, int codelens[MAX_SENTENCE_LEN],
    REAL_t *neu1,  REAL_t *syn3cc, REAL_t *syn1neg, const int size,
    const np.uint32_t indexes[MAX_SENTENCE_LEN], const REAL_t alpha, REAL_t *work,
    int i, int j, int k, int cbow_mean, unsigned long long next_random, REAL_t *word_locks,
    const int _compute_loss, REAL_t *_running_training_loss_param) nogil:

    cdef long long a
    cdef long long row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, count, inv_count = 1.0, label, log_e_f_dot, f_dot
    cdef np.uint32_t target_index, word_index
    cdef int d, m

    word_index = indexes[i]

    memset(neu1, 0, size * cython.sizeof(REAL_t))
    count = <REAL_t>0.0
    for m in range(j, k):
        if m == i:
            continue
        else:
            count += ONEF
            our_saxpy(&size, &ONEF, &syn3cc[indexes[m] * size], &ONE, neu1, &ONE)
    if count > (<REAL_t>0.5):
        inv_count = ONEF/count
    if cbow_mean:
        sscal(&size, &inv_count, neu1, &ONE)  # (does this need BLAS-variants like saxpy?)

    memset(work, 0, size * cython.sizeof(REAL_t))

    for d in range(negative+1):
        if d == 0:
            target_index = word_index
            label = ONEF
        else:
            target_index = bisect_left(cum_table, (next_random >> 16) % cum_table[cum_table_len-1], 0, cum_table_len)
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == word_index:
                continue
            label = <REAL_t>0.0

        row2 = target_index * size
        f_dot = our_dot(&size, neu1, &ONE, &syn1neg[row2], &ONE)
        if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (label - f) * alpha

        if _compute_loss == 1:
            f_dot = (f_dot if d == 0  else -f_dot)
            if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                continue
            log_e_f_dot = LOG_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
            _running_training_loss_param[0] = _running_training_loss_param[0] - log_e_f_dot

        our_saxpy(&size, &g, &syn1neg[row2], &ONE, work, &ONE)
        our_saxpy(&size, &g, neu1, &ONE, &syn1neg[row2], &ONE)

    if not cbow_mean:  # divide error over summed window vectors
        sscal(&size, &inv_count, work, &ONE)  # (does this need BLAS-variants like saxpy?)

    for m in range(j,k):
        if m == i:
            continue
        else:
            our_saxpy(&size, &word_locks[indexes[m]], work, &ONE, &syn3cc[indexes[m]*size], &ONE)

    return next_random


def train_batch_sg(model, sentences, alpha, _work, _word_work, _neu1, _sub_neu, compute_loss, context_size, cc_sampling, pairs):
    #0 is regular sample, 1 is no sampling, 2 is flexible
    cdef int hs = model.hs
    cdef int negative = model.negative
    cdef int sample = (model.vocabulary.sample != 0)
    cdef int cc_sampling_int = cc_sampling
    cdef int pairs_int = pairs

    cdef int _compute_loss = (1 if compute_loss == True else 0)
    cdef REAL_t _running_training_loss = model.running_training_loss
    cdef int *_skipped_pairs = <int *>(np.PyArray_DATA(model.skipped_pairs))

    #cdef REAL_t *syn0 = <REAL_t *>(np.PyArray_DATA(model.wv.vectors))
    cdef REAL_t *word_locks = <REAL_t *>(np.PyArray_DATA(model.trainables.vectors_lockf))
    cdef REAL_t *work
    cdef REAL_t _alpha = alpha
    cdef int size = model.wv.vector_size

    cdef int codelens[MAX_SENTENCE_LEN]
    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
    cdef np.uint32_t reduced_windows[MAX_SENTENCE_LEN]
    cdef int sentence_idx[MAX_SENTENCE_LEN + 1]
    cdef int window = model.window

    cdef int i, j, k, qqq, offset, input_ind, i2, low_ind, high_ind, rrr_ind, rrr, cc_set_count
    cdef int effective_words = 0, effective_sentences = 0
    cdef int sent_idx, idx_start, idx_end

    # For hierarchical softmax
    cdef REAL_t *syn1
    cdef np.uint32_t *points[MAX_SENTENCE_LEN]
    cdef np.uint8_t *codes[MAX_SENTENCE_LEN]

    # For negative sampling
    cdef REAL_t *syn3cc
    cdef REAL_t *syn1neg
    cdef np.uint32_t *cum_table
    cdef unsigned long long cum_table_len
    # for sampling (negative and frequent-word downsampling)
    cdef unsigned long long next_random
    
    
    # For subword stuff
    cdef int *index_subword_breakdown = <int *>(np.PyArray_DATA(model.sub.index_subword_breakdown))
    cdef int subword_list_len = len(model.sub.index_subword_breakdown[0])

    #print(len(model.sub.index_subword_breakdown[0]))

    cdef int sub_padding_num = model.sub.padding_number
    cdef REAL_t *syn2subword
    #print(index_subword_breakdown[0])
    #print(index_subword_breakdown[5*subword_list_len])
    
    
    #Raj: for flexible and no sampling
    cdef np.uint32_t full_indexes[MAX_SENTENCE_LEN * 2]  #keeps track of original corpus for flexible sampling etc
    cdef int full_effective_words = 0  #keep track of where we are in above matrix
    cdef np.uint32_t corresponding_index_tracker[MAX_SENTENCE_LEN] #This one keeps track of where the word in indexes (sampled) is in full_indexes (unsampled)
    
    
    cdef np.uint32_t cc_set[MAX_CONTEXT_SIZE] #Raj this stores indexes of the context clues

    for ind in range(len(cc_set)):
        cc_set[ind] = -1  #fill in with -1 values
   
   
    context_half = context_size / 2
    int_context_half = <int> context_half
    
    #print('hello')
    
    
    if hs:
        syn1 = <REAL_t *>(np.PyArray_DATA(model.trainables.syn1))

    if negative:
        syn3cc = <REAL_t *>(np.PyArray_DATA(model.trainables.syn3cc))
        syn1neg = <REAL_t *>(np.PyArray_DATA(model.trainables.syn1neg))
        syn2subword = <REAL_t *>(np.PyArray_DATA(model.trainables.syn2subword))
        cum_table = <np.uint32_t *>(np.PyArray_DATA(model.vocabulary.cum_table))
        cum_table_len = len(model.vocabulary.cum_table)
    if negative or sample:
        next_random = (2**24) * model.random.randint(0, 2**24) + model.random.randint(0, 2**24)

    # convert Python structures to primitive types, so we can release the GIL
    work = <REAL_t *>np.PyArray_DATA(_work)  #Raj: I believe this will be used to sum gradients before applying to final
    word_work = <REAL_t *>np.PyArray_DATA(_word_work)
    neu1 = <REAL_t *>np.PyArray_DATA(_neu1)  #Raj: added in from cbow code to store sum of context clues
    sub_neu = <REAL_t *>np.PyArray_DATA(_sub_neu)  #Raj: added in to store subword sum
    

    # prepare C structures so we can go "full C" and release the Python GIL
    vlookup = model.wv.vocab
    sentence_idx[0] = 0  # indices of the first sentence always start at 0
    for sent in sentences:
        if not sent:
            continue  # ignore empty sentences; leave effective_sentences unchanged
        for token in sent:
            word = vlookup[token] if token in vlookup else None
            
            
            #If UNK for the context clues matters, build full_indexes here without filtering UNK
            #----------------------------------------------------------------------------------
            if full_effective_words < MAX_SENTENCE_LEN * 2:
                if word is None:
                    full_indexes[full_effective_words] = -1
                else:
                    full_indexes[full_effective_words] = word.index
            else:
                #with gil:
                print('over full indexes limit')
            #-----------------------------------------------------------------------------------                    
            
            
            
            if word is None:
                continue  # leaving `effective_words` unchanged = shortening the sentence = expanding the window
            
            '''
            if full_effective_words < MAX_SENTENCE_LEN * 2:
                full_indexes[full_effective_words] = word.index
            
            else:
                #with gil:
                print('over full indexes limit')
            '''    

            
            if sample and word.sample_int < random_int32(&next_random):
                full_effective_words += 1  #should I break if greater than MAX_SENTENCE_LEN ?  This will fill up faster than sampled out ones
                continue
            indexes[effective_words] = word.index
            corresponding_index_tracker[effective_words] = full_effective_words
            #print(token)
            #print(word)
            #print(word.index)
            #print('---------')
            if hs:
                codelens[effective_words] = <int>len(word.code)
                codes[effective_words] = <np.uint8_t *>np.PyArray_DATA(word.code)
                points[effective_words] = <np.uint32_t *>np.PyArray_DATA(word.point)
            effective_words += 1
            full_effective_words += 1  #should I break if greater than MAX_SENTENCE_LEN ?  This will fill up faster than sampled out ones
            
            if effective_words == MAX_SENTENCE_LEN:
                break  # TODO: log warning, tally overflow?

        # keep track of which words go into which sentence, so we don't train
        # across sentence boundaries.
        # indices of sentence number X are between <sentence_idx[X], sentence_idx[X])
        effective_sentences += 1
        sentence_idx[effective_sentences] = effective_words

        if effective_words == MAX_SENTENCE_LEN:
            break  # TODO: log warning, tally overflow?

    #print(effective_words)        
    
    # precompute "reduced window" offsets in a single randint() call
    for i, item in enumerate(model.random.randint(0, window, effective_words)):
        reduced_windows[i] = item  #Raj this decides a window for each word (for skipgram word pairs)

    # release GIL & train on all sentences
    with nogil:
        for sent_idx in range(effective_sentences):
            idx_start = sentence_idx[sent_idx]
            idx_end = sentence_idx[sent_idx + 1]
            for i in range(idx_start, idx_end):
                j = i - window + reduced_windows[i]
                if j < idx_start:
                    j = idx_start
                k = i + window + 1 - reduced_windows[i]
                if k > idx_end:
                    k = idx_end
                    
                
                #---------------------------------------------------------
                #Raj:  here we build context clue set per i (it may be faster to do this all at once instead of each pair)
                
                
                                    
                input_ind = 0
                
                
                #int_context_half
                for qqq in range(0, int_context_half):  #will only iterate through first (context_size) elements of array
                

                    offset = qqq + 1
                        

                    
                    

                    
                    if cc_sampling_int == 0:  #regular
                        if i - offset >= idx_start:
                            cc_set[input_ind] = indexes[i - offset]
                        else:
                            cc_set[input_ind] = -1  #if -1 will be ignored
                            
                        input_ind = input_ind + 1
                        
                        
                        if i + offset < idx_end:
                            cc_set[input_ind] = indexes[i + offset]
                        else:
                            cc_set[input_ind] = -1  #if -1 will be ignored
                        input_ind = input_ind + 1
                    
                    elif cc_sampling_int == 1: #no sampling
                        i2 = corresponding_index_tracker[i]  #where the word is in full_indexes
                        if i2 - offset >= corresponding_index_tracker[idx_start]:
                            cc_set[input_ind] = full_indexes[i2 - offset]
                        else:
                            cc_set[input_ind] = -1  #if -1 will be ignored
                            
                        input_ind = input_ind + 1
                        
                        
                        if i2 + offset < corresponding_index_tracker[idx_end]:
                            cc_set[input_ind] = full_indexes[i2 + offset]
                        else:
                            cc_set[input_ind] = -1  #if -1 will be ignored
                        input_ind = input_ind + 1
                    
                    elif cc_sampling_int == 2:  #flexible
                    
                        #want biggest offset possible
                        
                        if i - offset >=idx_start:
                            low_ind = corresponding_index_tracker[i - offset]

                        if i + offset < idx_end:
                            high_ind = corresponding_index_tracker[i + offset]
                    
                    else:
                         with gil:
                             print('pick a correct context clue sampling method! (cc_sampling)')

                    
                #-------------------------------------------------------
                
                if cc_sampling_int == 2: #flexible
                    #here write for loop that handles this, make sure cc_set is big enough, and handle when its not!!!
                                        
                    
                    
                    
                    #with gil:
                    #    print(str(high_ind - low_ind + 1))
                    #    print('too large')
                    
                            
                    rrr = low_ind
                    for rrr_ind in range(MAX_CONTEXT_SIZE):
                        if rrr <= high_ind and rrr != corresponding_index_tracker[i]:
                            cc_set[rrr_ind] = full_indexes[rrr]
                            rrr = rrr + 1
                        else:
                            cc_set[rrr_ind] = -1
                
                '''                
                with gil:
                    cc_set_count = 0
                    for yyy in range(len(cc_set)):
                        if cc_set[yyy] != -1:
                            cc_set_count = cc_set_count + 1
                    
                    print(cc_set_count)
                '''
                        
                
                for j in range(j, k):
                    if j == i:
                        continue
                    if hs:
                        #fast_sentence_sg_hs(points[i], codes[i], codelens[i], syn0, syn1, size, indexes[j], _alpha, work, word_locks, _compute_loss, &_running_training_loss)
                        with gil:
                            print('hs not implemented')
                    if negative:

                        # pairs set to 3 is default 
                        if pairs_int == 2:
                            #next_random = fast_sentence_sg_neg(negative, cum_table, cum_table_len, syn3cc, syn1neg, syn2subword, size, indexes[j], indexes[i], _alpha, work, word_work, neu1, sub_neu, next_random, word_locks, _compute_loss, &_running_training_loss, cc_set, index_subword_breakdown, subword_list_len, sub_padding_num, _skipped_pairs)
                            next_random = fast_sentence_sg_neg_ALL_2_PAIRS_MEAN(negative, cum_table, cum_table_len, syn3cc, syn1neg, syn2subword, size, indexes[j], indexes[i], _alpha, work, word_work, neu1, sub_neu, next_random, word_locks, _compute_loss, &_running_training_loss, cc_set, index_subword_breakdown, subword_list_len, sub_padding_num, _skipped_pairs)
                            
                        elif pairs_int == 3:
                            next_random = fast_sentence_sg_neg_ALL_3_PAIRS(negative, cum_table, cum_table_len, syn3cc, syn1neg, syn2subword, size, indexes[j], indexes[i], _alpha, work, word_work, neu1, sub_neu, next_random, word_locks, _compute_loss, &_running_training_loss, cc_set, index_subword_breakdown, subword_list_len, sub_padding_num, _skipped_pairs)

                        
                        elif pairs_int == 0:
                            with gil:
                                print('use a different pair')
                                
                        elif pairs_int == 4:
                            next_random = FASTER_2_PAIRS_MEAN(negative, cum_table, cum_table_len, syn3cc, syn1neg, syn2subword, size, indexes[j], indexes[i], _alpha, work, word_work, neu1, sub_neu, next_random, word_locks, _compute_loss, &_running_training_loss, cc_set, index_subword_breakdown, subword_list_len, sub_padding_num, _skipped_pairs)

                            


                        else:
                            with gil:
                                print('pick valid pairs code, 2 for 2 pair, 3 for 3 pair, 0 for 3 pair internal sum')
                        
                    #with gil:
                        #print(j)
    model.running_training_loss = _running_training_loss
    
    #print('did we return effective words?')
    return effective_words
    
def train_batch_sg2(model, sentences, alpha, _word1_work, _cc_work, _sub1_work, _neu1, _sub_neu, compute_loss, context_size, cc_sampling, pairs, _cc_sub_sum1, _cc_word_sum1, _word_sub2):
    #0 is regular sample, 1 is no sampling, 2 is flexible
    cdef int hs = model.hs
    cdef int negative = model.negative
    cdef int sample = (model.vocabulary.sample != 0)
    cdef int cc_sampling_int = cc_sampling
    cdef int pairs_int = pairs

    cdef int _compute_loss = (1 if compute_loss == True else 0)
    cdef REAL_t _running_training_loss = model.running_training_loss
    cdef int *_skipped_pairs = <int *>(np.PyArray_DATA(model.skipped_pairs))
    cdef REAL_t *_single_par = <REAL_t *>(np.PyArray_DATA(model.single_parameter))

    #cdef REAL_t *syn0 = <REAL_t *>(np.PyArray_DATA(model.wv.vectors))
    cdef REAL_t *word_locks = <REAL_t *>(np.PyArray_DATA(model.trainables.vectors_lockf))
    cdef REAL_t *work
    cdef REAL_t _alpha = alpha
    cdef int size = model.wv.vector_size

    cdef int codelens[MAX_SENTENCE_LEN]
    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
    cdef np.uint32_t reduced_windows[MAX_SENTENCE_LEN]
    cdef int sentence_idx[MAX_SENTENCE_LEN + 1]
    cdef int window = model.window

    cdef int i, j, k, qqq, offset, input_ind, i2, low_ind, high_ind, rrr_ind, rrr, cc_set_count
    cdef int effective_words = 0, effective_sentences = 0
    cdef int sent_idx, idx_start, idx_end

    # For hierarchical softmax
    cdef REAL_t *syn1
    cdef np.uint32_t *points[MAX_SENTENCE_LEN]
    cdef np.uint8_t *codes[MAX_SENTENCE_LEN]

    # For negative sampling
    cdef REAL_t *syn3cc
    cdef REAL_t *syn1neg
    cdef np.uint32_t *cum_table
    cdef unsigned long long cum_table_len
    # for sampling (negative and frequent-word downsampling)
    cdef unsigned long long next_random
    
    
    # For subword stuff
    cdef int *index_subword_breakdown = <int *>(np.PyArray_DATA(model.sub.index_subword_breakdown))
    cdef int subword_list_len = len(model.sub.index_subword_breakdown[0])
    cdef int sub_padding_num = model.sub.padding_number
    cdef REAL_t *syn2subword
    #print(index_subword_breakdown[0])
    #print(index_subword_breakdown[5*subword_list_len])
    
    
    #Raj: for flexible and no sampling
    cdef np.uint32_t full_indexes[MAX_SENTENCE_LEN * 2]  #keeps track of original corpus for flexible sampling etc
    cdef int full_effective_words = 0  #keep track of where we are in above matrix
    cdef np.uint32_t corresponding_index_tracker[MAX_SENTENCE_LEN] #This one keeps track of where the word in indexes (sampled) is in full_indexes (unsampled)
    
 
    cdef np.uint32_t cc_set[MAX_CONTEXT_SIZE] #Raj this stores indexes of the context clues

    for ind in range(len(cc_set)):
        cc_set[ind] = -1  #fill in with -1 values
   
   
    context_half = context_size / 2
    int_context_half = <int> context_half
    
    #print('hello')
    
    
    if hs:
        syn1 = <REAL_t *>(np.PyArray_DATA(model.trainables.syn1))

    if negative:
        syn3cc = <REAL_t *>(np.PyArray_DATA(model.trainables.syn3cc))  
        syn1neg = <REAL_t *>(np.PyArray_DATA(model.trainables.syn1neg))
        syn2subword = <REAL_t *>(np.PyArray_DATA(model.trainables.syn2subword))
        cum_table = <np.uint32_t *>(np.PyArray_DATA(model.vocabulary.cum_table))
        cum_table_len = len(model.vocabulary.cum_table)
    if negative or sample:
        next_random = (2**24) * model.random.randint(0, 2**24) + model.random.randint(0, 2**24)

    # convert Python structures to primitive types, so we can release the GIL

    word1_work = <REAL_t *>np.PyArray_DATA(_word1_work)
    cc_work = <REAL_t *>np.PyArray_DATA(_cc_work)
    sub1_work = <REAL_t *>np.PyArray_DATA(_sub1_work)

    
    neu1 = <REAL_t *>np.PyArray_DATA(_neu1)  #Raj: added in from cbow code to store sum of context clues
    sub_neu = <REAL_t *>np.PyArray_DATA(_sub_neu)  #Raj: added in to store subword sum
    
    
    #for internal_sum_version -----------------------------
    cc_sub_sum1 = <REAL_t *>np.PyArray_DATA(_cc_sub_sum1)
    cc_word_sum1 = <REAL_t *>np.PyArray_DATA(_cc_word_sum1)
    word_sub2 = <REAL_t *>np.PyArray_DATA(_word_sub2)
    #-----------------------------------------------------

    # prepare C structures so we can go "full C" and release the Python GIL
    vlookup = model.wv.vocab
    sentence_idx[0] = 0  # indices of the first sentence always start at 0
    for sent in sentences:
        if not sent:
            continue  # ignore empty sentences; leave effective_sentences unchanged
        for token in sent:
            word = vlookup[token] if token in vlookup else None
            
            
            #If UNK for the context clues matters, build full_indexes here without filtering UNK
            #----------------------------------------------------------------------------------
            if full_effective_words < MAX_SENTENCE_LEN * 2:
                if word is None:
                    full_indexes[full_effective_words] = -1
                else:
                    full_indexes[full_effective_words] = word.index
            else:
                #with gil:
                print('over full indexes limit')
            #-----------------------------------------------------------------------------------                    
            
            
            
            if word is None:
                continue  # leaving `effective_words` unchanged = shortening the sentence = expanding the window
            
            '''
            if full_effective_words < MAX_SENTENCE_LEN * 2:
                full_indexes[full_effective_words] = word.index
            
            else:
                #with gil:
                print('over full indexes limit')
            '''    

            
            if sample and word.sample_int < random_int32(&next_random):
                full_effective_words += 1  #should I break if greater than MAX_SENTENCE_LEN ?  This will fill up faster than sampled out ones
                continue
            indexes[effective_words] = word.index
            corresponding_index_tracker[effective_words] = full_effective_words
            #print(token)
            #print(word)
            #print(word.index)
            #print('---------')
            if hs:
                codelens[effective_words] = <int>len(word.code)
                codes[effective_words] = <np.uint8_t *>np.PyArray_DATA(word.code)
                points[effective_words] = <np.uint32_t *>np.PyArray_DATA(word.point)
            effective_words += 1
            full_effective_words += 1  #should I break if greater than MAX_SENTENCE_LEN ?  This will fill up faster than sampled out ones
            
            if effective_words == MAX_SENTENCE_LEN:
                break  # TODO: log warning, tally overflow?

        # keep track of which words go into which sentence, so we don't train
        # across sentence boundaries.
        # indices of sentence number X are between <sentence_idx[X], sentence_idx[X])
        effective_sentences += 1
        sentence_idx[effective_sentences] = effective_words

        if effective_words == MAX_SENTENCE_LEN:
            break  # TODO: log warning, tally overflow?

    #print(effective_words)        
    
    # precompute "reduced window" offsets in a single randint() call
    for i, item in enumerate(model.random.randint(0, window, effective_words)):
        reduced_windows[i] = item  #Raj this decides a window for each word (for skipgram word pairs)

    # release GIL & train on all sentences
    with nogil:
        for sent_idx in range(effective_sentences):
            idx_start = sentence_idx[sent_idx]
            idx_end = sentence_idx[sent_idx + 1]
            for i in range(idx_start, idx_end):
                j = i - window + reduced_windows[i]
                if j < idx_start:
                    j = idx_start
                k = i + window + 1 - reduced_windows[i]
                if k > idx_end:
                    k = idx_end
                    
                
                #---------------------------------------------------------
                #Raj:  here we build context clue set per i (it may be faster to do this all at once instead of each pair)
                
                
                                    
                input_ind = 0
                
                
                #int_context_half
                for qqq in range(0, int_context_half):  #will only iterate through first (context_size) elements of array
                

                    offset = qqq + 1
                        

                    
                    

                    
                    if cc_sampling_int == 0:  #regular
                        if i - offset >= idx_start:
                            cc_set[input_ind] = indexes[i - offset]
                        else:
                            cc_set[input_ind] = -1  #if -1 will be ignored
                            
                        input_ind = input_ind + 1
                        
                        
                        if i + offset < idx_end:
                            cc_set[input_ind] = indexes[i + offset]
                        else:
                            cc_set[input_ind] = -1  #if -1 will be ignored
                        input_ind = input_ind + 1
                    
                    elif cc_sampling_int == 1: #no sampling
                        i2 = corresponding_index_tracker[i]  #where the word is in full_indexes
                        if i2 - offset >= corresponding_index_tracker[idx_start]:
                            cc_set[input_ind] = full_indexes[i2 - offset]
                        else:
                            cc_set[input_ind] = -1  #if -1 will be ignored
                            
                        input_ind = input_ind + 1
                        
                        
                        if i2 + offset < corresponding_index_tracker[idx_end]:
                            cc_set[input_ind] = full_indexes[i2 + offset]
                        else:
                            cc_set[input_ind] = -1  #if -1 will be ignored
                        input_ind = input_ind + 1
                    
                    elif cc_sampling_int == 2:  #flexible
                    
                        #want biggest offset possible
                        
                        if i - offset >=idx_start:
                            low_ind = corresponding_index_tracker[i - offset]

                        if i + offset < idx_end:
                            high_ind = corresponding_index_tracker[i + offset]
                    
                    else:
                         with gil:
                             print('pick a correct context clue sampling method! (cc_sampling)')

                    
                #-------------------------------------------------------
                
                if cc_sampling_int == 2: #flexible
                    #here write for loop that handles this, make sure cc_set is big enough, and handle when its not!!!
                                        
                    
                    
                    
                    #with gil:
                    #    print(str(high_ind - low_ind + 1))
                    #    print('too large')
                    
                            
                    rrr = low_ind
                    for rrr_ind in range(MAX_CONTEXT_SIZE):
                        if rrr <= high_ind and rrr != corresponding_index_tracker[i]:
                            cc_set[rrr_ind] = full_indexes[rrr]
                            rrr = rrr + 1
                        else:
                            cc_set[rrr_ind] = -1
                
                '''                
                with gil:
                    cc_set_count = 0
                    for yyy in range(len(cc_set)):
                        if cc_set[yyy] != -1:
                            cc_set_count = cc_set_count + 1
                    
                    print(cc_set_count)
                '''
                        
                
                for j in range(j, k):
                    if j == i:
                        continue
                    if hs:
                        #fast_sentence_sg_hs(points[i], codes[i], codelens[i], syn0, syn1, size, indexes[j], _alpha, work, word_locks, _compute_loss, &_running_training_loss)
                        with gil:
                            print('hs not implemented')
                    if negative:

                   
                        if pairs_int == 0:
                            next_random = fast_sentence_sg_neg_INTERNAL_SUM(negative, cum_table, cum_table_len, syn3cc, syn1neg, syn2subword, size, indexes[j], indexes[i], _alpha, word1_work, cc_work, sub1_work, neu1, sub_neu, next_random, word_locks, _compute_loss, &_running_training_loss, cc_set, index_subword_breakdown, subword_list_len, sub_padding_num, _skipped_pairs, cc_sub_sum1, cc_word_sum1, word_sub2)

                        elif pairs_int == 5:
                            next_random = FASTER_3_PAIRS_MEAN(negative, cum_table, cum_table_len, syn3cc, syn1neg, syn2subword, size, indexes[j], indexes[i], _alpha, cc_work, sub1_work, neu1, sub_neu, cc_sub_sum1, next_random, word_locks, _compute_loss, &_running_training_loss, cc_set, index_subword_breakdown, subword_list_len, sub_padding_num, _skipped_pairs)

                        elif pairs_int == 6:
                            next_random =  FASTER_3_PAIRS_MEAN_WITH_SINGLE_PARAMETER(negative, cum_table, cum_table_len, syn3cc, syn1neg, syn2subword, size, indexes[j], indexes[i], _alpha, cc_work, sub1_work, neu1, sub_neu, cc_sub_sum1, next_random, word_locks, _compute_loss, &_running_training_loss, cc_set, index_subword_breakdown, subword_list_len, sub_padding_num, _skipped_pairs, _single_par)
                        else:
                            with gil:
                                print('pick valid pairs code 0 for 3 pair internal sum,  5 for faster_3_pair, 6 for sing paramter 2 pair faster')
                        
                    #with gil:
                        #print(j)
    model.running_training_loss = _running_training_loss
    
    #print('did we return effective words?')
    return effective_words

    
def train_batch_cbow(model, sentences, alpha, _work, _neu1, compute_loss):
    cdef int hs = model.hs
    cdef int negative = model.negative
    cdef int sample = (model.vocabulary.sample != 0)
    cdef int cbow_mean = model.cbow_mean

    cdef int _compute_loss = (1 if compute_loss == True else 0)
    cdef REAL_t _running_training_loss = model.running_training_loss

    cdef REAL_t *syn0 = <REAL_t *>(np.PyArray_DATA(model.wv.vectors))
    cdef REAL_t *word_locks = <REAL_t *>(np.PyArray_DATA(model.trainables.vectors_lockf))
    cdef REAL_t *work
    cdef REAL_t _alpha = alpha
    cdef int size = model.wv.vector_size

    cdef int codelens[MAX_SENTENCE_LEN]
    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
    cdef np.uint32_t reduced_windows[MAX_SENTENCE_LEN]
    cdef int sentence_idx[MAX_SENTENCE_LEN + 1]
    cdef int window = model.window

    cdef int i, j, k
    cdef int effective_words = 0, effective_sentences = 0
    cdef int sent_idx, idx_start, idx_end

    # For hierarchical softmax
    cdef REAL_t *syn1
    cdef np.uint32_t *points[MAX_SENTENCE_LEN]
    cdef np.uint8_t *codes[MAX_SENTENCE_LEN]

    # For negative sampling
    cdef REAL_t *syn1neg
    cdef np.uint32_t *cum_table
    cdef unsigned long long cum_table_len
    # for sampling (negative and frequent-word downsampling)
    cdef unsigned long long next_random

    if hs:
        syn1 = <REAL_t *>(np.PyArray_DATA(model.trainables.syn1))

    if negative:
        syn1neg = <REAL_t *>(np.PyArray_DATA(model.trainables.syn1neg))
        cum_table = <np.uint32_t *>(np.PyArray_DATA(model.vocabulary.cum_table))
        cum_table_len = len(model.vocabulary.cum_table)
    if negative or sample:
        next_random = (2**24) * model.random.randint(0, 2**24) + model.random.randint(0, 2**24)

    # convert Python structures to primitive types, so we can release the GIL
    work = <REAL_t *>np.PyArray_DATA(_work)
    neu1 = <REAL_t *>np.PyArray_DATA(_neu1)

    # prepare C structures so we can go "full C" and release the Python GIL
    vlookup = model.wv.vocab
    sentence_idx[0] = 0  # indices of the first sentence always start at 0
    for sent in sentences:
        if not sent:
            continue  # ignore empty sentences; leave effective_sentences unchanged
        for token in sent:
            word = vlookup[token] if token in vlookup else None
            if word is None:
                continue  # leaving `effective_words` unchanged = shortening the sentence = expanding the window
            if sample and word.sample_int < random_int32(&next_random):
                continue
            indexes[effective_words] = word.index
            if hs:
                codelens[effective_words] = <int>len(word.code)
                codes[effective_words] = <np.uint8_t *>np.PyArray_DATA(word.code)
                points[effective_words] = <np.uint32_t *>np.PyArray_DATA(word.point)
            effective_words += 1
            if effective_words == MAX_SENTENCE_LEN:
                break  # TODO: log warning, tally overflow?

        # keep track of which words go into which sentence, so we don't train
        # across sentence boundaries.
        # indices of sentence number X are between <sentence_idx[X], sentence_idx[X])
        effective_sentences += 1
        sentence_idx[effective_sentences] = effective_words

        if effective_words == MAX_SENTENCE_LEN:
            break  # TODO: log warning, tally overflow?

    # precompute "reduced window" offsets in a single randint() call
    for i, item in enumerate(model.random.randint(0, window, effective_words)):
        reduced_windows[i] = item

    # release GIL & train on all sentences
    with nogil:
        for sent_idx in range(effective_sentences):
            idx_start = sentence_idx[sent_idx]
            idx_end = sentence_idx[sent_idx + 1]
            for i in range(idx_start, idx_end):
                j = i - window + reduced_windows[i]
                if j < idx_start:
                    j = idx_start
                k = i + window + 1 - reduced_windows[i]
                if k > idx_end:
                    k = idx_end
                if hs:
                    fast_sentence_cbow_hs(points[i], codes[i], codelens, neu1, syn0, syn1, size, indexes, _alpha, work, i, j, k, cbow_mean, word_locks, _compute_loss, &_running_training_loss)
                if negative:
                    next_random = fast_sentence_cbow_neg(negative, cum_table, cum_table_len, codelens, neu1, syn0, syn1neg, size, indexes, _alpha, work, i, j, k, cbow_mean, next_random, word_locks, _compute_loss, &_running_training_loss)

    model.running_training_loss = _running_training_loss
    return effective_words


# Score is only implemented for hierarchical softmax
def score_sentence_sg(model, sentence, _work):

    cdef REAL_t *syn0 = <REAL_t *>(np.PyArray_DATA(model.wv.vectors))
    cdef REAL_t *work
    cdef int size = model.wv.vector_size

    cdef int codelens[MAX_SENTENCE_LEN]
    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
    cdef int sentence_len
    cdef int window = model.window

    cdef int i, j, k
    cdef long result = 0

    cdef REAL_t *syn1
    cdef np.uint32_t *points[MAX_SENTENCE_LEN]
    cdef np.uint8_t *codes[MAX_SENTENCE_LEN]

    syn1 = <REAL_t *>(np.PyArray_DATA(model.trainables.syn1))

    # convert Python structures to primitive types, so we can release the GIL
    work = <REAL_t *>np.PyArray_DATA(_work)

    vlookup = model.wv.vocab
    i = 0
    for token in sentence:
        word = vlookup[token] if token in vlookup else None
        if word is None:
            continue  # should drop the
        indexes[i] = word.index
        codelens[i] = <int>len(word.code)
        codes[i] = <np.uint8_t *>np.PyArray_DATA(word.code)
        points[i] = <np.uint32_t *>np.PyArray_DATA(word.point)
        result += 1
        i += 1
        if i == MAX_SENTENCE_LEN:
            break  # TODO: log warning, tally overflow?
    sentence_len = i

    # release GIL & train on the sentence
    work[0] = 0.0

    with nogil:
        for i in range(sentence_len):
            if codelens[i] == 0:
                continue
            j = i - window
            if j < 0:
                j = 0
            k = i + window + 1
            if k > sentence_len:
                k = sentence_len
            for j in range(j, k):
                if j == i or codelens[j] == 0:
                    continue
                score_pair_sg_hs(points[i], codes[i], codelens[i], syn0, syn1, size, indexes[j], work)

    return work[0]

cdef void score_pair_sg_hs(
    const np.uint32_t *word_point, const np.uint8_t *word_code, const int codelen,
    REAL_t *syn0, REAL_t *syn1, const int size,
    const np.uint32_t word2_index, REAL_t *work) nogil:

    cdef long long b
    cdef long long row1 = word2_index * size, row2, sgn
    cdef REAL_t f

    for b in range(codelen):
        row2 = word_point[b] * size
        f = our_dot(&size, &syn0[row1], &ONE, &syn1[row2], &ONE)
        sgn = (-1)**word_code[b] # ch function: 0-> 1, 1 -> -1
        f *= sgn
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = LOG_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        work[0] += f

def score_sentence_cbow(model, sentence, _work, _neu1):

    cdef int cbow_mean = model.cbow_mean

    cdef REAL_t *syn0 = <REAL_t *>(np.PyArray_DATA(model.wv.vectors))
    cdef REAL_t *work
    cdef REAL_t *neu1
    cdef int size = model.wv.vector_size

    cdef int codelens[MAX_SENTENCE_LEN]
    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
    cdef int sentence_len
    cdef int window = model.window

    cdef int i, j, k
    cdef long result = 0

    # For hierarchical softmax
    cdef REAL_t *syn1
    cdef np.uint32_t *points[MAX_SENTENCE_LEN]
    cdef np.uint8_t *codes[MAX_SENTENCE_LEN]

    syn1 = <REAL_t *>(np.PyArray_DATA(model.trainables.syn1))

    # convert Python structures to primitive types, so we can release the GIL
    work = <REAL_t *>np.PyArray_DATA(_work)
    neu1 = <REAL_t *>np.PyArray_DATA(_neu1)

    vlookup = model.wv.vocab
    i = 0
    for token in sentence:
        word = vlookup[token] if token in vlookup else None
        if word is None:
            continue  # for score, should this be a default negative value?
        indexes[i] = word.index
        codelens[i] = <int>len(word.code)
        codes[i] = <np.uint8_t *>np.PyArray_DATA(word.code)
        points[i] = <np.uint32_t *>np.PyArray_DATA(word.point)
        result += 1
        i += 1
        if i == MAX_SENTENCE_LEN:
            break  # TODO: log warning, tally overflow?
    sentence_len = i

    # release GIL & train on the sentence
    work[0] = 0.0
    with nogil:
        for i in range(sentence_len):
            if codelens[i] == 0:
                continue
            j = i - window
            if j < 0:
                j = 0
            k = i + window + 1
            if k > sentence_len:
                k = sentence_len
            score_pair_cbow_hs(points[i], codes[i], codelens, neu1, syn0, syn1, size, indexes, work, i, j, k, cbow_mean)

    return work[0]

cdef void score_pair_cbow_hs(
    const np.uint32_t *word_point, const np.uint8_t *word_code, int codelens[MAX_SENTENCE_LEN],
    REAL_t *neu1, REAL_t *syn0, REAL_t *syn1, const int size,
    const np.uint32_t indexes[MAX_SENTENCE_LEN], REAL_t *work,
    int i, int j, int k, int cbow_mean) nogil:

    cdef long long a, b
    cdef long long row2
    cdef REAL_t f, g, count, inv_count, sgn
    cdef int m

    memset(neu1, 0, size * cython.sizeof(REAL_t))
    count = <REAL_t>0.0
    for m in range(j, k):
        if m == i or codelens[m] == 0:
            continue
        else:
            count += ONEF
            our_saxpy(&size, &ONEF, &syn0[indexes[m] * size], &ONE, neu1, &ONE)
    if count > (<REAL_t>0.5):
        inv_count = ONEF/count
    if cbow_mean:
        sscal(&size, &inv_count, neu1, &ONE)

    for b in range(codelens[i]):
        row2 = word_point[b] * size
        f = our_dot(&size, neu1, &ONE, &syn1[row2], &ONE)
        sgn = (-1)**word_code[b] # ch function: 0-> 1, 1 -> -1
        f *= sgn
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = LOG_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        work[0] += f


def init():
    """
    Precompute function `sigmoid(x) = 1 / (1 + exp(-x))`, for x values discretized
    into table EXP_TABLE.  Also calculate log(sigmoid(x)) into LOG_TABLE.

    """
    global our_dot
    global our_saxpy

    cdef int i
    cdef float *x = [<float>10.0]
    cdef float *y = [<float>0.01]
    cdef float expected = <float>0.1
    cdef int size = 1
    cdef double d_res
    cdef float *p_res


    
    # build the sigmoid table
    for i in range(EXP_TABLE_SIZE):
        #print(i)
        EXP_TABLE[i] = <REAL_t>exp((i / <REAL_t>EXP_TABLE_SIZE * 2 - 1) * MAX_EXP)
        EXP_TABLE[i] = <REAL_t>(EXP_TABLE[i] / (EXP_TABLE[i] + 1))
        LOG_TABLE[i] = <REAL_t>log( EXP_TABLE[i] )
    #print('done')

    
    # check whether sdot returns double or float
    d_res = dsdot(&size, x, &ONE, y, &ONE)
    #print('done1.5')
    p_res = <float *>&d_res
    if (abs(d_res - expected) < 0.0001):
        #print('done2a')
        our_dot = our_dot_double
        our_saxpy = saxpy
        #print('done2a')
        return 0  # double
    elif (abs(p_res[0] - expected) < 0.0001):
        #print('done2b')
        our_dot = our_dot_float
        our_saxpy = saxpy
        #print('done2b')
        return 1  # float
    else:
        # neither => use cython loops, no BLAS
        # actually, the BLAS is so messed up we'll probably have segfaulted above and never even reach here
        #print('done2c')
        our_dot = our_dot_noblas
        our_saxpy = our_saxpy_noblas
        #print('done2c')
        return 2
        


FAST_VERSION = init()  # initialize the module
MAX_WORDS_IN_BATCH = MAX_SENTENCE_LEN
