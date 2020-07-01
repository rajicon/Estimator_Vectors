#This is also getting very complicated :(
#Need to do a deeper, easier method?
#(A) treat RHS and one with neg samples, so don't pass in a set  (PROBABLY THIS ROUTE)
#(B) calc neg samples before calling on a pair, pass in set of neg samples and then treat LHS and RHS similarly (don't do this)




#Raj:  Trains input of 2 estimator sets, LHS and RHS, too be called by outside function
#Raj:  Note that if a word vector is passed here, it must be passed in as a one element list of indices (similar to context clues and subwords)
#Raj:  Added in by me, may have to add to header file
#Raj:  LHS_set is set of estimators for word2 (row1), RHS_set is set of estimators for word1 (row2)
#Note: subwords make this extremely complicated!  probably need to pass in index_breakdown along with LHS_sub = True and RHS_sub = True
#Be VERY careful with above then, maybe check if LHS_set or RHS_set is a 1 element list, if it is, break it down to subwords as new LHS_set/RHS_set
#Raj:  for now, index_subword_breakdown is being passed , but maybe instead we should have a RHS estimator translater passed?  Should LHS and RHS?
#neuLHS is sum of LHS vectors, work is total gradient to edit LHS
cdef unsigned long long inner_train_pair_sg_neg(
    const int negative, np.uint32_t *cum_table, unsigned long long cum_table_len,
    REAL_t *LHS_syn, REAL_t *RHS_syn, const int size, const np.uint32_t word_index,
    const np.uint32_t word2_index, const REAL_t alpha, REAL_t *work, REAL_t *neuLHS, REAL_t *neuRHS,
    unsigned long long next_random, REAL_t *word_locks,
    const int _compute_loss, REAL_t *_running_training_loss_param, np.uint32_t* LHS_set, int LHS_list_len, int LHS_padding_num,  np.uint32_t* RHS_set, int RHS_list_len, int RHS_padding_num, index_subword_breakdown) nogil:


    cdef long long a
    cdef long long row1 = word2_index * size, row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, label, f_dot, log_e_f_dot
    cdef np.uint32_t target_index
    cdef int d, m, qqqq

    memset(work, 0, size * cython.sizeof(REAL_t))  #I think work is for storing gradient updates for LHS_syn (with all negs/target word)
                                                   #in our case, work will store all gradients for each context clue I guess?
                                                   #I think memset is analoguous to np.zeros, although I'm not sure

    #Raj: it would appear that row 1 is matched with word2 and row 2 is matched with word1/neg samples for some reason
    #Raj:  EXP_TABLE is precomputed sigmoid, LOG_TABLE is log sigmoid


    #Raj calculate sum of context clues instead of LHS_syn[row1], use neuLHS
    #based on how cbow does it
    memset(neuLHS, 0, size * cython.sizeof(REAL_t))
    #count = <REAL_t>0.0
    for m in range(0, LHS_list_len):  #MAX_CONTEXT_SIZE
        #count += ONEF
        if LHS_set[m] != LHS_padding_num:  #-1
            our_saxpy(&size, &ONEF, &LHS_syn[LHS_set[m] * size], &ONE, neuLHS, &ONE)  #our alpha here is 1, so just adds each context clue
    
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
			
        #calculate RHS sum
        memset(neuRHS, 0, size * cython.sizeof(REAL_t))  #reset subword sum
        for q in range(0, RHS_list_len):
            sub_ind = index_subword_breakdown[(target_index * RHS_list_len)+q]
            if sub_ind != sub_padding_num:
                our_saxpy(&size, &ONEF, &syn2subword[sub_ind * size], &ONE, neuRHS, &ONE)  #our alpha here is 1, so just adds each context clue
            else:
                break

        row2 = target_index * size
        #f_dot = our_dot(&size, &LHS_syn[row1], &ONE, &RHS_syn[row2], &ONE)
        f_dot = our_dot(&size, neuLHS, &ONE, &RHS_syn[row2], &ONE)
        if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]  #this is basically applying sigmoid function but using EXP_TABLE for fast calculations
        g = (label - f) * alpha   #so this is label - sigmoid(u*v) * lr

        if _compute_loss == 1:
            f_dot = (f_dot if d == 0  else -f_dot)
            if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                continue
            log_e_f_dot = LOG_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
            _running_training_loss_param[0] = _running_training_loss_param[0] - log_e_f_dot



        our_saxpy(&size, &g, &RHS_syn[row2], &ONE, work, &ONE)  #I think this stores gradient updates for LHS_syn?

        our_saxpy(&size, &g, neuLHS, &ONE, &RHS_syn[row2], &ONE) #updates neg sample/word (word emb in CCV case)
        
    
    for qqqq in range(0, LHS_list_len):
        if LHS_set[qqqq] != -1:
            our_saxpy(&size, &word_locks[word2_index], work, &ONE, &LHS_syn[LHS_set[qqqq] * size], &ONE)  #updates LHS_syn (context clues?)
    
    
    #with gil:
    #    print('got here')
    return next_random