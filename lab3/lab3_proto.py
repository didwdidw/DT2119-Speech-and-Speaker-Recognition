import numpy as np
from lab3_tools import *


def words2phones(wordList, pronDict, addSilence=True, addShortPause=True):
    """ word2phones: converts word level to phone level transcription adding silence

    Args:
       wordList: list of word symbols
       pronDict: pronunciation dictionary. The keys correspond to words in wordList
       addSilence: if True, add initial and final silence
       addShortPause: if True, add short pause model "sp" at end of each word
    Output:
       list of phone symbols
    """
    phone_list = []
    if addSilence:
        phone_list.append('sil')  # Add initial silence

    for word in wordList:
        if word in pronDict:
            # Add phones from pronunciation dictionary
            phone_list.extend(pronDict[word])
            if addShortPause:
                phone_list.append('sp')  # Add short pause after each word

    if addSilence:
        phone_list.append('sil')  # Add final silence

    return phone_list


def forcedAlignment(lmfcc, phoneHMMs, phoneTrans):
    """ forcedAlignmen: aligns a phonetic transcription at the state level

    Args:
       lmfcc: NxD array of MFCC feature vectors (N vectors of dimension D)
              computed the same way as for the training of phoneHMMs
       phoneHMMs: set of phonetic Gaussian HMM models
       phoneTrans: list of phonetic symbols to be aligned including initial and
                   final silence

    Returns:
       list of strings in the form phoneme_index specifying, for each time step
       the state from phoneHMMs corresponding to the viterbi path.
    """
    import sys
    sys.path.append('/Users/tim/Desktop/Speech/lab2')
    from lab2_proto import concatHMMs, viterbi
    from lab2_tools import log_multivariate_normal_density_diag

    phones = sorted(phoneHMMs.keys())
    nstates = {phone: phoneHMMs[phone]['means'].shape[0] for phone in phones}
    stateList = [ph + '_' + str(id)
                 for ph in phones for id in range(nstates[ph])]
    stateTrans = [phone + '_' + str(stateid)
                  for phone in phoneTrans for stateid in range(nstates[phone])]

    # Create a combined model for this specific utterance:
    utteranceHMM = concatHMMs(phoneHMMs, phoneTrans)

    # NxM array of emission(observation) log likelihoods, N frames, M states
    obsloglik = log_multivariate_normal_density_diag(
        lmfcc, utteranceHMM['means'], utteranceHMM['covars'])
    log_startprob = np.log(utteranceHMM['startprob'][:-1])
    log_transmat = np.log(utteranceHMM['transmat'][:-1, :-1])
    vloglik, vpath = viterbi(obsloglik, log_startprob, log_transmat)

    # Converting vpath from state names to state indices to save memory
    stateIndicesOfStateList = [stateList.index(stateTrans[i]) for i in vpath]

    return stateIndicesOfStateList
