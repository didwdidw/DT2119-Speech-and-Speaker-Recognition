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
    from sklearn.mixture import GaussianMixture
    import numpy as np

    # Concatenate all HMMs for the phones in the transcription
    state_list = []
    start_prob = []
    trans_mat = []
    means = []
    covars = []

    for phone in phoneTrans:
        model = phoneHMMs[phone]
        n_states = len(model['startprob'])
        start_prob.extend(model['startprob'])
        means.extend(model['means'])
        covars.extend(model['covars'])

        # Expand transition matrix
        expanded_trans_mat = np.zeros(
            (len(trans_mat) + n_states, len(trans_mat) + n_states))
        expanded_trans_mat[:len(trans_mat), :len(trans_mat)] = trans_mat
        expanded_trans_mat[len(trans_mat):, len(
            trans_mat):] = model['transmat']
        trans_mat = expanded_trans_mat

        # Define state list
        state_list.extend([f"{phone}_{i}" for i in range(n_states)])

    # Convert lists to numpy arrays for log probabilities
    log_startprob = np.log(np.array(start_prob))
    log_transmat = np.log(np.array(trans_mat))

    # Compute log likelihood for each state using Gaussian mixture model
    gmm = GaussianMixture(n_components=len(means), covariance_type='diag',
                          means_init=means, precisions_init=np.linalg.inv(covars))
    gmm.means_ = np.array(means)
    gmm.covariances_ = np.array(covars)
    gmm.weights_ = np.ones(len(means)) / len(means)
    log_emlik = gmm.score_samples(lmfcc)

    # Align using Viterbi
    import sys
    sys.path.append('/Users/tim/Desktop/Speech/lab2')
    from lab2_proto import viterbi
    viterbi_path = viterbi(log_emlik, log_startprob, log_transmat)
    aligned_phones = [state_list[state] for state in viterbi_path]

    return aligned_phones
