import numpy as np
from lab2_tools import *


def concatTwoHMMs(hmm1, hmm2):
    """ Concatenates 2 HMM models

    Args:
       hmm1, hmm2: two dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be different for each)

    Output
       dictionary with the same keys as the input but concatenated models:
          startprob: K+1 array with priori probability of state
          transmat: (K+1)x(K+1) transition matrix
             means: KxD array of mean vectors
            covars: KxD array of variances

    K is the sum of the number of emitting states from the input models

    Example:
       twoHMMs = concatHMMs(phoneHMMs['sil'], phoneHMMs['ow'])

    See also: the concatenating_hmms.pdf document in the lab package
    """
    combinedhmm = dict()

    # Number of states in hmm1 (excluding the non-emitting state)
    M1 = len(hmm1['startprob']) - 1
    M2 = len(hmm2['startprob']) - 1

    # Concatenate start probabilities
    combinedhmm['startprob'] = np.zeros(M1 + M2 + 1)
    # Exclude the non-emitting state's probability
    combinedhmm['startprob'][:M1] = hmm1['startprob'][:-1]
    # Transition from hmm1's non-emitting state
    combinedhmm['startprob'][M1:] = hmm1['startprob'][-1] * hmm2['startprob']

    # Create the transition matrix
    combinedhmm['transmat'] = np.zeros((M1 + M2 + 1, M1 + M2 + 1))
    # Internal transitions in hmm1
    combinedhmm['transmat'][:M1, :M1] = hmm1['transmat'][:-1, :-1]
    # Internal transitions in hmm2
    combinedhmm['transmat'][M1:, M1:] = hmm2['transmat']
    combinedhmm['transmat'][:M1, M1:] = np.outer(
        hmm1['transmat'][:-1, -1], hmm2['startprob'])  # From hmm1 to hmm2

    # Concatenate means and covariances
    combinedhmm['means'] = np.vstack((hmm1['means'], hmm2['means']))
    combinedhmm['covars'] = np.vstack((hmm1['covars'], hmm2['covars']))

    return combinedhmm


# this is already implemented, but based on concat2HMMs() above
def concatHMMs(hmmmodels, namelist):
    """ Concatenates HMM models in a left to right manner

    Args:
       hmmmodels: dictionary of models indexed by model name. 
       hmmmodels[name] is a dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances
       namelist: list of model names that we want to concatenate

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be
      different in each model)

    Output
       combinedhmm: dictionary with the same keys as the input but
                    combined models:
         startprob: K+1 array with priori probability of state
          transmat: (K+1)x(K+1) transition matrix
             means: KxD array of mean vectors
            covars: KxD array of variances

    K is the sum of the number of emitting states from the input models

    Example:
       wordHMMs['o'] = concatHMMs(phoneHMMs, ['sil', 'ow', 'sil'])
    """
    concat = hmmmodels[namelist[0]]
    for idx in range(1, len(namelist)):
        concat = concatTwoHMMs(concat, hmmmodels[namelist[idx]])
    return concat


def gmmloglik(log_emlik, weights):
    """Log Likelihood for a GMM model based on Multivariate Normal Distribution.

    Args:
        log_emlik: array like, shape (N, K).
            contains the log likelihoods for each of N observations and
            each of K distributions
        weights:   weight vector for the K components in the mixture

    Output:
        gmmloglik: scalar, log likelihood of data given the GMM model.
    """


def forward(log_emlik, log_startprob, log_transmat):
    """Forward (alpha) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: log transition probability from state i to j

    Output:
        forward_prob: NxM array of forward log probabilities for each of the M states in the model
    """
    forward_prob = np.zeros(log_emlik.shape)

    forward_prob[0] = log_startprob.T + log_emlik[0]
    for n in range(1, forward_prob.shape[0]):
        for j in range(forward_prob.shape[1]):
            forward_prob[n, j] = logsumexp(
                forward_prob[n-1, :] + log_transmat[:, j]) + log_emlik[n, j]

    return forward_prob


def backward(log_emlik, log_startprob, log_transmat):
    """Backward (beta) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j

    Output:
        backward_prob: NxM array of backward log probabilities for each of the M states in the model
    """
    backward_prob = np.zeros(log_emlik.shape)

    for n in reversed(range(backward_prob.shape[0] - 1)):
        for i in range(backward_prob.shape[1]):
            backward_prob[n, i] = logsumexp(
                log_transmat[i, :] + log_emlik[n+1, :] + backward_prob[n+1, :])

    return backward_prob


def viterbi(log_emlik, log_startprob, log_transmat, forceFinalState=True):
    """Viterbi path.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j
        forceFinalState: if True, start backtracking from the final state in
                  the model, instead of the best state at the last time step

    Output:
        viterbi_loglik: log likelihood of the best path
        viterbi_path: best path
    """
    N = log_emlik.shape[0]
    M = log_emlik.shape[1]

    viterbi_loglik = 0
    viterbi_path = np.empty((N), dtype=int)
    V = np.zeros((N, M))
    B = np.zeros((N, M))

    for j in range(M):
        V[0, j] = log_startprob[j] + log_emlik[0, j]

    for n in range(1, N):
        for j in range(M):
            V[n, j] = np.max(V[n-1, :] + log_transmat[:, j]) + log_emlik[n, j]
            B[n, j] = np.argmax(V[n-1, :] + log_transmat[:, j])

    viterbi_path[-1] = np.argmax(V[-1, :])
    viterbi_loglik = V[N-1, viterbi_path[-1]]

    for n in range(0, N-1):
        viterbi_path[n] += np.max(V[n-1, :])

    for n in reversed(range(N-1)):
        for j in range(M):
            viterbi_path[n] = B[n+1, viterbi_path[n+1]]

    return (viterbi_loglik, viterbi_path)


def statePosteriors(log_alpha, log_beta):
    """State posterior (gamma) probabilities in log domain.

    Args:
        log_alpha: NxM array of log forward (alpha) probabilities
        log_beta: NxM array of log backward (beta) probabilities
    where N is the number of frames, and M the number of states

    Output:
        log_gamma: NxM array of gamma probabilities for each of the M states in the model
    """
    log_gamma = np.zeros((log_alpha.shape[0], log_alpha.shape[1]))
    
    for i in range(log_alpha.shape[0]):
        log_gamma[i] = log_alpha[i, :] + log_beta[i, :] - logsumexp(log_alpha[-1])

    return log_gamma
    
    

def updateMeanAndVar(X, log_gamma, varianceFloor=5.0):
    """ Update Gaussian parameters with diagonal covariance

    Args:
         X: NxD array of feature vectors
         log_gamma: NxM state posterior probabilities in log domain
         varianceFloor: minimum allowed variance scalar
    were N is the lenght of the observation sequence, D is the
    dimensionality of the feature vectors and M is the number of
    states in the model

    Outputs:
         means: MxD mean vectors for each state
         covars: MxD covariance (variance) vectors for each state
    """
    N = X.shape[0]
    D = X.shape[1]
    M = log_gamma.shape[1]
    
    gamma = np.exp(log_gamma)
    means = np.zeros((M, D))
    covars = np.zeros((M, D))

    
    
    for i in range(M):
        gm = np.sum(gamma[:, i])
        means[i] = np.dot(gamma[:, i].T, X)/gm
        covars[i] = np.dot(gamma[:, i].T, (X - means[i])**2)/gm

    covars = np.clip(covars, varianceFloor, None)

    return means, covars
        





    

    
