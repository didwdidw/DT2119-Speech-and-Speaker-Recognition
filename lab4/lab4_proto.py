
# DT2119, Lab 4 End-to-end Speech Recognition
import numpy as np
from torchaudio.transforms import MelSpectrogram
from torchaudio.transforms import FrequencyMasking, TimeMasking
from torch import nn
from torchaudio.functional import edit_distance
import torch


# Variables to be defined --------------------------------------
''' 
train-time audio transform object, that transforms waveform -> spectrogram, with augmentation
'''
train_audio_transform = nn.Sequential(
    MelSpectrogram(n_mels=80), FrequencyMasking(15), TimeMasking(35)
)
'''
test-time audio transform object, that transforms waveform -> spectrogram, without augmentation 
'''
test_audio_transform = nn.Sequential(MelSpectrogram(n_mels=80))

# Functions to be implemented ----------------------------------


def intToStr(labels):
    '''
        convert list of integers to string
    Args: 
        labels: list of ints
    Returns:
        string with space-separated characters
    '''
    # String containing all valid characters
    vocab = "' abcdefghijklmnopqrstuvwxyz"

    # Convert each number in the list to the corresponding character and concatenate them
    result = ""
    for label in labels:
        int_label = int(label)
        result += vocab[int_label]

    return result


def strToInt(text):
    '''
        convert string to list of integers
    Args:
        text: string
    Returns:
        list of ints
    '''
    text = text.lower()  # Convert text to lowercase
    vocab = "' abcdefghijklmnopqrstuvwxyz"

    # Convert each character in the string to its index in the vocab
    result = []
    for char in text:
        if char in vocab:
            result.append(vocab.index(char))
    return result


def dataProcessing(data, transform):
    '''
    process a batch of speech data
    arguments:
        data: list of tuples, representing one batch. Each tuple is of the form
            (waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id)
        transform: audio transform to apply to the waveform
    returns:
        a tuple of (spectrograms, labels, input_lengths, label_lengths) 
        -   spectrograms - tensor of shape B x C x T x M 
            where B=batch_size, C=channel, T=time_frames, M=mel_band.
            spectrograms are padded the longest length in the batch.
        -   labels - tensor of shape B x L where L is label_length. 
            labels are padded to the longest length in the batch. 
        -   input_lengths - list of half spectrogram lengths before padding
        -   label_lengths - list of label lengths before padding
    '''
    # Process spectrograms
    spectrograms_list = []
    input_lengths = []
    for batch in data:
        # Apply the transform to the audio part of the batch (index 0), squeeze and transpose
        spectrogram = transform(batch[0]).squeeze(0).transpose(0, 1)
        spectrograms_list.append(spectrogram)
        input_lengths.append(spectrogram.shape[0] // 2)

    # Pad and format the list of spectrograms
    spectrograms = nn.utils.rnn.pad_sequence(
        spectrograms_list, batch_first=True)
    spectrograms = spectrograms.unsqueeze(1).transpose(2, 3)

    # Process labels
    labels_list = []
    label_lengths = []
    for batch in data:
        # Convert text labels (index 2) to integer indices
        label = torch.Tensor(strToInt(batch[2]))
        labels_list.append(label)
        label_lengths.append(len(label))

    # Pad the list of labels
    labels = nn.utils.rnn.pad_sequence(labels_list, batch_first=True)

    return spectrograms, labels, input_lengths, label_lengths


# ------------------------------------------------------------------------------------ #


# To help you verify that your data processing function does the right thing,
# A small example batch of size 5 including input and output of the function is provided
example = torch.load("lab4_example.pt")
Y = dataProcessing(example["data"], test_audio_transform)

# Checking the output against the example data
errors = {
    "spectrograms_error": np.nanmax(np.abs(Y[0] - example["spectrograms"])),
    "labels_error": np.nanmax(np.abs(torch.Tensor(Y[1]) - example["labels"])),
    "input_lengths_error": np.nanmax(np.abs(np.array(Y[2]) - np.array(example["input_lengths"]))),
    "label_lengths_error": np.nanmax(np.abs(np.array(Y[3]) - np.array(example["label_lengths"])))
}

# Print the comparison errors
for error_type, value in errors.items():
    print(f"{error_type}: {value}")


# ------------------------------------------------------------------------------------ #


def greedyDecoder(output, blank_label=28):
    '''
    decode a batch of utterances 
    arguments:
        output: network output tensor, shape B x T x C where B=batch_size, T=time_steps, C=characters
        blank_label: id of the blank label token
    returns:
        list of decoded strings
    '''
    decoded_strings = []
    for batch in output:
        # Extract the most likely labels for each time step.
        best_labels = torch.argmax(batch, dim=1)
        # Filter out repeated labels and blanks directly by comparing to shifted version and filtering blanks.
        filtered_labels = best_labels[1:][(
            best_labels[1:] != best_labels[:-1]) & (best_labels[1:] != blank_label)]
        # Convert numerical indices to string using intToStr (ensure intToStr can handle a tensor directly).
        decoded_strings.append(intToStr(filtered_labels.tolist()))

    return decoded_strings


def levenshteinDistance(ref, hyp):
    '''
    calculate levenshtein distance (edit distance) between two sequences
    arguments:
        ref: reference sequence
        hyp: sequence to compare against the reference
    output:
        edit distance (int)
    '''
    len_ref, len_hyp = len(ref), len(hyp)

    # Initialize a matrix with dimensions (len_ref + 1) x (len_hyp + 1)
    matrix = [[0] * (len_hyp + 1) for _ in range(len_ref + 1)]

    # Fill the first column of the matrix
    for i in range(len_ref + 1):
        matrix[i][0] = i

    # Fill the first row of the matrix
    for j in range(len_hyp + 1):
        matrix[0][j] = j

    # Compute the cost and fill the matrix
    for i in range(1, len_ref + 1):
        for j in range(1, len_hyp + 1):
            # If characters match, cost is 0, otherwise cost is 1
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            # Take the minimum of three possible operations: deletion, insertion, substitution
            matrix[i][j] = min(matrix[i - 1][j] + 1,  # Deletion
                               matrix[i][j - 1] + 1,  # Insertion
                               matrix[i - 1][j - 1] + cost)  # Substitution

    # Return the computed Levenshtein distance (bottom-right cell of the matrix)
    return matrix[len_ref][len_hyp]


def languageDecoder(output, decoder):
    """
    Decode the network output using a provided CTC decoder.

    Arguments:
        output: network output tensor of shape (batch, time, character)
        decoder: a pre-built CTC decoder

    Returns:
        A list of decoded strings of the batch.
    """
    # Initialize the list for storing decoded sequences
    decoded_strings = []

    # Iterate over each example in the batch
    for batch_idx in range(output.shape[0]):
        # Decode the example using the provided decoder
        text = decoder.decode(output[batch_idx].cpu().detach().numpy())
        decoded_strings.append(text)

    return decoded_strings
