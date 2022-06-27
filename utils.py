import numpy as np
import torch
import torch.nn as nn

def process(x, y):

    # Each element in a batch is a sequence of 25 time-windows
    # This makes sense, as we don't want to feed in random sequences of time-windows
    batch_size = 10
    seq_length = 25

    # eeg = x[:, 0:1, :] --> They use EEG only, but we want all channels (which includes other signals)
    eeg = x
    input_sample_shape = eeg.shape[1:]
    target_sample_shape = y.shape[1:]
    batch_length = len(eeg) // batch_size
    num_seq = batch_length // seq_length

    batch_x = np.zeros((num_seq, batch_size, seq_length) + input_sample_shape, dtype=np.float32)
    batch_y = np.zeros((num_seq, batch_size, seq_length) + target_sample_shape, dtype=np.int)

    for b in range(batch_size):
        start_idx = b * batch_length
        end_idx = (b + 1) * batch_length
        seq_x = eeg[start_idx: end_idx]
        seq_y = y[start_idx: end_idx]

        for s in range(num_seq):
            start_idx = s * seq_length
            end_idx = (s + 1) * seq_length
            batch_x[s, b] = seq_x[start_idx: end_idx]
            batch_y[s, b] = seq_y[start_idx: end_idx]
            
    batch_x = batch_x.reshape((num_seq * batch_size, seq_length) + input_sample_shape)
    batch_y = batch_y.reshape((num_seq * batch_size, seq_length) + target_sample_shape)
    
    return batch_x, batch_y


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
