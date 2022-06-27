import numpy as np
from os import listdir, mkdir, system
from os.path import isfile, isdir, join, exists
import mne
import os


dir_in = './sleep-edfx/sleep-cassette/'
dir_out = './sleep-edfx/cassette_npz/'
fs = 100

# Here they remove stage 4, I'll think we'll stick with that for now
annot2label = {
    "Sleep stage ?": -1,
    "Movement time": -1,
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4
}

data_dict = {}

psg_records = [f for f in listdir(dir_in) if isfile(join(dir_in, f)) if (f.find('-PSG.edf') != -1)]
hyp_records = [f for f in listdir(dir_in) if isfile(join(dir_in, f)) if (f.find('-Hypnogram.edf') != -1)]
psg_records.sort()
hyp_records.sort()

subset_size = len(psg_records)

for i in range(subset_size):
    psg_fn = psg_records[i]
    hyp_fn = hyp_records[i]
    record_id = psg_fn[:6]
    print('loading %d PSG %s' % (i, record_id))

    if i < int(subset_size * 0.6):
        set_name = 'train'
    elif i < int(subset_size * 0.8):
        set_name = 'val'
    else:
        set_name = 'test'

    psg = mne.io.read_raw_edf(dir_in + psg_fn, verbose=False).get_data()
    ch_num = psg.shape[0]
    epoch_samp = 30 * fs
    psg = np.transpose(psg).reshape(-1, epoch_samp, ch_num)
    psg = np.transpose(psg, (0, 2, 1))
    annos = mne.read_annotations(dir_in + hyp_fn)
    anno_len = int((annos[-1]['onset'] + annos[-1]['duration']) // 30)
    anno_arr = np.zeros(anno_len, dtype=int)

    for a in annos:
        onset = int(a['onset'] // 30)
        dur = int(a['duration'] // 30)
        anno_arr[onset: onset + dur] = annot2label[a['description']]
    x = psg.astype(np.float32)
    y = anno_arr.astype(np.int32)

    w_edge_mins = 30
    nw_idx = np.where(y != annot2label["Sleep stage W"])[0]
    start_idx = nw_idx[0] - (w_edge_mins * 2)
    end_idx = nw_idx[-1] + (w_edge_mins * 2)
    if start_idx < 0:
        start_idx = 0
    if end_idx >= len(y):
        end_idx = len(y) - 1

    x = x[start_idx: end_idx + 1]
    y = y[start_idx: end_idx + 1]

    remove_idx = np.where(y == -1)[0]
    if len(remove_idx) > 0:
        select_idx = np.setdiff1d(np.arange(len(x)), remove_idx)
        x = x[select_idx]
        y = y[select_idx]
    print(x.shape, y.shape)

    filename = record_id + ".npz"
    save_path = os.path.join(dir_out, set_name, filename)

    save_dict = {
        "x": x,
        "y": y,
        "fs": fs,
        "n_epochs": len(x),
    }
    print("saving to ", save_path)
    np.savez(save_path, **save_dict)
