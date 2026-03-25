import os
import pdb
import numpy as np
import lmdb
import joblib

import torch
from torch.utils.data import Dataset, DataLoader
#from visual_src.visual_tools import VisualTools

from utils.lmdb_tools.datum_pb2 import SimpleDatum

class LmdbDataset(Dataset):
    def __init__(self, lmdb_dir, split, normalized_features_wts_file=None, ignore=None, segment_len=None, data_process_fn=None, is_train=False) -> None:
        super().__init__()
        self.split = split
        self.ignore = ignore
        self.segment_len = segment_len
        self.data_process_fn = data_process_fn
        #self.visial_tools = VisualTools()
        self.lmdb_dir = lmdb_dir
        self.keys = []
        with open(os.path.join(lmdb_dir, 'keys.txt'), 'r') as f:
            lines = f.readlines()
            for k in lines:
                if self.ignore is not None and self.ignore in k:
                    continue
                if int(k[4]) in self.split: # check which split the file belongs to
                    self.keys.append(k.strip())
        # self.env = lmdb.open(lmdb_dir, readonly=True, readahead=True, lock=False)
        self.env = None
        self.spec_scaler = None
        if normalized_features_wts_file is not None:
            self.spec_scaler = joblib.load(normalized_features_wts_file)
        self.is_train = is_train
    
    def _get_env(self):
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_dir,
                readonly=True,
                lock=False,
                readahead=True,
                meminit=False
            )
        return self.env

    def __getstate__(self):
        d = self.__dict__.copy()
        d["env"] = None  # pickle할 때 env 제거
        return d
       
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        # txn = self.env.begin()
        # with txn.cursor() as cursor:
        #     k = self.keys[index].strip().encode()
        #     cursor.set_key(k)
        env = self._get_env()
        k = self.keys[index].encode()

        with env.begin(write=False) as txn:
            buf = txn.get(k)
        datum=SimpleDatum()
        datum.ParseFromString(buf) #cursor.value()
        data = np.frombuffer(datum.data, dtype=np.float32).reshape(-1, datum.data_dim)
        if self.spec_scaler is not None:
            data = self.spec_scaler.transform(data)
        #pdb.set_trace()
        label = np.frombuffer(datum.label, dtype=np.float32).reshape(-1, datum.label_dim)
        wav_name = datum.wave_name.decode()
        if self.segment_len is not None:
            if self.is_train:
                if label.shape[0] < self.segment_len:
                    data = np.pad(data, pad_width=((0,self.segment_len*5-data.shape[0]), (0,0)))
                    label = np.pad(label, pad_width=((0,self.segment_len-label.shape[0]), (0,0)))
                elif label.shape[0] > self.segment_len:
                    max_label_start = label.shape[0] - self.segment_len
                    start_label_idx = np.random.randint(0, max_label_start + 1)
                    
                    start_data_idx = start_label_idx * 5
                    
                    label = label[start_label_idx : start_label_idx + self.segment_len]
                    data = data[start_data_idx : start_data_idx + self.segment_len * 5]
            else:
                if label.shape[0] < self.segment_len:
                    data = np.pad(data, pad_width=((0,self.segment_len*5-data.shape[0]), (0,0)))
                    label = np.pad(label, pad_width=((0,self.segment_len-label.shape[0]), (0,0)))
        if self.data_process_fn is not None:
            data, label = self.data_process_fn(data, label)
        #print('feat {}'.format(data.shape))
        #print('label {}'.format(label.shape))
        #print('wavname {}'.format(wav_name))
        return {'data': data, 'label':label, 'wav_name':wav_name}


    # def collater(self, samples):
    #     feats = [s['data'] for s in samples]
    #     labels = [s['label'] for s in samples]
    #     wav_names = [s['wav_name'] for s in samples]
        
    #     collated_feats = np.stack(feats, axis=0)
    #     collated_labels = np.stack(labels, axis=0)

    #     out = {}
    #     out['input'] = torch.from_numpy(collated_feats)
    #     out['target'] = torch.from_numpy(collated_labels)
    #     out['wav_names'] = wav_names

    #     return out
    
    def collater(self, samples):
        feats = [s['data'] for s in samples]
        labels = [s['label'] for s in samples]
        wav_names = [s['wav_name'] for s in samples]

        ref_feat_shape = feats[0].shape
        ref_label_shape = labels[0].shape

        for i, (w, f, l) in enumerate(zip(wav_names, feats, labels)):
            assert isinstance(f, np.ndarray), f"feat not ndarray: idx={i}, wav={w}, type={type(f)}"
            assert isinstance(l, np.ndarray), f"label not ndarray: idx={i}, wav={w}, type={type(l)}"
            assert f.shape == ref_feat_shape, f"feat shape mismatch: idx={i}, wav={w}, got={f.shape}, expected={ref_feat_shape}"
            assert l.shape == ref_label_shape, f"label shape mismatch: idx={i}, wav={w}, got={l.shape}, expected={ref_label_shape}"

        collated_feats = np.stack(feats, axis=0)
        collated_labels = np.stack(labels, axis=0)

        return {
            'input': torch.from_numpy(collated_feats).float(),
            'target': torch.from_numpy(collated_labels).float(),
            'wav_names': wav_names
        }
    
class LmdbDataset_seddoa_sedsde(Dataset):
    def __init__(self, lmdb_dir, split, seddoa_normalized_features_wts_file=None, sedsde_normalized_features_wts_file=None, ignore=None, segment_len=None, data_process_fn=None) -> None:
        super().__init__()
        self.split = split
        self.ignore = ignore
        self.segment_len = segment_len
        self.data_process_fn = data_process_fn
        #self.visial_tools = VisualTools()
        self.keys = []
        with open(os.path.join(lmdb_dir, 'keys.txt'), 'r') as f:
            lines = f.readlines()
            for k in lines:
                if self.ignore is not None and self.ignore in k:
                    continue
                if int(k[4]) in self.split: # check which split the file belongs to
                    self.keys.append(k.strip())
        # self.env = lmdb.open(lmdb_dir, readonly=True, readahead=True, lock=False)
        self.env = None
        self.lmdb_dir = lmdb_dir
        self.seddoa_spec_scaler = None
        self.sedsde_spec_scaler = None
        if seddoa_normalized_features_wts_file is not None:
            self.seddoa_spec_scaler = joblib.load(seddoa_normalized_features_wts_file)
        if sedsde_normalized_features_wts_file is not None:
            self.sedsde_spec_scaler = joblib.load(sedsde_normalized_features_wts_file)
    
    def _get_env(self):
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_dir,
                readonly=True,
                lock=False,
                readahead=True,
                meminit=False
            )
        return self.env

    def __getstate__(self):
        d = self.__dict__.copy()
        d["env"] = None  # pickle할 때 env 제거
        return d
    
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        # txn = self.env.begin()
        # with txn.cursor() as cursor:
        #     k = self.keys[index].strip().encode()
        #     cursor.set_key(k)
        env = self._get_env()
        k = self.keys[index].encode()

        with env.begin(write=False) as txn:
            buf = txn.get(k)
        datum=SimpleDatum()
        datum.ParseFromString(buf) #cursor.value()
        data = np.frombuffer(datum.data, dtype=np.float32).reshape(-1, datum.data_dim)
        if self.seddoa_spec_scaler is not None:
            seddoa_data = self.seddoa_spec_scaler.transform(data)
        if self.sedsde_spec_scaler is not None:
            sedsde_data = self.sedsde_spec_scaler.transform(data)
        #pdb.set_trace()
        label = np.frombuffer(datum.label, dtype=np.float32).reshape(-1, datum.label_dim)

        wav_name = datum.wave_name.decode()
        if self.segment_len is not None and label.shape[0] < self.segment_len:
            seddoa_data = np.pad(seddoa_data, pad_width=((0,self.segment_len*5-data.shape[0]), (0,0)))
            sedsde_data = np.pad(sedsde_data, pad_width=((0,self.segment_len*5-data.shape[0]), (0,0)))
            label = np.pad(label, pad_width=((0,self.segment_len-label.shape[0]), (0,0)))
        if self.data_process_fn is not None:
            seddoa_data, label1 = self.data_process_fn(seddoa_data, label)
            sedsde_data, label2 = self.data_process_fn(sedsde_data, label)

        #print('feat {}'.format(data.shape))
        #print('label {}'.format(label.shape))
        #print('wavname {}'.format(wav_name))
        return {'seddoa_data': seddoa_data, 'sedsde_data':sedsde_data, 'label':label1, 'wav_name':wav_name}


    def collater(self, samples):
        seddoa_feats = [s['seddoa_data'] for s in samples]
        sedsde_feats = [s['sedsde_data'] for s in samples]
        labels = [s['label'] for s in samples]
        wav_names = [s['wav_name'] for s in samples]

        seddoa_collated_feats = np.stack(seddoa_feats, axis=0)
        sedsde_collated_feats = np.stack(sedsde_feats, axis=0)
        collated_labels = np.stack(labels, axis=0)

        out = {}
        out['input_seddoa'] = torch.from_numpy(seddoa_collated_feats)
        out['input_sedsde'] = torch.from_numpy(sedsde_collated_feats)
        out['target'] = torch.from_numpy(collated_labels)
        out['wav_names'] = wav_names

        return out

class LmdbDataset_eval(Dataset):
    def __init__(self, lmdb_dir, split, normalized_features_wts_file=None, ignore=None, segment_len=None, data_process_fn=None) -> None:
        super().__init__()
        self.split = split
        self.ignore = ignore
        self.segment_len = segment_len
        self.data_process_fn = data_process_fn
        self.keys = []
        with open(os.path.join(lmdb_dir, 'keys.txt'), 'r') as f:
            lines = f.readlines()
            for k in lines:
                if self.ignore is not None and self.ignore in k:
                    continue
                self.keys.append(k.strip())
        # self.env = lmdb.open(lmdb_dir, readonly=True, readahead=True, lock=False)
        self.env = None
        self.lmdb_dir = lmdb_dir
        self.spec_scaler = None
        if normalized_features_wts_file is not None:
            self.spec_scaler = joblib.load(normalized_features_wts_file)
    
    def _get_env(self):
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_dir,
                readonly=True,
                lock=False,
                readahead=True,
                meminit=False
            )
        return self.env

    def __getstate__(self):
        d = self.__dict__.copy()
        d["env"] = None  # pickle할 때 env 제거
        return d

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        # txn = self.env.begin()
        # with txn.cursor() as cursor:
        #     k = self.keys[index].strip().encode()
        #     cursor.set_key(k)
        env = self._get_env()
        k = self.keys[index].encode()

        with env.begin(write=False) as txn:
            buf = txn.get(k)
        datum=SimpleDatum()
        datum.ParseFromString(buf) #cursor.value()
        data = np.frombuffer(datum.data, dtype=np.float32).reshape(-1, datum.data_dim)
        if self.spec_scaler is not None:
            data = self.spec_scaler.transform(data)
        if self.data_process_fn is not None:
            data = self.data_process_fn(data)
        wav_name = datum.wave_name.decode()
        return {'data': data, 'wav_name':wav_name}

    def collater(self, samples):
        feats = [s['data'] for s in samples]
        wav_names = [s['wav_name'] for s in samples]

        collated_feats = np.stack(feats, axis=0)

        out = {}
        out['input'] = torch.from_numpy(collated_feats)
        out['wav_names'] = wav_names

        return out