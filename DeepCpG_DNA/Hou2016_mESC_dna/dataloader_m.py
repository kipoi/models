
from __future__ import print_function
from __future__ import division

from collections import OrderedDict
import os
import sys
import warnings

import argparse
import logging
import h5py as h5
import numpy as np
import pandas as pd

import json

import six
from six.moves import range

from deepcpg import data as dat
from deepcpg.data import annotations as an
from deepcpg import evaluation as ev
from deepcpg.data import stats
from deepcpg.data import dna
from deepcpg.data import fasta
from deepcpg.data import feature_extractor as fext
from deepcpg.utils import make_dir, to_list
from deepcpg.models.utils import decode_replicate_names, encode_replicate_names, get_sample_weights

from os import path as pt

from keras import backend as K
from keras import models as km
from keras import layers as kl
from keras.utils.np_utils import to_categorical
from deepcpg.data.dna import int_to_onehot

from kipoi.metadata import GenomicRanges
from kipoi.data import BatchIterator


def prepro_pos_table(pos_tables):
    """Extracts unique positions and sorts them."""
    if not isinstance(pos_tables, list):
        pos_tables = [pos_tables]

    pos_table = None
    for next_pos_table in pos_tables:
        if pos_table is None:
            pos_table = next_pos_table
        else:
            pos_table = pd.concat([pos_table, next_pos_table])
        pos_table = pos_table.groupby('chromo').apply(
            lambda df: pd.DataFrame({'pos': np.unique(df['pos'])}))
        pos_table.reset_index(inplace=True)
        pos_table = pos_table[['chromo', 'pos']]
        pos_table.sort_values(['chromo', 'pos'], inplace=True)
    return pos_table


def split_ext(filename):
    """Remove file extension from `filename`."""
    return os.path.basename(filename).split(os.extsep)[0]

def get_fh(filename, mode, *args, **kwargs):
    """ This function is only necessary because there is a bug in the 1.0.4 release version of DeepCpG.
    """
    is_gzip = filename.endswith('.gz')
    if is_gzip:
        return gzip.open(filename, mode, *args, **kwargs)
    else:
        return open(filename, mode, *args, **kwargs)


def read_cpg_profiles(filenames, log=None, *args, **kwargs):
    """Read methylation profiles.

    Input files can be gzip compressed.

    Returns
    -------
    dict
        `dict (key, value)`, where `key` is the output name and `value` the CpG
        table.
    """

    cpg_profiles = OrderedDict()
    for filename in filenames:
        if log:
            log(filename)
        #cpg_file = dat.GzipFile(filename, 'r')
        cpg_file = get_fh(filename, 'r')
        output_name = split_ext(filename)
        cpg_profile = dat.read_cpg_profile(cpg_file, sort=True, *args, **kwargs)
        cpg_profiles[output_name] = cpg_profile
        cpg_file.close()
    return cpg_profiles



def extract_seq_windows(seq, pos, wlen, seq_index=1, assert_cpg=False):
    """Extracts DNA sequence windows at positions.

    Parameters
    ----------
    seq: str
        DNA sequence.
    pos: list
        Positions at which windows are extracted.
    wlen: int
        Window length.
    seq_index: int
        Offset at which positions start.
    assert_cpg: bool
        If `True`, check if positions in `pos` point to CpG sites.

    Returns
    -------
    np.array
        Array with integer-encoded sequence windows.
    """

    delta = wlen // 2
    nb_win = len(pos)
    seq = seq.upper()
    seq_wins = np.zeros((nb_win, wlen), dtype='int8')

    for i in range(nb_win):
        p = pos[i] - seq_index
        if p < 0 or p >= len(seq):
            raise ValueError('Position %d not on chromosome!' % (p + seq_index))
        if seq[p:p + 2] != 'CG':
            warnings.warn('No CpG site at position %d!' % (p + seq_index))
        win = seq[max(0, p - delta): min(len(seq), p + delta + 1)]
        if len(win) < wlen:
            win = max(0, delta - p) * 'N' + win
            win += max(0, p + delta + 1 - len(seq)) * 'N'
            assert len(win) == wlen
        seq_wins[i] = dna.char_to_int(win)
    # Randomly choose missing nucleotides
    idx = seq_wins == dna.CHAR_TO_INT['N']
    seq_wins[idx] = np.random.randint(0, 4, idx.sum())
    assert seq_wins.max() < 4
    if assert_cpg:
        assert np.all(seq_wins[:, delta] == 3)
        assert np.all(seq_wins[:, delta + 1] == 2)
    return seq_wins


def map_values(values, pos, target_pos, dtype=None, nan=dat.CPG_NAN):
    """Maps `values` array at positions `pos` to `target_pos`.

    Inserts `nan` for uncovered positions.
    """
    assert len(values) == len(pos)
    assert np.all(pos == np.sort(pos))
    assert np.all(target_pos == np.sort(target_pos))

    values = values.ravel()
    pos = pos.ravel()
    target_pos = target_pos.ravel()
    idx = np.in1d(pos, target_pos)
    pos = pos[idx]
    values = values[idx]
    if not dtype:
        dtype = values.dtype
    target_values = np.empty(len(target_pos), dtype=dtype)
    target_values.fill(nan)
    idx = np.in1d(target_pos, pos).nonzero()[0]
    assert len(idx) == len(values)
    assert np.all(target_pos[idx] == pos)
    target_values[idx] = values
    return target_values


def map_cpg_tables(cpg_tables, chromo, chromo_pos):
    """Maps values from cpg_tables to `chromo_pos`.

    Positions in `cpg_tables` for `chromo`  must be a subset of `chromo_pos`.
    Inserts `dat.CPG_NAN` for uncovered positions.
    """
    chromo_pos.sort()
    mapped_tables = OrderedDict()
    for name, cpg_table in six.iteritems(cpg_tables):
        cpg_table = cpg_table.loc[cpg_table.chromo == chromo]
        cpg_table = cpg_table.sort_values('pos')
        mapped_table = map_values(cpg_table.value.values,
                                  cpg_table.pos.values,
                                  chromo_pos)
        assert len(mapped_table) == len(chromo_pos)
        mapped_tables[name] = mapped_table
    return mapped_tables


def format_out_of(out, of):
    return '%d / %d (%.1f%%)' % (out, of, out / of * 100)

def select_dict(data, idx):
    data = data.copy()
    for key, value in six.iteritems(data):
        if isinstance(value, dict):
            data[key] = select_dict(value, idx)
        else:
            data[key] = value[idx]
    return data


def annotate(anno_file, chromo, pos):
    #anno_file = dat.GzipFile(anno_file, 'r')
    anno_file = get_fh(anno_file, 'r')
    anno = pd.read_table(anno_file, header=None, usecols=[0, 1, 2],
                         dtype={0: 'str', 1: 'int32', 2: 'int32'})
    anno_file.close()
    anno.columns = ['chromo', 'start', 'end']
    anno.chromo = anno.chromo.str.upper().str.replace('CHR', '')
    anno = anno.loc[anno.chromo == chromo]
    anno.sort_values('start', inplace=True)
    start, end = an.join_overlapping(anno.start.values, anno.end.values)
    anno = np.array(an.is_in(pos, start, end), dtype='int8')
    return anno


def flatten_dict(obj,output_dict, prefix="", no_prefix = False):
    prefix = prefix.rstrip("/")
    assert (isinstance(obj, dict))
    for k in obj:
        local_prefix = ""
        if not no_prefix:
            local_prefix = prefix + "/"
        local_prefix +=  str(k)
        if isinstance(obj[k], dict):
            flatten_dict(obj[k], output_dict, local_prefix, no_prefix = False)
        else:
            output_dict[local_prefix] = obj[k]



def run_dcpg_data(pos_file = None,
                    cpg_profiles = None,
                    dna_files = None,
                    cpg_wlen=None,
                    cpg_cov = 1,
                    dna_wlen=1001,
                    anno_files=None,
                    chromos = None,
                    nb_sample = None,
                    nb_sample_chromo = None,
                    chunk_size = 32768,
                    seed = 0,
                    verbose = False):
    if seed is not None:
        np.random.seed(seed)


    # FIXME
    name = "dcpg_data"
    logging.basicConfig(format='%(levelname)s (%(asctime)s): %(message)s')
    log = logging.getLogger(name)
    if verbose:
        log.setLevel(logging.DEBUG)
    else:
        log.setLevel(logging.INFO)

    # Check input arguments
    if not cpg_profiles:
        if not (pos_file or dna_files):
            raise ValueError('Position table and DNA database expected!')

    if dna_wlen and dna_wlen % 2 == 0:
        raise 'dna_wlen must be odd!'
    if cpg_wlen and cpg_wlen % 2 != 0:
        raise 'cpg_wlen must be even!'

    """
    # Parse functions for computing output statistics
    cpg_stats_meta = None
    win_stats_meta = None
    if cpg_stats:
        cpg_stats_meta = get_stats_meta(cpg_stats)
    if win_stats:
        win_stats_meta = get_stats_meta(win_stats)
    """

    outputs = OrderedDict()

    # Read single-cell profiles if provided
    if cpg_profiles:
        log.info('Reading CpG profiles ...')
        outputs['cpg'] = read_cpg_profiles(
            cpg_profiles,
            chromos=chromos,
            nb_sample=nb_sample,
            nb_sample_chromo=nb_sample_chromo,
            log=log.info)

    # Create table with unique positions
    if pos_file:
        # Read positions from file
        log.info('Reading position table ...')
        pos_table = pd.read_table(pos_file, usecols=[0, 1],
                                  dtype={0: str, 1: np.int32},
                                  header=None, comment='#')
        pos_table.columns = ['chromo', 'pos']
        pos_table['chromo'] = dat.format_chromo(pos_table['chromo'])
        pos_table = prepro_pos_table(pos_table)
    else:
        # Extract positions from profiles
        pos_tables = []
        for cpg_table in list(outputs['cpg'].values()):
            pos_tables.append(cpg_table[['chromo', 'pos']])
        pos_table = prepro_pos_table(pos_tables)

    if chromos:
        pos_table = pos_table.loc[pos_table.chromo.isin(chromos)]
    if nb_sample_chromo:
        pos_table = dat.sample_from_chromo(pos_table, nb_sample_chromo)
    if nb_sample:
        pos_table = pos_table.iloc[:nb_sample]

    log.info('%d samples' % len(pos_table))


    # Iterate over chromosomes
    # ------------------------
    for chromo in pos_table.chromo.unique():
        log.info('-' * 80)
        log.info('Chromosome %s ...' % (chromo))
        idx = pos_table.chromo == chromo
        chromo_pos = pos_table.loc[idx].pos.values
        chromo_outputs = OrderedDict()

        if 'cpg' in outputs:
            # Concatenate CpG tables into single nb_site x nb_output matrix
            chromo_outputs['cpg'] = map_cpg_tables(outputs['cpg'],
                                                   chromo, chromo_pos)
            chromo_outputs['cpg_mat'] = np.vstack(
                list(chromo_outputs['cpg'].values())).T
            assert len(chromo_outputs['cpg_mat']) == len(chromo_pos)

        if 'cpg_mat' in chromo_outputs and cpg_cov:
            cov = np.sum(chromo_outputs['cpg_mat'] != dat.CPG_NAN, axis=1)
            assert np.all(cov >= 1)
            idx = cov >= cpg_cov
            tmp = '%s sites matched minimum coverage filter'
            tmp %= format_out_of(idx.sum(), len(idx))
            log.info(tmp)
            if idx.sum() == 0:
                continue

            chromo_pos = chromo_pos[idx]
            chromo_outputs = select_dict(chromo_outputs, idx)

        # Read DNA of chromosome
        chromo_dna = None
        if dna_files:
            chromo_dna = fasta.read_chromo(dna_files, chromo)

        annos = None
        if anno_files:
            log.info('Annotating CpG sites ...')
            annos = dict()
            for anno_file in anno_files:
                name = split_ext(anno_file)
                annos[name] = annotate(anno_file, chromo, chromo_pos)

        # Iterate over chunks
        # -------------------
        nb_chunk = int(np.ceil(len(chromo_pos) / chunk_size))
        for chunk in range(nb_chunk):
            log.info('Chunk \t%d / %d' % (chunk + 1, nb_chunk))
            chunk_start = chunk * chunk_size
            chunk_end = min(len(chromo_pos), chunk_start + chunk_size)
            chunk_idx = slice(chunk_start, chunk_end)
            chunk_pos = chromo_pos[chunk_idx]

            chunk_outputs = select_dict(chromo_outputs, chunk_idx)

            #filename = 'c%s_%06d-%06d.h5' % (chromo, chunk_start, chunk_end)
            #filename = os.path.join(out_dir, filename)
            #chunk_file = h5.File(filename, 'w')

            # Write positions
            #chunk_file.create_dataset('chromo', shape=(len(chunk_pos),),
            #                          dtype='S2')
            #chunk_file['chromo'][:] = chromo.encode()
            #chunk_file.create_dataset('pos', data=chunk_pos, dtype=np.int32)

            yield_dict = {}

            yield_dict["chromo"] = np.array([chromo.encode()]*len(chunk_pos), dtype='S2')
            yield_dict["pos"] = np.array(chunk_pos, dtype=np.int32)


            if len(chunk_outputs):
                #out_group = chunk_file.create_group('outputs')
                yield_dict["outputs"] = {}
                out_group = yield_dict["outputs"]


            # Write cpg profiles
            if 'cpg' in chunk_outputs:
                yield_dict["outputs"]['cpg']={}
                for name, value in six.iteritems(chunk_outputs['cpg']):
                    assert len(value) == len(chunk_pos)
                    # Round continuous values
                    #out_group.create_dataset('cpg/%s' % name,
                    #                         data=value.round(),
                    #                         dtype=np.int8,
                    #                         compression='gzip')
                    out_group['cpg'][name] = np.array(value.round(), np.int8)
                """
                # Compute and write statistics
                if cpg_stats_meta is not None:
                    log.info('Computing per CpG statistics ...')
                    cpg_mat = np.ma.masked_values(chunk_outputs['cpg_mat'],
                                                  dat.CPG_NAN)
                    mask = np.sum(~cpg_mat.mask, axis=1)
                    mask = mask < cpg_stats_cov
                    for name, fun in six.iteritems(cpg_stats_meta):
                        stat = fun[0](cpg_mat).data.astype(fun[1])
                        stat[mask] = dat.CPG_NAN
                        assert len(stat) == len(chunk_pos)
                        out_group.create_dataset('cpg_stats/%s' % name,
                                                 data=stat,
                                                 dtype=fun[1],
                                                 compression='gzip')
                """

            # Write input features
            #in_group = chunk_file.create_group('inputs')
            yield_dict["inputs"] = {}
            in_group = yield_dict["inputs"]

            # DNA windows
            if chromo_dna:
                log.info('Extracting DNA sequence windows ...')
                dna_wins = extract_seq_windows(chromo_dna, pos=chunk_pos,
                                               wlen=dna_wlen)
                assert len(dna_wins) == len(chunk_pos)
                #in_group.create_dataset('dna', data=dna_wins, dtype=np.int8,
                #                        compression='gzip')
                in_group['dna'] = np.array(dna_wins, dtype=np.int8)

            # CpG neighbors
            if cpg_wlen:
                log.info('Extracting CpG neighbors ...')
                cpg_ext = fext.KnnCpgFeatureExtractor(cpg_wlen // 2)
                #context_group = in_group.create_group('cpg')
                in_group['cpg'] = {}
                context_group = in_group['cpg']
                # outputs['cpg'], since neighboring CpG sites might lie
                # outside chunk borders and un-mapped values are needed
                for name, cpg_table in six.iteritems(outputs['cpg']):
                    cpg_table = cpg_table.loc[cpg_table.chromo == chromo]
                    state, dist = cpg_ext.extract(chunk_pos,
                                                  cpg_table.pos.values,
                                                  cpg_table.value.values)
                    nan = np.isnan(state)
                    state[nan] = dat.CPG_NAN
                    dist[nan] = dat.CPG_NAN
                    # States can be binary (np.int8) or continuous
                    # (np.float32).
                    state = state.astype(cpg_table.value.dtype, copy=False)
                    dist = dist.astype(np.float32, copy=False)

                    assert len(state) == len(chunk_pos)
                    assert len(dist) == len(chunk_pos)
                    assert np.all((dist > 0) | (dist == dat.CPG_NAN))

                    #group = context_group.create_group(name)
                    #group.create_dataset('state', data=state,
                    #                     compression='gzip')
                    #group.create_dataset('dist', data=dist,
                    #                     compression='gzip')
                    context_group[name] = {'state': state, 'dist':dist}

            """
            if win_stats_meta is not None and cpg_wlen:
                log.info('Computing window-based statistics ...')
                states = []
                dists = []
                cpg_states = []
                cpg_group = out_group['cpg']
                context_group = in_group['cpg']
                for output_name in six.iterkeys(cpg_group):
                    state = context_group[output_name]['state']#.value
                    states.append(np.expand_dims(state, 2))
                    dist = context_group[output_name]['dist']#.value
                    dists.append(np.expand_dims(dist, 2))
                    #cpg_states.append(cpg_group[output_name].value)
                    cpg_states.append(cpg_group[output_name])
                # samples x outputs x cpg_wlen
                states = np.swapaxes(np.concatenate(states, axis=2), 1, 2)
                dists = np.swapaxes(np.concatenate(dists, axis=2), 1, 2)
                cpg_states = np.expand_dims(np.vstack(cpg_states).T, 2)
                cpg_dists = np.zeros_like(cpg_states)
                states = np.concatenate([states, cpg_states], axis=2)
                dists = np.concatenate([dists, cpg_dists], axis=2)

                for wlen in win_stats_wlen:
                    idx = (states == dat.CPG_NAN) | (dists > wlen // 2)
                    states_wlen = np.ma.masked_array(states, idx)
                    group = out_group.create_group('win_stats/%d' % wlen)
                    for name, fun in six.iteritems(win_stats_meta):
                        stat = fun[0](states_wlen)
                        if hasattr(stat, 'mask'):
                            idx = stat.mask
                            stat = stat.data
                            if np.sum(idx):
                                stat[idx] = dat.CPG_NAN
                        group.create_dataset(name, data=stat, dtype=fun[1],
                                             compression='gzip')

            if annos:
                log.info('Adding annotations ...')
                group = in_group.create_group('annos')
                for name, anno in six.iteritems(annos):
                    group.create_dataset(name, data=anno[chunk_idx],
                                         dtype='int8',
                                         compression='gzip')
            """

            #chunk_file.close()

            flat_dict={}
            flatten_dict(yield_dict, flat_dict, no_prefix = True)
            yield flat_dict

    log.info('Done preprocessing!')




##### This function is needed to extract info on model architecture so that the output can be generated correctly.
def data_reader_config_from_model(model, config_out_fpath = None, replicate_names=None):
    """Return :class:`DataReader` from `model`.
    Builds a :class:`DataReader` for reading data for `model`.
    Parameters
    ----------
    model: :class:`Model`.
        :class:`Model`.
    outputs: bool
        If `True`, return output labels.
    replicate_names: list
        Name of input cells of `model`.
    Returns
    -------
    :class:`DataReader`
        Instance of :class:`DataReader`.
    """
    use_dna = False
    dna_wlen = None
    cpg_wlen = None
    output_names = None
    encode_replicates = False
    #
    input_shapes = to_list(model.input_shape)
    for input_name, input_shape in zip(model.input_names, input_shapes):
        if input_name == 'dna':
            # Read DNA sequences.
            use_dna = True
            dna_wlen = input_shape[1]
        elif input_name.startswith('cpg/state/'):
            # DEPRECATED: legacy model. Decode replicate names from input name.
            replicate_names = decode_replicate_names(input_name.replace('cpg/state/', ''))
            assert len(replicate_names) == input_shape[1]
            cpg_wlen = input_shape[2]
            encode_replicates = True
        elif input_name == 'cpg/state':
            # Read neighboring CpG sites.
            if not replicate_names:
                raise ValueError('Replicate names required!')
            if len(replicate_names) != input_shape[1]:
                tmp = '{r} replicates found but CpG model was trained with' \
                    ' {s} replicates. Use `--nb_replicate {s}` or ' \
                    ' `--replicate_names` option to select {s} replicates!'
                tmp = tmp.format(r=len(replicate_names), s=input_shape[1])
                raise ValueError(tmp)
            cpg_wlen = input_shape[2]
    output_names = model.output_names
    config = {"output_names":output_names,
                      "use_dna":use_dna,
                      "dna_wlen":dna_wlen,
                      "cpg_wlen":cpg_wlen,
                      "replicate_names":replicate_names,
                      "encode_replicates":encode_replicates}
    if config_out_fpath is not None:
        with open(config_out_fpath, "w") as ofh:
            json.dump(config, ofh)
    return config


def data_reader_from_config(config_fpath, outputs = True):
    with open(config_fpath, "r") as ifh:
        dr_kwargs = json.load(ifh)

    if not outputs:
        dr_kwargs["output_names"] = None
    
    return DataReader(**dr_kwargs)



class DataReader(object):
    """Read data from `dcpg_data.py` output files.
    Generator to read data batches from `dcpg_data.py` output files. Reads data
    using :func:`hdf.reader` and pre-processes data.
    Parameters
    ----------
    output_names: list
        Names of outputs to be read.
    use_dna: bool
        If `True`, read DNA sequence windows.
    dna_wlen: int
        Maximum length of DNA sequence windows.
    replicate_names: list
        Name of cells (profiles) whose neighboring CpG sites are read.
    cpg_wlen: int
        Maximum number of neighboring CpG sites.
    cpg_max_dist: int
        Value to threshold the distance of neighboring CpG sites.
    encode_replicates: bool
        If `True`, encode replicated names in key of returned dict. This option
        is deprecated and will be removed in the future.
    Returns
    -------
    tuple
        `dict` (`inputs`, `outputs`, `weights`), where `inputs`, `outputs`,
        `weights` is a `dict` of model inputs, outputs, and output weights.
        `outputs` and `weights` are not returned if `output_names` is undefined.
    """
    def __init__(self, output_names=None,
                 use_dna=True, dna_wlen=None,
                 replicate_names=None, cpg_wlen=None, cpg_max_dist=25000,
                 encode_replicates=False):
        self.output_names = to_list(output_names)
        self.use_dna = use_dna
        self.dna_wlen = dna_wlen
        self.replicate_names = to_list(replicate_names)
        self.cpg_wlen = cpg_wlen
        self.cpg_max_dist = cpg_max_dist
        self.encode_replicates = encode_replicates

    def _prepro_dna(self, dna):
        """Preprocess DNA sequence windows."""
        if self.dna_wlen:
            cur_wlen = dna.shape[1]
            center = cur_wlen // 2
            delta = self.dna_wlen // 2
            dna = dna[:, (center - delta):(center + delta + 1)]
        return int_to_onehot(dna)

    def _prepro_cpg(self, states, dists):
        """Preprocess the state and distance of neighboring CpG sites."""
        prepro_states = []
        prepro_dists = []
        for state, dist in zip(states, dists):
            nan = state == dat.CPG_NAN
            if np.any(nan):
                state[nan] = np.random.binomial(1, state[~nan].mean(),
                                                nan.sum())
                dist[nan] = self.cpg_max_dist
            dist = np.minimum(dist, self.cpg_max_dist) / self.cpg_max_dist
            prepro_states.append(np.expand_dims(state, 1))
            prepro_dists.append(np.expand_dims(dist, 1))
        prepro_states = np.concatenate(prepro_states, axis=1)
        prepro_dists = np.concatenate(prepro_dists, axis=1)
        if self.cpg_wlen:
            center = prepro_states.shape[2] // 2
            delta = self.cpg_wlen // 2
            tmp = slice(center - delta, center + delta)
            prepro_states = prepro_states[:, :, tmp]
            prepro_dists = prepro_dists[:, :, tmp]
        return (prepro_states, prepro_dists)


    def __call__(self, dcpg_data_kwargs, class_weights=None):
        """Return generator for reading data from `data_files`.
        Parameters
        ----------
        class_weights: dict
            dict of dict with class weights of individual outputs.
        *args: list
            Unnamed arguments passed to :func:`hdf.reader`
        *kwargs: dict
            Named arguments passed to :func:`hdf.reader`
        Returns
        -------
        generator
            Python generator for reading data.
        """
        names = []
        if self.use_dna:
            names.append('inputs/dna')

        if self.replicate_names:
            for name in self.replicate_names:
                names.append('inputs/cpg/%s/state' % name)
                names.append('inputs/cpg/%s/dist' % name)

        if self.output_names:
            for name in self.output_names:
                names.append('outputs/%s' % name)

        # check that the kwargs fit the model:
        if self.dna_wlen is not None:
            if ("dna_wlen" in dcpg_data_kwargs) and (dcpg_data_kwargs["dna_wlen"] != self.dna_wlen):
                log.warn("dna_wlen does not match requirements of the model (%d)"%self.dna_wlen)
            dcpg_data_kwargs["dna_wlen"] = self.dna_wlen

        if self.cpg_wlen is not None:
            if ("cpg_wlen" in dcpg_data_kwargs) and (dcpg_data_kwargs["cpg_wlen"] != self.cpg_wlen):
                log.warn("cpg_wlen does not match requirements of the model (%d)"%self.cpg_wlen)
            dcpg_data_kwargs["cpg_wlen"] = self.cpg_wlen

        ### Here insert the calling of run_dcpg_data(), require reformatting of the output
        data_iter = run_dcpg_data(**dcpg_data_kwargs)
        id_ctr_offset = 0
        for data_raw in data_iter:
            for k in names:
                if k not in data_raw:
                    raise ValueError('%s does not exist! Sample mismatch between model and input data?' % k)
            inputs = dict()

            if self.use_dna:
                inputs['dna'] = self._prepro_dna(data_raw['inputs/dna'])

            if self.replicate_names:
                states = []
                dists = []
                for name in self.replicate_names:
                    tmp = 'inputs/cpg/%s/' % name
                    states.append(data_raw[tmp + 'state'])
                    dists.append(data_raw[tmp + 'dist'])
                states, dists = self._prepro_cpg(states, dists)
                if self.encode_replicates:
                    # DEPRECATED: to support loading data for legacy models
                    tmp = '/' + encode_replicate_names(self.replicate_names)
                else:
                    tmp = ''
                inputs['cpg/state%s' % tmp] = states
                inputs['cpg/dist%s' % tmp] = dists

            outputs = dict()
            weights = dict()
            if not self.output_names:
                #yield inputs
                pass
            else:
                for name in self.output_names:
                    outputs[name] = data_raw['outputs/%s' % name]
                    cweights = class_weights[name] if class_weights else None
                    weights[name] = get_sample_weights(outputs[name], cweights)
                    if name.endswith('cat_var'):
                        output = outputs[name]
                        outputs[name] = to_categorical(output, 3)
                        outputs[name][output == dat.CPG_NAN] = 0

                #yield (inputs, outputs, weights)
            meta_data = {}
            # metadata is only generated if the respective window length is given
            if ("dna_wlen" in dcpg_data_kwargs) and (dcpg_data_kwargs["dna_wlen"] is not None):
                wlen = dcpg_data_kwargs["dna_wlen"]
                delta_pos = wlen // 2
                chrom = data_raw["chromo"].astype(str)
                start = data_raw["pos"] - delta_pos
                end = data_raw["pos"] + delta_pos + 1
                meta_data["dna_ranges"] = GenomicRanges(chrom, start, end, np.arange(chrom.shape[0])+id_ctr_offset)
            
            if ("cpg_wlen" in dcpg_data_kwargs) and (dcpg_data_kwargs["cpg_wlen"] is not None):
                wlen = dcpg_data_kwargs["cpg_wlen"]
                delta_pos = wlen // 2
                chrom = data_raw["chromo"].astype(str)
                start = data_raw["pos"] - delta_pos
                end = data_raw["pos"] + delta_pos + 1
                meta_data["cpg_ranges"] = GenomicRanges(chrom, start, end, np.arange(chrom.shape[0])+id_ctr_offset)

            id_ctr_offset += data_raw["chromo"].shape[0]
            # Weights are not supported at the moment 
            yield {"inputs": inputs, "targets":outputs, "metadata":meta_data}


class Dataloader(BatchIterator):
    def __init__(self, cpg_profiles, reference_fpath, batch_size = 100, outputs = True,
                class_weights=None):
        # derive the config file path from the path of the dataloader_m.py file:
        config_fpath = os.path.dirname(os.path.realpath(__file__)) + "/model_config.json"
        # compile arguments:
        assert isinstance(cpg_profiles, list)
        dcpg_data_kwargs = {"cpg_profiles": cpg_profiles,
                    "dna_files" : [reference_fpath],
                    "dna_wlen":1001,
                    "cpg_wlen":50,
                    "chunk_size" : batch_size} # chunksize === batch_size in current setup!
        self.dr_iter_obj = data_reader_from_config(config_fpath, outputs)(dcpg_data_kwargs, class_weights)

    def __next__(self):
        return self.dr_iter_obj.__next__()

    def __iter__(self):
        return self.dr_iter_obj

