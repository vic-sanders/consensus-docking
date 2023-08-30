#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
import numba as nb
import itertools
import random
import sys
import re
import os

from dataclasses import dataclass, field
from scipy import spatial, optimize
from typing import List, Tuple, Dict
from tqdm.auto import tqdm


VINA_RECORD_REGEX = re.compile(r'MODEL.+?ENDMDL', re.DOTALL)
MOL2_RECORD_REGEX = re.compile(r'(?<=@<TRIPOS>ATOM)[^@]+', re.DOTALL)


@dataclass(unsafe_hash=True)
class PoseRecord:
    
    filepath: str
    coords: np.ndarray = field(compare=False, repr=False)
    atomid: np.ndarray = field(compare=False, repr=False)
    atomtype: np.ndarray = field(compare=False, repr=False)
    what: str = field(default=None, compare=False, repr=False)
    score: float = field(default=0.0, compare=False)
    model: int = field(default=1)
        
    @property
    def filename(self) -> str:
        return os.path.split(self.filepath)[1]
        
    def geometric_center(self) -> np.ndarray:
        return np.mean(self.coords, axis=0)
    
    def rmsd(self, other: PoseRecord, method='auto') -> float:
        assert len(self) == len(other), 'Unequal number of atoms!'
        if method == 'auto':  # auto determine rmsd method
            if np.array_equal(self.atomid, other.atomid):
                 method = 'naive'
            elif set(self.atomid) == set(other.atomid):
                method = 'sort'
            else: method = 'lsa'
        if method == 'naive':
            return _rmsd_jit(self.coords, other.coords)
        elif method == 'sort':
            ix_i = self.atomid.argsort()
            ix_j = other.atomid.argsort()
            return _rmsd_jit(self.coords[ix_i], other.coords[ix_j])
        elif method == 'lsa':
            return _rmsd_jit(*_lsa_ordering(self, other))
        else:
            raise NameError(f'RMSD method: {method} not implemented!')
    
    def __mod__(self, other: PoseRecord) -> float:
        return self.rmsd(other, 'auto')
    
    def __len__(self) -> int:
        return self.coords.shape[0]
    
    
@dataclass
class PoseCluster:
    
    records: List[PoseRecord] = field(repr=False)
        
    def centers(self) -> np.ndarray:
        mapped = map(lambda x: x.geometric_center(), self.records)
        return np.vstack(list(mapped))
        
    def centroid(self) -> np.ndarray:
        return np.vstack(self.centers()).mean(axis=0)
    
    def max_radius(self) -> float:
        centers = self.centers()
        return spatial.distance.cdist(centers, centers).max()
    
    def scores(self) -> np.ndarray:
        return np.fromiter((x.score for x in self), dtype=float)
    
    def rmsd_matrix(self, method='auto', square: bool = True) -> np.ndarray:
        if len(self) <= 1:
            return np.array([[0.0,]])
        rmsd_arr = []
        for i, j in itertools.combinations(self.records, 2):
            rmsd_arr.append(i.rmsd(j, method=method))
        if square:
            return spatial.distance.squareform(rmsd_arr)
        return np.array(rmsd_arr)
    
    def mean_rmsd(self, method='auto'):
        rmsd_mat = self.rmsd_matrix(method, False)
        return np.mean(rmsd_mat.ravel())
    
    def append(self, record: PoseRecord):
        self.records.append(record)
        
    def __getitem__(self, index) -> PoseRecord:
        return self.records[index]
        
    def __len__(self) -> int:
        return len(self.records)
    
    def __repr__(self) -> str:
        return '{}(records={})'.format(
            self.__class__.__name__,
            len(self)
        )
    

@nb.njit(cache=True, fastmath=True)  # ~10x speed gain
def _rmsd_jit(i: np.ndarray, j: np.ndarray) -> float:
    return np.sqrt(((i - j) ** 2).sum(axis=-1).mean())


def _lsa_ordering(i: PoseRecord, j: PoseRecord) -> float:
    """"""
    i_map, j_map = [], []
    for atype in np.unique(i.atomtype):
        i_idx = np.argwhere(i.atomtype == atype).flatten()
        j_idx = np.argwhere(j.atomtype == atype).flatten()
        if len(i_idx) != len(j_idx):
            raise ValueError('Unequal number of "%s" atoms' % a_type)
        if len(i_idx) == 1:
            i_map.append(i_idx)
            j_map.append(j_idx)
            continue
        cost = spatial.distance.cdist(i.coords[i_idx], j.coords[j_idx])
        cost = cost - cost.min(axis=0) - cost.min(axis=1).reshape(-1, 1)
        tmp_i, tmp_j = optimize.linear_sum_assignment(cost)
        i_map.append(i_idx[tmp_i])
        j_map.append(j_idx[tmp_j])
        i_coords = i.coords[np.hstack(i_map)]
        j_coords = j.coords[np.hstack(j_map)]
    return i_coords, j_coords


def get_file_extension(filepath: str) -> str:
    """"""
    return os.path.splitext(filepath)[1]
        
        
def parse_vina(filepath: str, read_h: bool = False) -> List[PoseRecord]:
    """"""
    records = []
    with open(filepath, 'r') as buffer:
        raw = buffer.read()
    models = VINA_RECORD_REGEX.findall(raw)
    assert len(models) > 0, f'Empty file: {filepath}!'
    atom_records = map(lambda x: _read_model_vina(x, read_h), models)
    for ix, (aid, xyz, elm, score) in enumerate(atom_records):
        r = PoseRecord(filepath, xyz, aid, elm, 'vina', score, ix + 1)
        records.append(r)
    return records


def _read_model_vina(model: str, read_h: bool) -> Tuple:
    """"""
    aid, xyz, elm, score = [], [], [], 0.0
    lines = model.splitlines()
    _ = lines.pop(0)  # model
    for line in lines:
        if line.startswith('REMARK VINA RESULT'):
            score = float(line.split()[3])
        elif line.startswith('ATOM'):
            tkns = line.strip().split()
            atm = tkns[2]
            if atm.startswith('H') and read_h is False:
                continue
            aid.append(atm)
            xyz.append((tkns[6], tkns[7], tkns[8]))
            elm.append(atm[0])
    aid = np.asanyarray(aid, dtype=np.dtype('<U4'))
    xyz = np.asanyarray(xyz, dtype=np.float32)
    elm = np.asanyarray(elm, dtype=np.dtype('<U1'))
    return aid, xyz, elm, score


def parse_mol2(filepath: str, read_h: bool = False) -> List[PoseRecord]:
    """"""
    records = []
    with open(filepath, 'r') as buffer:
        raw = buffer.read()
    models = MOL2_RECORD_REGEX.findall(raw)
    assert len(models) > 0, f'Empty file: {filepath}!'
    atom_records = map(lambda x: _read_model_mol2(x, read_h), models)
    for ix, (aid, xyz, elm) in enumerate(atom_records):
        r = PoseRecord(filepath, xyz, aid, elm, 'mol2', 0.0, ix + 1)
        records.append(r)
    return records


def _read_model_mol2(model: str, read_h: bool) -> Tuple:
    """"""
    aid, xyz, elm = [], [], []
    lines = model.splitlines()
    atomr = [i for i in lines if len(i) > 0]
    for record in atomr:
        tkns = record.split()
        atm = tkns[1]
        if atm.startswith('H') and read_h is False:
            continue
        if atm.startswith('*'):
            continue
        aid.append(atm)
        xyz.append((tkns[2], tkns[3], tkns[4]))
        elm.append(aid[0])
    aid = np.asanyarray(aid, dtype=np.dtype('<U4'))
    xyz = np.asanyarray(xyz, dtype=np.float32)
    elm = np.asanyarray(elm, dtype=np.dtype('<U1'))
    return aid, xyz, elm


def parse_multiple_records(filepaths: List[str], read_h: bool = False) -> List[PoseRecord]:
    """"""
    records = []
    for f in filepaths:
        ext = get_file_extension(f)
        if ext == '.mol2':
            records.extend(parse_mol2(f))
        else:  # default to vina file
            records.extend(parse_vina(f))
    return records


def enumerate_dir(path: str, pattern: bool = None) -> List[str]:
    """"""
    assert os.path.isdir(path), f'Path: {path} is not a directory!' 
    pattern = re.compile(pattern) if pattern else re.compile(r'^.*$')
    paths = [os.path.join(path, x) for x in os.listdir(path) if re.match(pattern, x)]
    return paths


def get_plants_score_dict(filename: str) -> Dict[str, float]:
    """"""
    score_dict = {}
    with open(filename, 'r') as f:
        next(f)
        data = f.readlines()
    for i in data:
        split_data = i.split(',')
        name = split_data[0] + '_p.mol2'
        score_dict[name] = float(split_data[1])
    return score_dict


def add_scores_from_dict(records: List[PoseRecord], scores: Dict[str, float]) -> None:
    """"""
    for record in records:
        record.score = scores.get(record.filename, 0.0)
        
        
def normalise_scores(records: List[PoseRecord]) -> None:
    """"""
    scores = np.fromiter((x.score for x in records), dtype=float)
    scores = (scores - scores.min()) / scores.ptp()
    for score, record in zip(scores, records):
        record.score = score


def rmsd_cluster(
    records: List[PoseRecord],
    threshold: float = 2.0,
    method: str = 'auto',
    progress: bool = True
) -> List[PoseCluster]: 
    """"""
    clusters, removed, p = [], set(), not progress
    for i in tqdm(records, desc='Clustering (RMSD)', disable=p):
        if i in removed:
            continue
        cluster = PoseCluster([i])
        for j in records:
            if j == i or j in removed:
                continue
            if i.rmsd(j, method=method) <= threshold:
                cluster.append(j)
                removed.add(j)
        clusters.append(cluster)
    return clusters


def rmsd_cull(records: List[PoseRecord], pick: str = 'max', **kwargs) -> List[PoseRecord]:
    """"""
    clusters = rmsd_cluster(records, **kwargs)
    if pick == 'max':
        scores = [x.scores for x in clusters]
        culled_list = [c[np.argmax(s)] for s, c in zip(scores, clusters)]
    elif pick == 'min':
        scores = [x.scores for x in clusters]
        culled_list = [c[np.argmin(s)] for s, c in zip(scores, clusters)]
    elif pick == 'random':
        culled_list = [random.choice(x) for x in clusters]
    else:
        raise ValueError(f'Unknown pick method {pick}!')
    return culled_list


def consensus_cluster(
    runs: List[Tuple[List[PoseRecord], str]],
    threshold: float = 2.0
) -> List[PoseCluster]:
    """"""
    n, clusters = len(runs), []
    assert n != 0, 'Expected a list with length > 0'
    if n == 1: return rmsd_cluster(runs[0][0])
    reference, ref_name = runs[0]
    for r_record in tqdm(reference, desc='Consensus RMSD clustering'):
        consensus, cluster = {ref_name}, [r_record]
        for query, query_name in runs[1:]:
            for q_record in query:
                if r_record % q_record <= 2.0:
                    cluster.append(q_record)
                    consensus.add(query_name)
        for q_record in reference:
            if q_record == r_record:
                continue
            if r_record % q_record <= 2.0:
                cluster.append(q_record)
        if len(consensus) >= 2:
            clusters.append(PoseCluster(cluster))
    return clusters


def cc_unique_records(cc: List[PoseCluster]) -> List[PoseRecord]:
    """"""
    return list(set(itertools.chain(*[x.records for x in cc])))


def split_and_sort(records: List[PoseRecord], patterns: List[str]) -> List[List[PoseRecord]]:
    split = []
    for pattern in patterns:
        r_split = []
        for record in unique_records:
            if re.match(pattern, record.filename):
                r_split.append(record)
        r_split = sorted(r_split, key=lambda x: x.score, reverse=True)
        split.append(r_split)
    return split


if __name__ == '__main__':
    
    print('Consensus Clustering (consensus.py):\n')
    
    # Load and cull VINA data
    print('Processing VINA files...')
    VINA = os.path.abspath('./VINA_7KOX_WG009')
    vina_pattern = r'^VINA_.*_v$'
    vina_paths = enumerate_dir(VINA, vina_pattern)
    vina_records = parse_multiple_records(vina_paths)
    vina_records = sorted(vina_records, key=lambda x: x.score, reverse=True)
    normalise_scores(vina_records)
    vina_records = rmsd_cull(vina_records, method='naive')
    print(f'Extracted {len(vina_records)} pose clusters.\n')
    print (vina_records)
    
    # Load and cull PLANTS data
    print('Processing PLANTS files...')
    PLANTS = os.path.abspath('./PLANTS_7KOX_WG009')
    plants_pattern = r'^WG009_entry_.*_p.mol2$'
    plants_paths = enumerate_dir(PLANTS, plants_pattern)
    plants_records = parse_multiple_records(plants_paths)
    plants_scores = get_plants_score_dict(os.path.join(PLANTS, 'ranking.csv'))
    add_scores_from_dict(plants_records, plants_scores)
    plants_records = sorted(plants_records, key=lambda x: x.score, reverse=True)
    normalise_scores(plants_records)
    plants_records = rmsd_cull(plants_records, method='naive')
    print(f'Extracted {len(plants_records)} pose clusters.\n')
    
    
    # Consensus clustering (VINA vs PLANTS) (easy to extend to multiple runs)
    runs = [(vina_records, 'vina'), (plants_records, 'plants')]
    consensus_clusters = consensus_cluster(runs, threshold=2.0)
    print(f'Found {len(consensus_clusters)} consensus cluster(s)\n')
    if len(consensus_clusters) == 0:
        sys.exit(0)
      
    # Get unique records within all clusters
    unique_records = cc_unique_records(consensus_clusters)
    final_solutions = split_and_sort(unique_records, [vina_pattern, plants_pattern])
    
    # Print these solutions!
    print('Consenus poses:')
    for ix, solution_lst in enumerate(final_solutions):
        print(f'\nSolutions {ix + 1}:' + '\n------------')
        for solution in solution_lst:
            print(f'File: {solution.filename}, normalised score: {solution.score:3f}')
            
    # Output a pymol script for visualisation of clusters (could be better).
    # I.e. select residues within centroid region
    # I.e. group molecules by cluster etc.
    binding_site = os.path.join(PLANTS, 'protein_bindingsite_fixed.mol2')
    with open('cluster_result.py', 'w') as pymol:
        pymol.write('#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n')
        pymol.write('from pymol import cmd\n')
        pymol.write('cmd.load("{}")\n'.format(binding_site))  # load protein
        pymol.write('cmd.set("sphere_scale",0.8)\n')
        for ix, cluster in enumerate(consensus_clusters):
            name = f'cluster_{ix + 1}'
            centroid = cluster.centroid()
            pos = ','.join(map(str, centroid))
            pymol.write('cmd.pseudoatom("{}", pos=[{}])\n'.format(name, pos))
            pymol.write('cmd.show("spheres", "{}")\n'.format(name))
        for record in unique_records:
            filepath = record.filepath
            if filepath.endswith('_v'):
                pymol.write('cmd.load("{}", format="pdb")\n'.format(record.filepath))
            else:
                pymol.write('cmd.load("{}")\n'.format(record.filepath))
            pymol.write('cmd.show("sticks", "{}")\n'.format(os.path.split(record.filename)[0]))
                
                
    # Do other stuff.....
            
    print('\nFin.')   
    
