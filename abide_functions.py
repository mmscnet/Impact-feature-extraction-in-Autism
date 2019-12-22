"""
ABIDE funstions
"""

import numpy as np
import os
from os.path import join

def get_timeseries(phenotypic, atlas, timeseries_dir):
    _timeseries = []
    _IDs_subject = []
    _diagnosis = []
    _subject_ids = phenotypic['SUB_ID']
    for index, subject_id in enumerate(_subject_ids):
        this_pheno = phenotypic[phenotypic['SUB_ID'] == subject_id]
        this_file=this_pheno['FILE_ID'].values[0]
        this_timeseries = join(timeseries_dir, 'rois_'+ atlas,
                               this_file + '_rois_'+ atlas+ '.1D')
        if os.path.exists(this_timeseries):
            _timeseries.append(np.loadtxt(this_timeseries, skiprows=0))
            _IDs_subject.append(subject_id)
            _diagnosis.append(this_pheno['DX_GROUP'].values[0])
    return _timeseries, _diagnosis, _IDs_subject