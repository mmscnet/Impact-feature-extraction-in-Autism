""" 
------------------------------------------------------------------------------------------
Article:    Impact of feature extraction in Autism prediction from functional connectivity 
Authors:    Moises Silva - Manuel Graña
Date:       12/22/2019
------------------------------------------------------------------------------------------
"""
# General
import numpy as np
import pandas as pd
# scikit-learn
from sklearn.covariance import LedoitWolf
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
# nilearn
from nilearn.connectome import ConnectivityMeasure
# function owner
from abide_estimators import acc_abide_classifiers
import abide_math 
import abide_functions

# Path to data directory
timeseries_dir = './abide/Outputs/cpac/filt_noglobal'

# Variables' atlas, dimensions and measures
atlases    = ['aal', 'cc200', 'dosenbach160', 'ez', 'ho', 'tt']
dimensions = {'aal': 116, 'cc200': 200, 'dosenbach160': 161, 'ez': 116, 'ho': 111, 'tt': 97}
measures   = ['correlation', 'partial correlation', 'tangent']

# Data archive ABIDE results
columns = ['atlas', 'measure', 'classifier', 'scores', 'iter_shuffle_split',
           'dataset', 'dimensionality', 'scoring', 'reduction']
# Dictionary archive ABIDE results
abide_results = dict()
for column_name in columns:
    abide_results.setdefault(column_name, [])
    
# Phenotypic Archive - ABIDE Preprocessed
pheno_dir = 'Phenotypic_V1_0b_preprocessed1.csv'
phenotypic = pd.read_csv(pheno_dir)

#scoring_type = 'roc_auc'
scoring_type = 'accuracy'

# n_splits 
cv = StratifiedShuffleSplit(n_splits=2, test_size=0.25, random_state=0)

print("Running predictions:", scoring_type)
print("-------------------------------")

for atlas in atlases:
    print("Timeseries --> Correlation's matrix")
    
    timeseries, diagnosis, id_subjects = abide_functions.get_timeseries(phenotypic, atlas, timeseries_dir)
    diagnosis = np.asarray(diagnosis)
    diagnosis = np.where(diagnosis==1, -1, diagnosis)
    diagnosis = np.where(diagnosis==2, 1, diagnosis)
    iter_for_prediction = cv.split(timeseries, diagnosis)

    for index, (train_index, test_index) in enumerate(iter_for_prediction):
        print('Fold --> Train and test')
        
        for measure in measures:
            print('Measures -->')
            
            correlation_measure = ConnectivityMeasure(cov_estimator=LedoitWolf(assume_centered=True, block_size=1000, store_precision=False), kind=measure, vectorize=True)
            abide_data_orig = correlation_measure.fit_transform(timeseries)
            diagnosis = np.asarray(diagnosis)
            
            # Percentiles 
            print('Percentiles')
            print('-----------')
            
            value_percentiles = np.asarray([90, 95, 96, 97, 98, 99])
            for value_percentil in value_percentiles:
                
                abide_data_train =  abide_data_orig[train_index].T
                _pearson = np.asarray([abide_math.pearson_function(mat, diagnosis[train_index]) for mat in abide_data_train])
                _percentile = abide_math.percentile_vec(_pearson, value_percentil)
                pos = np.where(abs(_pearson)>=_percentile)[0]
                features= abide_data_orig.T[pos]
                abide_data=features.T
                
                for classifier in acc_abide_classifiers.keys():
                    print('Running: Fold', format(index), ' Measure: ',measure,' Reduction: ', format(value_percentil), ' Classifier: ', format(classifier))
                    estimator = acc_abide_classifiers[classifier]
                    score = cross_val_score(estimator, abide_data, diagnosis, scoring=scoring_type, cv=[(train_index, test_index)])
                    abide_results['atlas'].append(atlas)
                    abide_results['iter_shuffle_split'].append(index)
                    abide_results['measure'].append(measure)
                    abide_results['classifier'].append(classifier)
                    abide_results['dataset'].append('ABIDE')
                    abide_results['dimensionality'].append(dimensions[atlas])
                    abide_results['scores'].append(score)
                    abide_results['scoring'].append(scoring_type)
                    abide_results['reduction'].append(value_percentil)

abide_final_results = pd.DataFrame(abide_results)
abide_final_results.to_csv('predictions_abide_preprocessed_reduction_percentil.csv')
