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
import abide_functions

# reduction' methods
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis
from sklearn.manifold import MDS
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.pipeline import make_pipeline

# Path to data directory

# Variables' atlas, dimensions and measures
timeseries_dir = './abide/Outputs/cpac/filt_noglobal'

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
            print('Reductions')
            print('----------')
            
            pca = make_pipeline(PCA(n_components=train_index.shape[0]))
            pca2 = make_pipeline(PCA(n_components=int(train_index.shape[0]/2)))
            pca3 = make_pipeline(PCA(n_components=int(train_index.shape[0]/3)))
            fa = make_pipeline(FactorAnalysis(n_components=train_index.shape[0]))
            fa2 = make_pipeline(FactorAnalysis(n_components=int(train_index.shape[0]/2)))
            fa3 = make_pipeline(FactorAnalysis(n_components=int(train_index.shape[0]/3)))
            fa1000 = make_pipeline(FactorAnalysis(n_components=1000))
            fa2000 = make_pipeline(FactorAnalysis(n_components=2000))
            lle = make_pipeline(LocallyLinearEmbedding(n_components=train_index.shape[0]-1))
            lle2 = make_pipeline(LocallyLinearEmbedding(n_components=int(train_index.shape[0]/2)))
            lle3 = make_pipeline(LocallyLinearEmbedding(n_components=int(train_index.shape[0]/3)))

            mds = make_pipeline(MDS(n_components=train_index.shape[0]))
            mds2 = make_pipeline(MDS(n_components=int(train_index.shape[0]/2)))
            mds3 = make_pipeline(MDS(n_components=int(train_index.shape[0]/3)))
            mds1000 = make_pipeline(MDS(n_components=1000))
            mds2000 = make_pipeline(MDS(n_components=2000))
    
            dim_reduction_methods_fit_and_transform = [('pca', pca), ('pca2', pca2), ('pca3', pca3),
                                                       ('fa', fa), ('fa2', fa2), ('fa3', fa3), 
                                                       ('fa1000', fa1000), ('fa2000', fa2000),
                                                       ('lle', lle), ('lle2', lle2), ('lle3', lle3)]

            dim_reduction_methods_fit_transform = [('mds', mds), ('mds2', mds2), ('mds3', mds3), 
                                                   ('mds1000', mds1000), ('mds2000', mds2000)]

            for indice, (name, model) in enumerate(dim_reduction_methods_fit_and_transform):
                print('Model: ', indice, name)
                X_train = model.fit_transform(abide_data_orig[train_index])
                X_test = model.transform(abide_data_orig[test_index])
                abide_data = np.zeros((abide_data_orig.shape[0],X_train.shape[1])) 
                abide_data[train_index] = X_train
                abide_data[test_index] = X_test
                
                for classifier in acc_abide_classifiers.keys():
                    print('Running: Fold', format(index), ' Measure: ',measure,' Reduction: ', format(name), ' Classifier: ', format(classifier))
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
                    abide_results['reduction'].append(name)


            for indice, (name, model) in enumerate(dim_reduction_methods_fit_transform):
                print('Model: ', indice, name)
                X_train = model.fit_transform(abide_data_orig[train_index])
                X_test = model.fit_transform(abide_data_orig[test_index])
                abide_data = np.zeros((abide_data_orig.shape[0],X_train.shape[1])) 
                abide_data[train_index] = X_train
                abide_data[test_index] = X_test
                
                for classifier in acc_abide_classifiers.keys():
                    print('Running: Fold', format(index), ' Measure: ',measure,' Reduction: ', format(name), ' Classifier: ', format(classifier))
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
                    abide_results['reduction'].append(name)

abide_final_results = pd.DataFrame(abide_results)
abide_final_results.to_csv('predictions_abide_preprocessed_reduction_others.csv')
