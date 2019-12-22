"""
Estimators
Base article: Benchmarking functional connectome-based predictive models for resting-state fMRI
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Lasso, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model

feature_selection = SelectPercentile(f_classif, percentile=10)
svc_l1 = LinearSVC(penalty='l1', dual=False, random_state=0)
anova_svcl1 = Pipeline([('anova', feature_selection), ('svc', svc_l1)])
svc_l2 = LinearSVC(penalty='l2', random_state=0)
anova_svcl2 = Pipeline([('anova', feature_selection), ('svc', svc_l2)])
gnb = GaussianNB()
randomf = RandomForestClassifier(random_state=0)
logregression_l1 = LogisticRegression(penalty='l1', dual=False, random_state=0)
logregression_l2 = LogisticRegression(penalty='l2', random_state=0)
lasso = Lasso(random_state=0)
knn = KNeighborsClassifier(n_neighbors=1)
ridge = RidgeClassifier()
netn5 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,), random_state=1)
netn5a = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(5,), random_state=1)
netn10 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10,), random_state=1)
netn10a = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(10,), random_state=1)
ridgebay = linear_model.BayesianRidge()

# For Accuracy
acc_abide_classifiers = {'GaussianNB': gnb,
                       'RandomF': randomf,
                       'logistic_l1': logregression_l1,
                       'logistic_l2': logregression_l2,
                       'anova_svcl1': anova_svcl1,
                       'anova_svcl2': anova_svcl2,
                       'svc_l1': svc_l1,
                       'svc_l2': svc_l2,
                       'ridge': ridge,
                       'neural5': netn5,
                       'neural5a': netn5a,
                       'neural10': netn10,
                       'neural10a': netn10a,
                       'knn': knn}

# For AUC
auc_abide_classifiers = {'GaussianNB': gnb,
                       'RandomF': randomf,
                       'logistic_l1': logregression_l1,
                       'logistic_l2': logregression_l2,
                       'lasso': lasso,
                       'anova_svcl1': anova_svcl1,
                       'anova_svcl2': anova_svcl2,
                       'svc_l1': svc_l1,
                       'svc_l2': svc_l2,
                       'ridge': ridge,
                       'neural5': netn5,
                       'neural5a': netn5a,
                       'neural10': netn10,
                       'neural10a': netn10a,
                       'ridgebayesian': ridgebay,
                       'knn': knn}
