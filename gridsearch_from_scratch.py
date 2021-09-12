
################################Author: Babatunde John Olanipekun#################################################################################
'''
My approach for this exercise will be to create a dictionary of the parameters of interest
use the combination method in itertools module to create a tuple of all possible combinations,
then loop through each of the combination to apply the .fit(), .predict()
and obtain respective accuracy scores.
Each checkpoint is appended to a list which I use to create a dataframe of prediction history for the grid search
'''

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score # other metrics too pls!
from sklearn.ensemble import RandomForestClassifier # more!
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import itertools




#import pdb; pdb.set_trace()


# adapt this code below to run your analysis
# 1. Write a function to take a list or dictionary of clfs and hypers(i.e. use logistic regression), each with 3 different sets of hyper parameters for each
# 2. Expand to include larger number of classifiers and hyperparameter settings
# 3. Find some simple data
# 4. generate matplotlib plots that will assist in identifying the optimal clf and parampters settings
# 5. Please set up your code to be run and save the results to the directory that its executed from
# 6. Investigate grid search function

# M = np.array([[1,2],[3,4],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5]])

# #L = np.ones(M.shape[0])
# mn = [np.ones(int(M.shape[0]*0.5)), np.zeros(int(M.shape[0]*0.5))]
# L = np.ravel(mn)
# n_folds = 5

# data = (M, L, n_folds)


#my plan will be to have a tuple that each contains classifier and a dictionary of parameters to parse



###################################The Grid Search Function#########################################################################################################
def Grid_search(a_clf, data, hyper_p = {}): #, n_estimators, max_depth, class_weight):    
    assert a_clf in [RandomForestClassifier,SVC, LogisticRegression, GradientBoostingClassifier], "Sorry, I'm only good for RandomForestClassifier, SVC or LogisticRegression]"
    a_clf in [RandomForestClassifier,SVC, LogisticRegression, GradientBoostingClassifier]

    if a_clf==RandomForestClassifier:
        a_clf=RandomForestClassifier()
    elif a_clf==SVC:
        a_clf=SVC()
    elif a_clf==GradientBoostingClassifier:
        a_clf=GradientBoostingClassifier()
    else:
        a_clf=LogisticRegression()

    M, L, n_folds = data # unpack data container
    kf = KFold(n_splits=n_folds) # Establish the cross validation
    clf_list=[]
    train_index_list = []
    test_index_list =[]    
    accuracy_score_list=[]
    ids_list=[]

#We need to make a combination of the parameter arguments to parse into the estimator objects.
    params_comb_=[dict(zip(hyper_p,x)) for x in itertools.product(*hyper_p.values())] #the * will unpack a dictionary

    for idx, params in enumerate(params_comb_):
        print('working on===>', idx, params) #Progress check
        for ids, (train_index, test_index) in enumerate(kf.split(M, L)):
            print('dataset===>', ids) #progress check
            #the set_params(**params) allows us utilize a dictionary style to insert hyperparameters
            clf = a_clf.set_params(**params)
            clf.fit(M[train_index], L[train_index])
            pred = clf.predict(M[test_index])            
            clf_list.append(clf)
            train_index_list.append(train_index)
            acc_score = accuracy_score(L[test_index], pred)
            accuracy_score_list.append(acc_score)
            ids_list.append(ids)
            test_index_list.append(test_index)
    #the training and prediction histories are ready to booked into a dataframe.
    pred_df = pd.DataFrame(list(zip(ids_list, clf_list, train_index_list,test_index_list,accuracy_score_list)),
    columns=['ids', 'clf', 'train_index', 'test_index', 'accuracy_score'])
    best_parameter = pred_df.loc[pred_df['accuracy_score'].idxmax()]['clf']
    #################Below are optional outputs###################
    
    #best_parameter = best_param['clf']

    # print('I have ', len(pred_df), '  rows: Grid_search')
    # print('Best parameters are====> ', best_parameter)
    # print('......................head...................................................')
    # print(pred_df.head(n=10))
    # print('......................tail...................................................')
    # print(pred_df.tail(n=10))

    return pred_df, best_parameter


#######################################Function calls here##############################################################################################

'''
Here we test the custom grid search on the digits dataset loaded from sci-kit learn:
The task is to predict the digit that an image represents:
more information at https://scikit-learn.org/stable/tutorial/basic/tutorial.html
'''



from sklearn import datasets
digits = datasets.load_digits()
M_ = digits.data
L_ = digits.target
n_folds = 5

data_ = (M_, L_, n_folds)
params_ = [
    (SVC, {'C':[0.1,1,10], 'gamma':['scale', 'auto'], 'kernel':['linear', 'rbf', 'poly', 'sigmoid']}),
    (LogisticRegression, {'max_iter': [100,500,1000], 'C': [1e-2,1,10], 'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}),
    (RandomForestClassifier, {'n_estimators':list(range(50, 500, 100)), 'max_depth': [3,4], 'class_weight':['balanced', 'balanced_subsample']}), 
    (GradientBoostingClassifier, {'learning_rate':np.logspace(-4,0, num=5), 'n_estimators': [50,100, 500]})
]



############################Accummulate the best parameters########

best_parameters=[]
counter=0
for val in params_:
    counter+=1
    df, best_params_=Grid_search(val[0], data=data_, hyper_p=val[1])
    df.boxplot(column=['accuracy_score'], by=['clf'])
    best_parameters.append(best_params_)
    df.to_csv('digits_class_new_{}.csv'.format(counter))
print('Here are the best parameters: \n', best_parameters)

    



