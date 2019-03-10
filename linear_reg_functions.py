#import all needed libraries
import zip
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
import sklearn
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import statsmodels.stats.api as sms
import statsmodels.api as sm
import statsmodels.formula.api as smf

'''
The following functions take in a dataframe, the continous data columns, and a optional
new dataframe. They then return new columns that has been trandformed--the name of the 
column gives the transformation.
'''

def log_transform(cols, df,new_df=None):
    #logarithmic transformation
    trans_cols = []
    for i in cols:
        name = i+'_log'
        trans_cols.append(name)
        if new_df != None:
            new_df[name] = np.log(df[i])
        else: 
            df[name] = np.log(df[i])
    return trans_cols
    
def cube_root_transform(cols, df,new_df=None):
    #cube root tranformation
    trans_cols = []
    for i in cols:
        name = i+'_cuberoot'
        trans_cols.append(name)
        if new_df != None:
            new_df[name] = df[i].apply(lambda x: x**(1/3))
        else: 
            df[name] = df[i].apply(lambda x: x**(1/3))
    return trans_cols

def sqrt_trandform(cols, df,new_df=None):
    #square root transformation
    trans_cols = []
    for i in cols:
        name = i+'_sqrt'
        trans_cols.append(name)
        if new_df != None:
            new_df[name] = df[i].apply(lambda x: x**(1/2))
        else: 
            df[name] = df[i].apply(lambda x: x**(1/2))
    return trans_cols

def sqre_trandform(cols, df,new_df=None):
    '''
    sqaure transformation
    '''
    trans_cols = []
    for i in cols:
        name = i+'_squared'
        trans_cols.append(name)
        if new_df != None:
            new_df[name] = df[i].apply(lambda x: x**2)
        else: 
            df[name] = df[i].apply(lambda x: x**2)
    return trans_cols

def recip_trandform(cols, df,new_df=None):
    #reciprocal transformation
    trans_cols = []
    for i in cols:
        name = i+'_recip'
        trans_cols.append(name)
        if new_df != None:
            new_df[name] = df[i].apply(lambda x: 1/x)
        else: 
            df[name] = df[i].apply(lambda x: 1/x)
    return trans_cols

def make_subplots_scedasticity(numr,numc,dims,target,predictors,df,ylab,xlab,filename=None):
    fig, axes = plt.subplots(nrows=numr, ncols=numc, figsize=dims)
    for i in range(numr):
        for j in range(numc):
            k = sum([3 if i == 1 else 0])
            n = (k+j) 
            f = target+'~'+predictors[n]
            mod = smf.ols(formula=f, data=df).fit()
            pred_tots = mod.predict(df[predictors[n]])
            resid_tots = mod.resid
            sns.regplot(pred_tots,resid_tots,fit_reg=True,scatter_kws={'alpha': 0.5},ax=axes[i][j])
            ax=axes[i][j] 
            ax.set_xlabel(ylab)
            ax.set_ylabel(xlab)
            ax.set_title(predictors[n].replace('_',' ')+' '+'Test for Homoscedasticity')
            fig.tight_layout()
    if filename == None:
        return 
    else: 
        plt.savefig(filename)
        return

def QQ_plot_fun(dims,model,filename):
    fig = plt.figure(figsize=dims)
    fig = sm.graphics.qqplot(model.resid, dist=stats.norm, line='45', fit=True)
    plt.savefig(filename);
    return 

def model_and_model_summary(target,list_of_predictors,df):
    f = target+'~'+ '+'.join(list_of_predictors)
    model = smf.ols(formula=f, data=df).fit()
    return model,model.summary

def combination_r_squared(combos,target,df):
    '''
    This takes in a list of possible combinations and returns a dict
    with the index of the combo and its adjusted r-squared value
    '''
    r_squared = []
    ind = []
    for index,vals in enumerate(combos):
        f = target+'~'+ '+'.join(vals)
        model = smf.ols(formula=f, data=df).fit()
        r_squared.append(model.rsquared_adj)
        ind.append(index)

    rsquar_dict = dict(zip(ind,r_squared))
    return rsquar_dict

