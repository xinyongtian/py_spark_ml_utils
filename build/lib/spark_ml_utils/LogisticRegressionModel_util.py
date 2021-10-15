import pandas as pd
def extract_feature_name(dataframe,featuresVector,stat=False):
    """
    dataframe: spark dataframe,  must contain the featuresVector column
    featuresVector,string name of the features vector
    stat: boolean, if True, return mean and sd
    return a pandas dataframe, include columns
    """
    from itertools import chain
    try:
        attrs = sorted(
            (attr["idx"], attr["name"]) for attr in (chain(*dataframe
                .schema[featuresVector]
                .metadata["ml_attr"]["attrs"].values()))) 
    except Exception as ex:
        print(f""""error occur in extract_feature_name(), it looks like column {featuresVector} does not have such metadata.
        error type:  {type(ex).__name__} , error message: {ex.args}
        error handled by   fill the feature name with index""")
        vectorsize=dataframe.rdd.collect()[0][featuresVector].size
        attrs=zip(list(range(vectorsize)),list(range(vectorsize)))
    df=pd.DataFrame(attrs,columns=['feature_index','feature_name'])
    if stat:
        from pyspark.ml.feature import StandardScaler
        standardScaler = StandardScaler(inputCol=featuresVector, outputCol="scaled")
        smodel = standardScaler.fit(dataframe)
        df['N']=dataframe.count()
        df['mean']=smodel.mean.toArray()
        df['std']=smodel.std.toArray()
    return df
def feature_importance(lrm_model,trainDF,trainFeatures,nonzero_only=True):
    """
    lrm_model:an instance of pyspark.ml.classification.LogisticRegressionModel  
    trainDF: spark DataFrame, including the training features column
    trainFeatures: a string, training features name
    nonzero_only: boolean, if True, only return variable with non zero coefficient
    return a pandas dataframe with following columns:
    feature_index: index in trainFeatures
    feature_name: variable name inside trainFeatures
    coef: coefficient from lrm_model
    mean: mean of the variables inside trainFeatures
    std: standard deviation of the variables inside trainFeatures
    std_coef: standardized coefficient =coef*std
    feature_importance: absolute value of std
    example:
    df=feature_importance(lrm_model=model.stages[3], trainDF=training_pred, trainFeatures='features',nonzero_only=False)
    df.head(1)
    |   | feature_index | feature_name  | coef | mean | std | std_coef  | feature_importance |
    |---|---------------|---------------|------|------|-----|-----------|--------------------|
    | 0 | 4             | text_vector_2 | 3.059| 0.50 | 0.57| 1.766666  |     1.766666       |
    """
    coef=extract_feature_name(trainDF,trainFeatures,stat=True)
    coef['coef']=lrm_model.coefficients.toArray()
    coef["std_coef"]=coef["coef"]*coef["std"]
    coef["feature_importance"]=coef.std_coef.abs() 
    if nonzero_only:
        coef=coef.loc[coef.coef!=0,:]    
    coef.sort_values(by=["feature_importance"],ascending=False,inplace=True)
    coef.reset_index(drop=True,inplace=True)
    coef=coef[['feature_index', 'feature_name','coef',  'mean', 'std',  'std_coef', 'feature_importance']]
    return coef