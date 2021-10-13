#original function take variable  's name string as input which is eval() in the function . but it won't work in in module as the variable does 
#not exist in module globals
#to deal with it
# 1.rename all function toke variable name as inpput, by add '_'
#2 def new function which take variable as input
    #def pm_to_p(obj,obj_name='TheInput'): #obj_name='TheInput' no need in the output, it can be put inside the function 
    #    globals()[obj_name]=obj   
    #    return _pm_to_p(obj_name)
    #def _pm_to_p(pmstr): #orignal function
#it might work with a decorator, but the complication is that, the origal function still need be called
    
    
import re
import pandas as pd
import copy
#give a stages of a PiplineModel of PiplineModels , return flatten list of stages
#2021_05_14 add pipeline
def flatenStages(stg):
    """give a list stages of a Pipeline or PipelineModel , return a flatten list of stages
    """
    output=[]
    for s in stg:
        if str(type(s))=="<class 'pyspark.ml.pipeline.PipelineModel'>":
            output.extend(flatenStages(s.stages))
        elif str(type(s))=="<class 'pyspark.ml.pipeline.Pipeline'>":
            output.extend(flatenStages(s.getStages()))
        else:
            output.append(s)
    return output
def getallstages_p(pipeline,pipeline_name='TheInput'):
    """input
    pipeline: a Pipeline
    pipeline_name: a string, name of the pipeline
    return a pandas dataframe , include all the stages and important parameter
    """
    globals()[pipeline_name]=pipeline   
    return _getallstages_p(pipeline_name)
def _getallstages_p(pstr):
    """pstr: a string,  pipeline  name 
       return a pandas Dataframe: of all leaf stages of transformer.
       to print return in a cell , use print_return(df)
       dependency: dependent on _getallstages_pm , as pipeline can contain pipeline model, eg pipelineModel of sqltransformer
    """
    #import pdb; pdb.set_trace()
    p=eval(pstr)
    output=[]
    for i,s in enumerate(p.getStages()):
        if str(type(s))=="<class 'pyspark.ml.pipeline.Pipeline'>":
            pstr2=f"{pstr}.getStages()[{i}]"
            output.append(_getallstages_p(pstr2))
        elif str(type(s))=="<class 'pyspark.ml.pipeline.PipelineModel'>": #different from pm, pipeline can have pipelinemodel inside
            pstr2=f"{pstr}.getStages()[{i}]"
            df2=_getallstages_pm(pstr2)
            df2.columns = ['estimator','estimator_name','inputcol','outputcol','other_parameters']
            output.append(df2) 
        else:
            tn=re.sub(r"^.*\.(\w+)\b.*",r"\1",str(type(s)))
            pstr2=f"{pstr}.getStages()[{i}]"
            temp=pd.DataFrame([[pstr2,tn,None,None,None]],columns=['estimator','estimator_name','inputcol','outputcol','other_parameters'])
            if temp.estimator_name.iloc[0]=="SQLTransformer":
                st='"statement=\n'+re.sub('\t','     ',eval(pstr2).getStatement())+'"' 
                if len(st)>=32767: #if lenth exceed 32767 keep 20000
                    idx1=st.rfind('\n',0,10000)
                    idx2=st.find('\n',len(st)-10000,len(st))
                    newst=st[:idx1]+"\n\n..........\n"+st[idx2:]
                    st=newst.replace("statement=","TRUNCATED !!!\n\nstatement=")
                temp["other_parameters"]=st
            elif temp.estimator_name.iloc[0]=="RFormula": 
                temp["outputcol"]=[value for key, value in eval(pstr2).extractParamMap().items() if key.name=='featuresCol']            
                form="formular: "+[value for key, value in eval(pstr2).extractParamMap().items() if key.name=='formula'][0]
                temp["other_parameters"]=f"number of inputCol in formula: {form.count('+')+1}"
            else: #2021_05_19 _getallstages_p(),change from "temp.estimator_name.iloc[0]=="CountVectorizer":"  so it can deal with any transformer
                #temp["inputcol"]=[value for key, value in eval(pstr2).extractParamMap().items() if key.name=='inputCol']
                #temp["outputcol"]=[value for key, value in eval(pstr2).extractParamMap().items() if key.name=='outputCol']
                #2021_07_27 add if from obove .or when inputcol is [] will throw error
                ip=[value for key, value in eval(pstr2).extractParamMap().items() if key.name=='inputCol']
                if len(ip)>0:
                    temp["inputcol"]=ip
                op=[value for key, value in eval(pstr2).extractParamMap().items() if key.name=='outputCol']
                if len(op)>0:
                    temp["outputcol"]=op                
            output.append(temp)
    outputdf=pd.concat(output)
    outputdf=outputdf.reset_index(drop=True)
    return outputdf

def getallstages_pm(pipelinemodel,pipelinemodel_name='TheInput'):
    globals()[pipelinemodel_name]=pipelinemodel   
    return _getallstages_pm(pipelinemodel_name)

def _getallstages_pm(pmstr):
    """pmstr: a pipelinemodel name in quote
       return a df: of all leaf stages of transformer.
       to print return in a cell , use print_return(df)
    """
    #import pdb; pdb.set_trace()
    pm=eval(pmstr)
    output=[]
    for i,s in enumerate(pm.stages):
        if str(type(s))=="<class 'pyspark.ml.pipeline.PipelineModel'>":
            pmstr2=f"{pmstr}.stages[{i}]"
            output.append(_getallstages_pm(pmstr2)) 
        else:
            tn=re.sub(r"^.*\.(\w+)\b.*",r"\1",str(type(s)))
            pmstr2=f"{pmstr}.stages[{i}]"
            temp=pd.DataFrame([[pmstr2,tn,None,None,None]],columns=['transformer','transformer_name','inputcol','outputcol','other_parameters'])
            if temp.transformer_name.iloc[0]=="SQLTransformer":
                #import pdb; pdb.set_trace() #####################
                st='"statement=\n'+re.sub('\t','     ',eval(pmstr2).getStatement())+'"' 
                if len(st)>=32767: #if lenth exceed 32767 keep 20000
                    idx1=st.rfind('\n',0,10000)
                    idx2=st.find('\n',len(st)-10000,len(st))
                    newst=st[:idx1]+"\n\n..........\n"+st[idx2:]
                    st=newst.replace("statement=","TRUNCATED !!!\n\nstatement=")
                temp["other_parameters"]=st
            elif temp.transformer_name.iloc[0]=="CountVectorizerModel": 
                temp["other_parameters"]="vocabulary="+str(eval(pmstr2).vocabulary)
            elif temp.transformer_name.iloc[0]=="RFormulaModel": 
                temp["outputcol"]=[value for key, value in eval(pmstr2).extractParamMap().items() if key.name=='featuresCol']            
                form="formular: "+[value for key, value in eval(pmstr2).extractParamMap().items() if key.name=='formula'][0]
                temp["other_parameters"]=f"number of inputCol in formula: {form.count('+')+1}"
            elif temp.transformer_name.iloc[0]=='LogisticRegressionModel': #2019-04-03add elasticNetParam,regParam
                label=[value for key, value in eval(pmstr2).extractParamMap().items() if key.name=='labelCol'][0]
                elasticNetParam=[value for key, value in eval(pmstr2).extractParamMap().items() if key.name=='elasticNetParam'][0]
                regParam=[value for key, value in eval(pmstr2).extractParamMap().items() if key.name=='regParam'][0]
                temp["other_parameters"]=f"labelCol : {label}, elasticNetParam : {elasticNetParam}, regParam : {regParam}"
            else: #2021_05_19 _getallstages_p(),change from "temp.estimator_name.iloc[0]=="CountVectorizer":"  so it can deal with any transformer
                #temp["inputcol"]=[value for key, value in eval(pmstr2).extractParamMap().items() if key.name=='inputCol']
                #temp["outputcol"]=[value for key, value in eval(pmstr2).extractParamMap().items() if key.name=='outputCol']
                #2021_07_27 add if .or when inputcol is [] will throw error
                ip=[value for key, value in eval(pmstr2).extractParamMap().items() if key.name=='inputCol']
                if len(ip)>0:
                    temp["inputcol"]=ip
                op=[value for key, value in eval(pmstr2).extractParamMap().items() if key.name=='outputCol']
                if len(op)>0:
                    temp["outputcol"]=op                
            output.append(temp)
    outputdf=pd.concat(output)
    outputdf=outputdf.reset_index(drop=True)
    return outputdf
def getallstages(obj,obj_name='ThisInput'):
    """obj: a pipeline or pipelineModel
    obj_name: a string, name of the obj
    return a pandas dataframe , list all the stages and some properties
    """
    if type(obj).__name__=='PipelineModel': 
        print(f"This is a PipelineModel ")
        return getallstages_pm(pipelinemodel=obj,pipelinemodel_name=obj_name)        
    elif type(obj).__name__=='Pipeline':
        print(f"This is a Pipeline ")
        return getallstages_p(pipeline=obj,pipeline_name=obj_name) 
    else: 
        print('the input must be a Pipeline or PipelineModel!')
    #2021_10_01 change output , 1.only define none default parameter 2. only use """ when there is \n multi line
def getCode(obj,output_objstr=None,printout=True):
    """
    Given a pyspark.ml object (obj) which could be  a Pipeline,PipelineModel, estimator, transformer, return a string containing the code for 
    re-creating the corresponding Pipeline or estimator
    output_objstr: the name of the object to be re-created
    """
    obj_name='TheInput'
    globals()[obj_name]=obj   
    return _getCode(obj_name,output_objstr=output_objstr,printout=printout)

def _getCode(objstr,output_objstr=None,printout=True):
    """ given a pyspark.ml object name, return its  definition in str, 
    if the object is a fitted Model, return the estimator(or pipeline) definition"""
    #if object is a pipeline or pipelineModel
    def _get_def_strs(tf): #2021_10_01 define a function so can be use twice. 
        #internal function given a tf/estimator, return two strings, import str and parameter string
        tf=tf.copy() #need copy() to avoid change on original pm
        tp=type(tf)
        from1=re.search(r'(pyspark.*)\.',str(tp)).group(1) # from1: example:  import from1 from from2
        from2=re.search(r'(pyspark.*)\.(\w+)',str(tp)).group(2) #from2 : 
        all_imp=f"from {from1} import {from2}"
        #par=[f'{k.name}="""{v}"""' if isinstance(v, str) else f"{k.name}={v}" for k,v in tf.extractParamMap().items() if re.search('current:',tf.explainParam(k))]
        par=[] 
        for k,v in tf.extractParamMap().items(): # 2021_10_01 only use """ when there is \n
            if re.search('current:',tf.explainParam(k)): #2021_10_01 if there is 'current' mean it not default
                if isinstance(v, str):
                    if re.search(r'\n',v):
                        par.append(f'{k.name}="""{v}"""') 
                    else:
                        par.append(f'{k.name}="{v}"')
                else:
                    par.append(f"{k.name}={v}")
                    
        allstags_str2=f"{from2}({','.join(par)})"
        return (all_imp,allstags_str2)
    if re.search(r'Pipeline(Model\W|\W)',str(type(eval(objstr)))): 
        if re.search(r'Pipeline(Model\W)',str(type(eval(objstr)))):

            global pipeline_converted_in_fun_getCode
            pipeline_converted_in_fun_getCode=eval(f"_pm_to_p('{objstr}')")
            objstr=objstr+'_pipeline'
            #import pdb; pdb.set_trace()
            als=_getallstages_p('pipeline_converted_in_fun_getCode')
        else:
            als=_getallstages_p(objstr)
        for i in range(als.shape[0]):
            tf=eval(als.loc[i,'estimator']) 
            als.loc[i,'imp_str'], als.loc[i,'stage_str']=_get_def_strs(tf)


        all_imp='from pyspark.ml import Pipeline\n'+'\n'.join(als.imp_str.unique())
        allstags_str= '\n\n,'.join('#'*40+'stage'+als.index.map(str)+'\n'+als.stage_str) 
        allstags_str2=f"Pipeline(stages=[\n{allstags_str}\n])"
    #if object is not a pipeline or pipelineModel, can be other like classifier, feature Model
    else:
        if re.search(r'Model\W',str(type(eval(objstr)))): # if the objected is a model, convert to the estimator
            tf=eval(f"model_to_estimator({objstr})")
            objstr=objstr+'_estimator'
        else:
            tf=eval(objstr)
        all_imp, allstags_str2=_get_def_strs(tf)
    #return final str
    if  output_objstr is None:
        output_objstr=re.sub(r'\W','_',objstr)+'_def'
    final_str=all_imp+'\n\n'+output_objstr+'='+allstags_str2
    if(printout):
        print(final_str)
    return final_str
def model_to_estimator(model):
    """given a model , return a estimator"""
    tf=model.copy()
    tp=type(tf)
    from1=re.search(r'(pyspark.*)\.',str(tp)).group(1) # from1: string between, import ....from
    from2=re.search(r'(pyspark.*)\.(\w+)',str(tp)).group(2) #from2 : after from

    from2=re.sub(pattern='DecisionTreeRegressionModel', repl='DecisionTreeRegressor', string=from2)
    from2=re.sub(pattern='GBTRegressionModel', repl='GBTRegressor', string=from2)
    from2=re.sub(pattern='RandomForestRegressionModel', repl='RandomForestRegressor', string=from2)
    from2=re.sub(pattern='ClassificationModel$', repl='Classifier', string=from2)
    from2=re.sub(pattern='Model$', repl='', string=from2)

    exec(f"from {from1} import {from2}")
    p=eval(from2+"()") #new estimator 
    tf_parent=tf.params[0].parent
    pp=p.extractParamMap() #a dict of p's params
    tfp=tf.extractParamMap()#dict of tf' params
    #update p ' parameter from tf , if exists. if not keep original

    for k in p.params:
        k2=copy.deepcopy(k)
        k2.parent=tf_parent
        v2=tfp.get(k2,pp.get(k,None)) #if k2 not find use original
        #import pdb; pdb.set_trace()
        if v2 is not None:
            if re.search('current:',tf.explainParam(k2)): #2021_ 10_01 only update  not default (has current value)
                p.set(k,v2)            
    return p

#2021_ 10_01 only update  not default (has current value)

def pm_to_p(pipelinemodel):
    """convert a PipelineModel to Pipeline"""
    pipelinemodel_name='TheInput'
    globals()[pipelinemodel_name]=pipelinemodel   
    return _pm_to_p(pipelinemodel_name)
def _pm_to_p(pmstr):
    """
    Given a pm name in str , return a corresponding pipeline
    dependency: _getallstages_pm()
    """
    als=_getallstages_pm(pmstr)
    for i in range(als.shape[0]):
        #print(i)
        tf=eval(als.loc[i,'transformer']) 
        est=model_to_estimator(tf)
        als.loc[i,'p_stage']=est
    from pyspark.ml import Pipeline
    p2=Pipeline(stages=als.p_stage.tolist())
    return p2



