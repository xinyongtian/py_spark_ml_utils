<div class="cell code">

``` python
```

</div>

<div class="cell markdown">

# Prepare example data and model

</div>

<div class="cell code" data-execution_count="3">

``` python
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml.feature import SQLTransformer
from pyspark.ml.feature import RFormula
```

</div>

<div class="cell code" data-execution_count="22">

``` python
import pandas as pd
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.max_rows', 500000)
pd.set_option('display.max_colwidth', None)
pd.set_option("display.max_columns",5000)
```

</div>

<div class="cell markdown">

#### example data

</div>

<div class="cell code" data-execution_count="4">

``` python
# create dataframe
training = spark.createDataFrame([
    (0,'y', "a b c d e spark", 1.0),
    (1,'y', "b d", 0.0),
    (2, None, "spark f g h", 1.0),
    (3, 'n',"hadoop mapreduce", 0.0)
], ["id",'category', "text", "label"])
```

</div>

<div class="cell markdown">

#### pipeline

</div>

<div class="cell code" data-execution_count="5">

``` python
#process 'categor' column
category_process=SQLTransformer(statement="""select *, coalesce(category, 'unknown') category_fillNA 
                                            from __THIS__ """)
```

</div>

<div class="cell code" data-execution_count="6">

``` python
#text_process: a pipeline , process text column
tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="text_vector",numFeatures=16)
text_process=Pipeline(stages=[tokenizer, hashingTF])
```

</div>

<div class="cell code" data-execution_count="7">

``` python
features_assemble=RFormula(formula="~category_fillNA+text_vector",featuresCol='features',handleInvalid='keep')
```

</div>

<div class="cell code" data-execution_count="8">

``` python
lr = LogisticRegression(maxIter=5, regParam=0.001)
```

</div>

<div class="cell code" data-execution_count="9">

``` python
#put together into a pipeline
pipeline = Pipeline(stages=[category_process, text_process,features_assemble, lr])
```

</div>

<div class="cell markdown">

#### create PipelineModel

</div>

<div class="cell code" data-execution_count="10">

``` python
model = pipeline.fit(training)
```

</div>

<div class="cell markdown">

#### apply the model

</div>

<div class="cell code" data-execution_count="24">

``` python
training_pred=model.transform(training)
```

</div>

<div class="cell code" data-execution_count="25">

``` python
# Prepare test documents, which are unlabeled (id,category, text) tuples.
test= spark.createDataFrame([
    (4,'y', "spark i j k"),
    (5,'n', "l m n"),
], ["id",'category', "text"])

# Make predictions on test documents and print columns of interest.
prediction = model.transform(test)
```

</div>

<div class="cell code" data-execution_count="26">

``` python
prediction.limit(1).toPandas()
```

<div class="output execute_result" data-execution_count="26">

``` 
   id category         text category_fillNA             words  \
0   4        y  spark i j k               y  [spark, i, j, k]   

                                                                        text_vector  \
0  (1.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)   

                                                                                          features  \
0  (1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)   

                             rawPrediction  \
0  [2.397351922197842, -2.397351922197842]   

                                probability  prediction  
0  [0.9166251513201248, 0.0833748486798753]         0.0  
```

</div>

</div>

<div class="cell code" data-execution_count="12">

``` python
#%run /home/c07520/work/Users/c07520/customfunction/start.ipynb
```

</div>

<div class="cell code" data-execution_count="15">

``` python
import sys
sys.path.insert(0, '/home/c07520/work/Users/c07520/sparkEXample/create_package/base_spark_ML_utils/')
```

</div>

<div class="cell markdown">

# pipeline\_util

Check Pipeline and PipelineModel

</div>

<div class="cell code" data-execution_count="51">

``` python
import spark_ml_utils.pipeline_util as pu
pu=spark_ml_utils.pipeline_util
```

</div>

<div class="cell code" data-execution_count="52">

``` python
import spark_ml_utils.pipeline_util
from importlib import reload
reload(spark_ml_utils.pipeline_util)
pu=spark_ml_utils.pipeline_util
```

</div>

<div class="cell markdown">

### getStages(): check Pipeline and PipelineModel

In practice, Pipeline and Pipelne Model could contain many stages. the
getStages() function will list all the stages for easy check.

</div>

<div class="cell code" data-execution_count="13">

``` python
#use native method getStages(), not enough information
pipeline.getStages()
```

<div class="output execute_result" data-execution_count="13">

    [SQLTransformer_f91302b3bef1,
     Pipeline_e1c89b18d4e8,
     RFormula_9ab00e5fdacc,
     LogisticRegression_90590144e087]

</div>

</div>

<div class="cell code" data-execution_count="27">

``` python
pu.getallstages(pipeline,'pipeline')
```

<div class="output stream stdout">

``` 
This is a Pipeline 
```

</div>

<div class="output execute_result" data-execution_count="27">

``` 
                                estimator      estimator_name inputcol  \
0                 pipeline.getStages()[0]      SQLTransformer     None   
1  pipeline.getStages()[1].getStages()[0]           Tokenizer     text   
2  pipeline.getStages()[1].getStages()[1]           HashingTF    words   
3                 pipeline.getStages()[2]            RFormula     None   
4                 pipeline.getStages()[3]  LogisticRegression     None   

     outputcol  \
0         None   
1        words   
2  text_vector   
3     features   
4         None   

                                                                                                                           other_attr  
0  "statement=\nselect *, coalesce(category, 'unknown') category_fillNA \n                                            from __THIS__ "  
1                                                                                                                                None  
2                                                                                                                                None  
3                                                                                                    number of inputCol in formula: 2  
4                                                                                                                                None  
```

</div>

</div>

<div class="cell code" data-execution_count="28">

``` python
#similar for PipelineModel
pu.getallstages(model,'model')
```

<div class="output stream stdout">

``` 
This is a PipelineModel 
```

</div>

<div class="output execute_result" data-execution_count="28">

``` 
                 transformer         transformer_name inputcol    outputcol  \
0            model.stages[0]           SQLTransformer     None         None   
1  model.stages[1].stages[0]                Tokenizer     text        words   
2  model.stages[1].stages[1]                HashingTF    words  text_vector   
3            model.stages[2]            RFormulaModel     None     features   
4            model.stages[3]  LogisticRegressionModel     None         None   

                                                                                                                           other_attr  
0  "statement=\nselect *, coalesce(category, 'unknown') category_fillNA \n                                            from __THIS__ "  
1                                                                                                                                None  
2                                                                                                                                None  
3                                                                                                    number of inputCol in formula: 2  
4                                                                           labelCol : label, elasticNetParam : 0.0, regParam : 0.001  
```

</div>

</div>

<div class="cell markdown">

#### usage

</div>

<div class="cell code" data-execution_count="36">

``` python
#check any stage
type(pipeline.getStages()[1].getStages()[1])
```

<div class="output execute_result" data-execution_count="36">

    pyspark.ml.feature.HashingTF

</div>

</div>

<div class="cell code" data-execution_count="41">

``` python
pipeline.getStages()[1].getStages()[1].getNumFeatures()
```

<div class="output execute_result" data-execution_count="41">

``` 
16
```

</div>

</div>

<div class="cell code">

``` python
#find and update the pipeline stages
```

</div>

<div class="cell code" data-execution_count="38">

``` python
pipeline_update=pipeline.copy()
```

</div>

<div class="cell code" data-execution_count="39">

``` python
pipeline_update.getStages()[1].getStages()[1].setNumFeatures(256)
```

<div class="output execute_result" data-execution_count="39">

    HashingTF_8753da0a8eba

</div>

</div>

<div class="cell code" data-execution_count="42">

``` python
pipeline_update.getStages()[1].getStages()[1].getNumFeatures()
```

<div class="output execute_result" data-execution_count="42">

    256

</div>

</div>

<div class="cell markdown">

### get\_code(): get the code showing how it is created

</div>

<div class="cell code" data-execution_count="31">

``` python
pstr=pu.get_code(pipeline,'pipeline2') #pstr is a string , same as the following, containing all the code for creating pipeline
```

<div class="output stream stdout">

    from pyspark.ml import Pipeline
    from pyspark.ml.feature import SQLTransformer
    from pyspark.ml.feature import Tokenizer
    from pyspark.ml.feature import HashingTF
    from pyspark.ml.feature import RFormula
    from pyspark.ml.classification import LogisticRegression
    
    pipeline2=Pipeline(stages=[
    ########################################stage0
    SQLTransformer(statement="""select *, coalesce(category, 'unknown') category_fillNA 
                                                from __THIS__ """)
    
    ,########################################stage1
    Tokenizer(outputCol="words",inputCol="text")
    
    ,########################################stage2
    HashingTF(numFeatures=16,outputCol="text_vector",inputCol="words")
    
    ,########################################stage3
    RFormula(featuresCol="features",handleInvalid="keep",formula="~category_fillNA+text_vector")
    
    ,########################################stage4
    LogisticRegression(maxIter=5,regParam=0.001)
    ])

</div>

</div>

<div class="cell code" data-execution_count="32">

``` python
#run the code 
exec(pstr)
```

</div>

<div class="cell code" data-execution_count="34">

``` python
#pipeline2 contains same stages as pipeline, although it is flatten.
pu.getallstages(pipeline2,'pipeline2')
```

<div class="output stream stdout">

``` 
This is a Pipeline 
```

</div>

<div class="output execute_result" data-execution_count="34">

``` 
                  estimator      estimator_name inputcol    outputcol  \
0  pipeline2.getStages()[0]      SQLTransformer     None         None   
1  pipeline2.getStages()[1]           Tokenizer     text        words   
2  pipeline2.getStages()[2]           HashingTF    words  text_vector   
3  pipeline2.getStages()[3]            RFormula     None     features   
4  pipeline2.getStages()[4]  LogisticRegression     None         None   

                                                                                                                           other_attr  
0  "statement=\nselect *, coalesce(category, 'unknown') category_fillNA \n                                            from __THIS__ "  
1                                                                                                                                None  
2                                                                                                                                None  
3                                                                                                    number of inputCol in formula: 2  
4                                                                                                                                None  
```

</div>

</div>

<div class="cell code" data-execution_count="35" data-scrolled="true">

``` python
#for PipelineModel, getcode() return the code for its corresponding pipeline
_=pu.get_code(model,'pipeline3')
```

<div class="output stream stdout">

    from pyspark.ml import Pipeline
    from pyspark.ml.feature import SQLTransformer
    from pyspark.ml.feature import Tokenizer
    from pyspark.ml.feature import HashingTF
    from pyspark.ml.feature import RFormula
    from pyspark.ml.classification import LogisticRegression
    
    pipeline3=Pipeline(stages=[
    ########################################stage0
    SQLTransformer(statement="""select *, coalesce(category, 'unknown') category_fillNA 
                                                from __THIS__ """)
    
    ,########################################stage1
    Tokenizer(outputCol="words",inputCol="text")
    
    ,########################################stage2
    HashingTF(numFeatures=16,outputCol="text_vector",inputCol="words")
    
    ,########################################stage3
    RFormula(featuresCol="features",handleInvalid="keep",formula="~category_fillNA+text_vector")
    
    ,########################################stage4
    LogisticRegression(maxIter=5,regParam=0.001)
    ])

</div>

</div>

<div class="cell code" data-execution_count="44">

``` python
#it also work for any ML estimator and transformer
_=pu.get_code(pipeline.getStages()[2],'obj')
```

<div class="output stream stdout">

    from pyspark.ml.feature import RFormula
    
    obj=RFormula(featuresCol="features",handleInvalid="keep",formula="~category_fillNA+text_vector")

</div>

</div>

<div class="cell code" data-execution_count="45">

``` python
_=pu.get_code(model.stages[1].stages[1],'obj')
```

<div class="output stream stdout">

    from pyspark.ml.feature import HashingTF
    
    obj=HashingTF(numFeatures=16,outputCol="text_vector",inputCol="words")

</div>

</div>

<div class="cell markdown">

### Other function

</div>

<div class="cell markdown">

#### flatenStages()

</div>

<div class="cell code" data-execution_count="48">

``` python
model.stages
```

<div class="output execute_result" data-execution_count="48">

    [SQLTransformer_f91302b3bef1,
     PipelineModel_d4c1008880e5,
     RFormula_9ab00e5fdacc,
     LogisticRegressionModel: uid = LogisticRegression_90590144e087, numClasses = 2, numFeatures = 19]

</div>

</div>

<div class="cell code" data-execution_count="49">

``` python
pu.flatenStages(model.stages)
```

<div class="output execute_result" data-execution_count="49">

    [SQLTransformer_f91302b3bef1,
     Tokenizer_3fa6d50bf10c,
     HashingTF_8753da0a8eba,
     RFormula_9ab00e5fdacc,
     LogisticRegressionModel: uid = LogisticRegression_90590144e087, numClasses = 2, numFeatures = 19]

</div>

</div>

<div class="cell markdown">

#### pm\_to\_p()

convert PipelineModel to Pipeline

</div>

<div class="cell code" data-execution_count="46">

``` python
pipeline4=pu.pm_to_p(model)
```

</div>

<div class="cell code" data-execution_count="47">

``` python
pu.getallstages(pipeline4,'pipeline4')
```

<div class="output stream stdout">

``` 
This is a Pipeline 
```

</div>

<div class="output execute_result" data-execution_count="47">

``` 
                  estimator      estimator_name inputcol    outputcol  \
0  pipeline4.getStages()[0]      SQLTransformer     None         None   
1  pipeline4.getStages()[1]           Tokenizer     text        words   
2  pipeline4.getStages()[2]           HashingTF    words  text_vector   
3  pipeline4.getStages()[3]            RFormula     None     features   
4  pipeline4.getStages()[4]  LogisticRegression     None         None   

                                                                                                                           other_attr  
0  "statement=\nselect *, coalesce(category, 'unknown') category_fillNA \n                                            from __THIS__ "  
1                                                                                                                                None  
2                                                                                                                                None  
3                                                                                                    number of inputCol in formula: 2  
4                                                                                                                                None  
```

</div>

</div>

<div class="cell code" data-execution_count="1">

``` python
import traceback
def bad_method():
    try:
        sqrt = 0**-1
    except Exception:
        print(traceback.print_exc())

bad_method()
```

<div class="output stream stdout">

    None

</div>

<div class="output stream stderr">

    Traceback (most recent call last):
      File "<ipython-input-1-830d9875d0cc>", line 4, in bad_method
        sqrt = 0**-1
    ZeroDivisionError: 0.0 cannot be raised to a negative power

</div>

</div>
