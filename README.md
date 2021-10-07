<h1 id="tocheading">Table of Contents</h1>
<div id="toc"></div>


```javascript
%%javascript
$.getScript('https://kmahelona.github.io/ipython_notebook_goodies/ipython_notebook_toc.js')
```


    <IPython.core.display.Javascript object>



```python

```

# Prepare example data and model


```python
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml.feature import SQLTransformer
from pyspark.ml.feature import RFormula
```


```python
import pandas as pd
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.max_rows', 500000)
pd.set_option('display.max_colwidth', None)
pd.set_option("display.max_columns",5000)
```

#### example data


```python
# create dataframe
training = spark.createDataFrame([
    (0,'y', "a b c d e spark", 1.0),
    (1,'y', "b d", 0.0),
    (2, None, "spark f g h", 1.0),
    (3, 'n',"hadoop mapreduce", 0.0)
], ["id",'category', "text", "label"])
```

#### pipeline


```python
#process 'categor' column
category_process=SQLTransformer(statement="""select *, coalesce(category, 'unknown') category_fillNA 
                                            from __THIS__ """)
```


```python
#text_process: a pipeline , process text column
tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="text_vector",numFeatures=16)
text_process=Pipeline(stages=[tokenizer, hashingTF])
```


```python
features_assemble=RFormula(formula="~category_fillNA+text_vector",featuresCol='features',handleInvalid='keep')
```


```python
lr = LogisticRegression(maxIter=5, regParam=0.001)
```


```python
#put together into a pipeline
pipeline = Pipeline(stages=[category_process, text_process,features_assemble, lr])
```

#### create PipelineModel


```python
model = pipeline.fit(training)
```

#### apply the model


```python
training_pred=model.transform(training)
```


```python
# Prepare test documents, which are unlabeled (id,category, text) tuples.
test= spark.createDataFrame([
    (4,'y', "spark i j k"),
    (5,'n', "l m n"),
], ["id",'category', "text"])

# Make predictions on test documents and print columns of interest.
prediction = model.transform(test)
```


```python
prediction.limit(1).toPandas()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>category</th>
      <th>text</th>
      <th>category_fillNA</th>
      <th>words</th>
      <th>text_vector</th>
      <th>features</th>
      <th>rawPrediction</th>
      <th>probability</th>
      <th>prediction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>y</td>
      <td>spark i j k</td>
      <td>y</td>
      <td>[spark, i, j, k]</td>
      <td>(1.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)</td>
      <td>(1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)</td>
      <td>[2.397351922197842, -2.397351922197842]</td>
      <td>[0.9166251513201248, 0.0833748486798753]</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#%run /home/c07520/work/Users/c07520/customfunction/start.ipynb
```


```python
import sys
sys.path.insert(0, '/home/c07520/work/Users/c07520/sparkEXample/create_package/base_spark_ML_utils/')
```

# pipeline_util
Check Pipeline and PipelineModel


```python
import spark_ml_utils.pipeline_util as pu
pu=spark_ml_utils.pipeline_util
```


```python
import spark_ml_utils.pipeline_util
from importlib import reload
reload(spark_ml_utils.pipeline_util)
pu=spark_ml_utils.pipeline_util
```

### getStages():  check Pipeline and PipelineModel
In practice, Pipeline and Pipelne Model could contain many stages. the getStages() function will list all the stages for easy check.


```python
#use native method getStages(), not enough information
pipeline.getStages()
```




    [SQLTransformer_f91302b3bef1,
     Pipeline_e1c89b18d4e8,
     RFormula_9ab00e5fdacc,
     LogisticRegression_90590144e087]




```python
pu.getallstages(pipeline,'pipeline')
```

    This is a Pipeline 





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>estimator</th>
      <th>estimator_name</th>
      <th>inputcol</th>
      <th>outputcol</th>
      <th>other_attr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>pipeline.getStages()[0]</td>
      <td>SQLTransformer</td>
      <td>None</td>
      <td>None</td>
      <td>"statement=\nselect *, coalesce(category, 'unknown') category_fillNA \n                                            from __THIS__ "</td>
    </tr>
    <tr>
      <th>1</th>
      <td>pipeline.getStages()[1].getStages()[0]</td>
      <td>Tokenizer</td>
      <td>text</td>
      <td>words</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pipeline.getStages()[1].getStages()[1]</td>
      <td>HashingTF</td>
      <td>words</td>
      <td>text_vector</td>
      <td>None</td>
    </tr>
    <tr>
      <th>3</th>
      <td>pipeline.getStages()[2]</td>
      <td>RFormula</td>
      <td>None</td>
      <td>features</td>
      <td>number of inputCol in formula: 2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>pipeline.getStages()[3]</td>
      <td>LogisticRegression</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>




```python
#similar for PipelineModel
pu.getallstages(model,'model')
```

    This is a PipelineModel 





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>transformer</th>
      <th>transformer_name</th>
      <th>inputcol</th>
      <th>outputcol</th>
      <th>other_attr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>model.stages[0]</td>
      <td>SQLTransformer</td>
      <td>None</td>
      <td>None</td>
      <td>"statement=\nselect *, coalesce(category, 'unknown') category_fillNA \n                                            from __THIS__ "</td>
    </tr>
    <tr>
      <th>1</th>
      <td>model.stages[1].stages[0]</td>
      <td>Tokenizer</td>
      <td>text</td>
      <td>words</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2</th>
      <td>model.stages[1].stages[1]</td>
      <td>HashingTF</td>
      <td>words</td>
      <td>text_vector</td>
      <td>None</td>
    </tr>
    <tr>
      <th>3</th>
      <td>model.stages[2]</td>
      <td>RFormulaModel</td>
      <td>None</td>
      <td>features</td>
      <td>number of inputCol in formula: 2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>model.stages[3]</td>
      <td>LogisticRegressionModel</td>
      <td>None</td>
      <td>None</td>
      <td>labelCol : label, elasticNetParam : 0.0, regParam : 0.001</td>
    </tr>
  </tbody>
</table>
</div>



#### usage


```python
#check any stage
type(pipeline.getStages()[1].getStages()[1])
```




    pyspark.ml.feature.HashingTF




```python
pipeline.getStages()[1].getStages()[1].getNumFeatures()
```




    16




```python
#find and update the pipeline stages
```


```python
pipeline_update=pipeline.copy()
```


```python
pipeline_update.getStages()[1].getStages()[1].setNumFeatures(256)
```




    HashingTF_8753da0a8eba




```python
pipeline_update.getStages()[1].getStages()[1].getNumFeatures()
```




    256



### get_code(): get the code showing how it is created


```python
pstr=pu.get_code(pipeline,'pipeline2') #pstr is a string , same as the following, containing all the code for creating pipeline
```

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



```python
#run the code 
exec(pstr)
```


```python
#pipeline2 contains same stages as pipeline, although it is flatten.
pu.getallstages(pipeline2,'pipeline2')
```

    This is a Pipeline 





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>estimator</th>
      <th>estimator_name</th>
      <th>inputcol</th>
      <th>outputcol</th>
      <th>other_attr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>pipeline2.getStages()[0]</td>
      <td>SQLTransformer</td>
      <td>None</td>
      <td>None</td>
      <td>"statement=\nselect *, coalesce(category, 'unknown') category_fillNA \n                                            from __THIS__ "</td>
    </tr>
    <tr>
      <th>1</th>
      <td>pipeline2.getStages()[1]</td>
      <td>Tokenizer</td>
      <td>text</td>
      <td>words</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pipeline2.getStages()[2]</td>
      <td>HashingTF</td>
      <td>words</td>
      <td>text_vector</td>
      <td>None</td>
    </tr>
    <tr>
      <th>3</th>
      <td>pipeline2.getStages()[3]</td>
      <td>RFormula</td>
      <td>None</td>
      <td>features</td>
      <td>number of inputCol in formula: 2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>pipeline2.getStages()[4]</td>
      <td>LogisticRegression</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>




```python
#for PipelineModel, getcode() return the code for its corresponding pipeline
_=pu.get_code(model,'pipeline3')
```

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



```python
#it also work for any ML estimator and transformer
_=pu.get_code(pipeline.getStages()[2],'obj')
```

    from pyspark.ml.feature import RFormula
    
    obj=RFormula(featuresCol="features",handleInvalid="keep",formula="~category_fillNA+text_vector")



```python
_=pu.get_code(model.stages[1].stages[1],'obj')
```

    from pyspark.ml.feature import HashingTF
    
    obj=HashingTF(numFeatures=16,outputCol="text_vector",inputCol="words")


### Other function

#### flatenStages()


```python
model.stages
```




    [SQLTransformer_f91302b3bef1,
     PipelineModel_d4c1008880e5,
     RFormula_9ab00e5fdacc,
     LogisticRegressionModel: uid = LogisticRegression_90590144e087, numClasses = 2, numFeatures = 19]




```python
pu.flatenStages(model.stages)
```




    [SQLTransformer_f91302b3bef1,
     Tokenizer_3fa6d50bf10c,
     HashingTF_8753da0a8eba,
     RFormula_9ab00e5fdacc,
     LogisticRegressionModel: uid = LogisticRegression_90590144e087, numClasses = 2, numFeatures = 19]



#### pm_to_p()
convert PipelineModel to Pipeline


```python
pipeline4=pu.pm_to_p(model)
```


```python
pu.getallstages(pipeline4,'pipeline4')
```

    This is a Pipeline 





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>estimator</th>
      <th>estimator_name</th>
      <th>inputcol</th>
      <th>outputcol</th>
      <th>other_attr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>pipeline4.getStages()[0]</td>
      <td>SQLTransformer</td>
      <td>None</td>
      <td>None</td>
      <td>"statement=\nselect *, coalesce(category, 'unknown') category_fillNA \n                                            from __THIS__ "</td>
    </tr>
    <tr>
      <th>1</th>
      <td>pipeline4.getStages()[1]</td>
      <td>Tokenizer</td>
      <td>text</td>
      <td>words</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pipeline4.getStages()[2]</td>
      <td>HashingTF</td>
      <td>words</td>
      <td>text_vector</td>
      <td>None</td>
    </tr>
    <tr>
      <th>3</th>
      <td>pipeline4.getStages()[3]</td>
      <td>RFormula</td>
      <td>None</td>
      <td>features</td>
      <td>number of inputCol in formula: 2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>pipeline4.getStages()[4]</td>
      <td>LogisticRegression</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>




```python
import traceback
def bad_method():
    try:
        sqrt = 0**-1
    except Exception:
        print(traceback.print_exc())

bad_method()

```

    None


    Traceback (most recent call last):
      File "<ipython-input-1-830d9875d0cc>", line 4, in bad_method
        sqrt = 0**-1
    ZeroDivisionError: 0.0 cannot be raised to a negative power

