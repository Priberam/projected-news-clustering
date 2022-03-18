# Priberam Clustering

## Swagger

Go to http://localhost:65065/index.html to see swagger docs.

## Training instructions
The SVM models are a tool that aids the clustering module in two tasks:

- To determine which cluster from the current cluster pool is the most adequate for a given new document (Ranking task);
- To decide if the best cluster (obtained from the previous task) is good enough and the document should merge into that cluster, or if the document should create a new cluster by itself (Merge task).

For the ranking task, the process occurs as follows:
- A document in the stream enters the clustering module and its representation is computed;
- The similarity between the document and each cluster in the cluster pool is evaluated (either through the previously trained SVM-Rank model, or through a main feature such as the cosine similarity)

Let us take the example case where we are registering 4 features, corresponding to the cosine similarity between the document and a cluster, and the other 3 features are the oldest, the newest and the most relevant timestamp.
Each document has an associated qid (which corresponds to an incrementor), and the format of the documents in the generated training dataset for the ranking model is as follows:
```
1 qid:14 1:0.931655 2:0.999919 3:0.999806 4:0.999929 
0 qid:14 1:0.173957 2:0.999988 3:0.999988 4:0.999988 
0 qid:14 1:0.10031 2:0.999365 3:0.997991 4:0.999497 
0 qid:14 1:0.020898 2:0.999957 3:0.999652 4:0.999989 
```
The first integer corresponds to a 1 in the case of a positive sample and 0 in the case of a negative sample. In this case, the gold cluster corresponds to the 1 label, and the other 3 clusters in the pool correspond to the 0 label. For the ranking dataset, only the cases where a document has a gold cluster are registered (that is, if the document is the first document of its cluster, then it is not registered in the dataset).
The qid portion represents the document's qid, and the features are represented in the format n:v, where n is the number of the feature and v is its attributed value.

After obtaining the training dataset, we proceed to perform its partitioning (an 80/20 split is usually performed). All samples with the same qid must remain in the same partition, and the resulting file must be sequentially ordered by the qids in order to be properly recognized by svm-rank, for example:

```
1 qid:12 1:0.978418 2:1 3:0.999853 4:0.999998 
0 qid:12 1:0.267276 2:0.999652 3:0.998534 4:0.999748 
0 qid:12 1:0.003128 2:0.999994 3:0.999946 4:0.999996 
1 qid:14 1:0.931655 2:0.999919 3:0.999806 4:0.999929 
0 qid:14 1:0.173957 2:0.999988 3:0.999988 4:0.999988 
0 qid:14 1:0.10031 2:0.999365 3:0.997991 4:0.999497 
0 qid:14 1:0.020898 2:0.999957 3:0.999652 4:0.999989 
1 qid:15 1:0.859364 2:0.999959 3:0.999588 4:0.999878 
0 qid:15 1:0.211206 2:0.999904 3:0.999904 4:0.999904 
0 qid:15 1:0.126018 2:0.999003 3:0.997379 4:0.99917 
0 qid:15 1:0.012462 2:0.999833 3:0.999373 4:0.999905 
```

A resulting .train (with 80% of the samples) and .dev (with the remaining 20%) files should be therefore obtained.


Afterwards, we run the obtained .train dataset through svm_rank_learn while varying the -c parameter in order to obtain our training models:

```

svm_rank_learn -c 0.001 clusters_multi_rank.svmin.train 4f_multi_rank_0001.model
svm_rank_learn -c 0.01 clusters_multi_rank.svmin.train 4f_multi_rank_001.model
svm_rank_learn -c 0.1 clusters_multi_rank.svmin.train 4f_multi_rank_01.model
svm_rank_learn -c 1 clusters_multi_rank.svmin.train 4f_multi_rank_1.model
svm_rank_learn -c 10 clusters_multi_rank.svmin.train 4f_multi_rank_10.model
svm_rank_learn -c 100 clusters_multi_rank.svmin.train 4f_multi_rank_100.model
svm_rank_learn -c 1000 clusters_multi_rank.svmin.train 4f_multi_rank_1000.model
svm_rank_learn -c 10000 clusters_multi_rank.svmin.train 4f_multi_rank_10000.model
svm_rank_learn -c 100000 clusters_multi_rank.svmin.train 4f_multi_rank_100000.model
svm_rank_learn -c 1000000 clusters_multi_rank.svmin.train 4f_multi_rank_1000000.model

```

Finally, we classify the train and dev data with the obtained models and pick the model with the lowest zero-one error percentage as our ranking model:


```
svm_rank_classify clusters_multi_rank.svmin.train 4f_multi_rank_0001.model out_4f_multi_rank_0001.out
svm_rank_classify clusters_multi_rank.svmin.train 4f_multi_rank_001.model out_4f_multi_rank_001.out
svm_rank_classify clusters_multi_rank.svmin.train 4f_multi_rank_01.model out_4f_multi_rank_01.out
svm_rank_classify clusters_multi_rank.svmin.train 4f_multi_rank_1.model out_4f_multi_rank_1.out
svm_rank_classify clusters_multi_rank.svmin.train 4f_multi_rank_10.model out_4f_multi_rank_10.out
svm_rank_classify clusters_multi_rank.svmin.train 4f_multi_rank_100.model out_4f_multi_rank_100.out
svm_rank_classify clusters_multi_rank.svmin.train 4f_multi_rank_1000.model out_4f_multi_rank_1000.out
svm_rank_classify clusters_multi_rank.svmin.train 4f_multi_rank_10000.model out_4f_multi_rank_10000.out
svm_rank_classify clusters_multi_rank.svmin.train 4f_multi_rank_100000.model out_4f_multi_rank_100000.out
svm_rank_classify clusters_multi_rank.svmin.train 4f_multi_rank_1000000.model out_4f_multi_rank_1000000.out


svm_rank_classify clusters_multi_rank.svmin.dev 4f_multi_rank_0001.model out_4f_multi_rank_0001.out
svm_rank_classify clusters_multi_rank.svmin.dev 4f_multi_rank_001.model out_4f_multi_rank_001.out
svm_rank_classify clusters_multi_rank.svmin.dev 4f_multi_rank_01.model out_4f_multi_rank_01.out
svm_rank_classify clusters_multi_rank.svmin.dev 4f_multi_rank_1.model out_4f_multi_rank_1.out
svm_rank_classify clusters_multi_rank.svmin.dev 4f_multi_rank_10.model out_4f_multi_rank_10.out
svm_rank_classify clusters_multi_rank.svmin.dev 4f_multi_rank_100.model out_4f_multi_rank_100.out
svm_rank_classify clusters_multi_rank.svmin.dev 4f_multi_rank_1000.model out_4f_multi_rank_1000.out
svm_rank_classify clusters_multi_rank.svmin.dev 4f_multi_rank_10000.model out_4f_multi_rank_10000.out
svm_rank_classify clusters_multi_rank.svmin.dev 4f_multi_rank_100000.model out_4f_multi_rank_100000.out
svm_rank_classify clusters_multi_rank.svmin.dev 4f_multi_rank_1000000.model out_4f_multi_rank_1000000.out
```

After obtaining the ranking model, we feed the model to the clustering module and repeat the process for the merge task.

In the case of the merge task, after the clustering module has obtained the best-ranked cluster for a given document, it must decide if the best-ranked cluster is good enough for the new document to join it, or if the new document should create a new cluster by itself.

Repeating the process for the example case with the 4 features (cosine similarity between the document and a cluster, oldest, newest and most relevant timestamp), the procedure for the merge dataset generation typically involves one sample per document. The label 1 indicates that, according to the gold dataset, the document merges into the cluster, and the label 0 indicates that the document creates a new cluster in the pool. The format of the generated dataset is as follows (once again, considering that we are training the model with the 4 forementioned features):

```
1 1:0.349417 2:1 3:0.999991 4:1 
1 1:0.633369 2:0.999999 3:0.999986 4:1 
1 1:0.440288 2:1 3:0.999982 4:1 
0 1:0.700255 2:0.985318 3:0.912474 4:0.983589 
0 1:0.340985 2:0.98206 3:0.736727 4:0.932775 
1 1:0.991226 2:0.999998 3:0.999998 4:0.999998 
1 1:0.817512 2:0.999999 3:0.999994 4:0.999999 
```

Performing negative sampling also improved the results, and as such, it is advised to generate two datasets for the merge task: 
- The original dataset that follows the gold decisions;
- A dataset that receives at most two samples from each document: if the sample has label 1, then a second sample is generated with the second-best ranked cluster (which is negative, and therefore has the label 0).

The dataset with negative sampling should look like this:

```
1 1:0.931655 2:0.999919 3:0.999806 4:0.999929 
0 1:0.173957 2:0.999988 3:0.999988 4:0.999988 
1 1:0.859364 2:0.999959 3:0.999588 4:0.999878 
0 1:0.211206 2:0.999904 3:0.999904 4:0.999904 
1 1:0.97613 2:0.99982 3:0.999348 4:0.999895 
0 1:0.296729 2:0.998971 3:0.997328 4:0.999141 
0 1:0.610337 2:0.998559 3:0.996686 4:0.998761 
1 1:0.686277 2:0.999961 3:0.998992 4:0.99994 
0 1:0.229661 2:0.998534 3:0.996648 4:0.998738 
1 1:0.98956 2:0.999988 3:0.999988 4:0.999988 
```

Following the same procedure as before, we do an 80/20 split both the original and the negative dataset, which should leave us with the following files:

```
clusters_multi_merge.svmin.train
clusters_multi_merge.svmin.neg.train
clusters_multi_merge.svmin.dev
clusters_multi_merge.svmin.neg.dev
```

Afterwards, we generate our model using liblinear. Our usual procedure involves varying the -c parameter, setting the -B parameter to 1 and training the model with the negative training dataset, as follows:

```
train -B 1 -c 0.001 clusters_multi_merge.svmin.neg.train 4f_multi_merge_0001.model
train -B 1 -c 0.01 clusters_multi_merge.svmin.neg.train 4f_multi_merge_001.model
train -B 1 -c 0.1 clusters_multi_merge.svmin.neg.train 4f_multi_merge_01.model
train -B 1 -c 1 clusters_multi_merge.svmin.neg.train 4f_multi_merge_1.model
train -B 1 -c 10 clusters_multi_merge.svmin.neg.train 4f_multi_merge_10.model
train -B 1 -c 100 clusters_multi_merge.svmin.neg.train 4f_multi_merge_100.model
train -B 1 -c 1000 clusters_multi_merge.svmin.neg.train 4f_multi_merge_1000.model
train -B 1 -c 10000 clusters_multi_merge.svmin.neg.train 4f_multi_merge_10000.model
train -B 1 -c 100000 clusters_multi_merge.svmin.neg.train 4f_multi_merge_100000.model
train -B 1 -c 1000000 clusters_multi_merge.svmin.neg.train 4f_multi_merge_1000000.model
```

Given these models, we proceed to evaluate them on the original .dev dataset (we use an additional evaluation script `svm_cross_eval.py`, as `predict` only delivers the predicted output by the model), as follows:

```

predict clusters_multi_merge.svmin.dev 4f_multi_merge_0001.model      out_4f_multi_merge_0001.out     
predict clusters_multi_merge.svmin.dev 4f_multi_merge_001.model      out_4f_multi_merge_001.out     
predict clusters_multi_merge.svmin.dev 4f_multi_merge_01.model       out_4f_multi_merge_01.out      
predict clusters_multi_merge.svmin.dev 4f_multi_merge_1.model        out_4f_multi_merge_1.out       
predict clusters_multi_merge.svmin.dev 4f_multi_merge_10.model       out_4f_multi_merge_10.out      
predict clusters_multi_merge.svmin.dev 4f_multi_merge_100.model      out_4f_multi_merge_100.out     
predict clusters_multi_merge.svmin.dev 4f_multi_merge_1000.model     out_4f_multi_merge_1000.out    
predict clusters_multi_merge.svmin.dev 4f_multi_merge_10000.model    out_4f_multi_merge_10000.out   
predict clusters_multi_merge.svmin.dev 4f_multi_merge_100000.model   out_4f_multi_merge_100000.out  
predict clusters_multi_merge.svmin.dev 4f_multi_merge_1000000.model  out_4f_multi_merge_1000000.out 

python svm_cross_eval.py clusters_multi_merge.svmin.dev out_4f_multi_merge_0001.out     
python svm_cross_eval.py clusters_multi_merge.svmin.dev out_4f_multi_merge_001.out     
python svm_cross_eval.py clusters_multi_merge.svmin.dev out_4f_multi_merge_01.out      
python svm_cross_eval.py clusters_multi_merge.svmin.dev out_4f_multi_merge_1.out       
python svm_cross_eval.py clusters_multi_merge.svmin.dev out_4f_multi_merge_10.out      
python svm_cross_eval.py clusters_multi_merge.svmin.dev out_4f_multi_merge_100.out     
python svm_cross_eval.py clusters_multi_merge.svmin.dev out_4f_multi_merge_1000.out    
python svm_cross_eval.py clusters_multi_merge.svmin.dev out_4f_multi_merge_10000.out   
python svm_cross_eval.py clusters_multi_merge.svmin.dev out_4f_multi_merge_100000.out  
python svm_cross_eval.py clusters_multi_merge.svmin.dev out_4f_multi_merge_1000000.out 
```

From the evaluated models, we pick the model with the highest evaluated F1 and proceed to the clustering evaluation without feeding the gold labels.