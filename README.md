# Categorizing Allegations Against CPD

## Context

Recently, the [Invisible Institute](https://invisible.institute/), a local investigative newsroom focused on police misconduct, obtained descriptive text on thousands of complaints through additional Freedom of Information Act (FOIA) efforts. With text descriptions of police incidents in hand, the public is able to comprehend police incidents in detail.

Upon comparing incident descriptions against complaint data, the Invisible Institute found that categories provided by CPD have been too broad or inaccurate to assist in providing journalistic leads. For example, incidences of sexual assault of a victim by a police officer are of interest to journalists to investigate and eventually expose systematic abuse, but sexual assault is not a classification provided in CPDP’s (Citizens Police Data Project) tabular data.

As a result, journalists must rely on text descriptions of each incident to identify specific violations. It is of public interest to improve access by providing more informative categories.

## Objectives

This project aims to function as a starting point so the Invisible Institute can more easily classify allegations. Therefore, we use both supervised and unsupervised learning in order to explore what works best for classifying these allegations.

## Repository structure

These repository's contents are as follows:

```
.
├── README.md
├── requirements.txt
└── src/
    ├── unsupervised/
    │    ├──    narratives.csv
    │    ├──    lda-analysis-sagemaker.ipynb
    │    ├──    preprocessing.py
    │    └──    lda_analysis_and_evaluation.ipynb
    └── supervised/
         ├──    Naive-Bayes-EM.ipynb
         ├──    supervised_pipeline.ipynb
         ├──    mlp_pipeline.ipynb
         ├──    cleaning.ipynb
         ├──    mlp_results.csv
         ├──    results.csv
         ├──    plain_text.csv
         └──    training_with_text.csv
```

## File description

Note: lines of code are estimates.

_Unsupervised_

- lda-analysis-sagemaker.ipynb: Jupyter notebook that was run on an ml instance on Sagemaker. This file first uses preprocess.py to obtain document-term count vectors which it then splits into a training and a test set. Then, the code uploads the data to an S3 bucket so it can be tuned. Sagemaker fires up instances and parallelizes the tuning of the alpha values and number of topic parameters and then outputs a best model (using per-word log-likelihood on the test dataset). The best model is then fetched from the S3 bucket. Finally, the code extracts the topics and top 10 keywords for each of them and creates an inference point so the documents' topics can be inferred. All external code is referred to in sources or in specific comments. Lines of code: ~ 260

- preprocessing.py: Helper file for preprocessing. This file uses nltk and scikit to clean, stem and vectorize the allegations text. A PreProcess class that overrides sklearn’s tokenizer was constructed; this class has a method that constructs the doc_term_matrix that is then used in lda-analysis-sagemaker.ipynb. Lines of code: ~ 131

- lda_analysis_and_evaluation.ipynb: This Jupyter Notebook imports and processes the data, builds the LDA models under different variations of the parameters, and evaluates the results with the help of data visualization. The notebook has ~ 290 lines of codes. First, we imported “narratives.csv” then pre-process to remove stop words and remove any duplicated content in the text column. The notebook also contains several functions: (1) create word cloud visualization (2) create a dataframe to group documents by dominant topic (3) create bar chart visualization. We created a loop to run through different variations of the parameter num_topics and show the data visualization and suggested topics for each model. Finally, we use the package pyLDAvis to create a final visualization for each model created. Lines of code: ~ 290

- narratives.csv: Contains the text content of 10k+ allegations obtained via a FOIA request. The relevant information for this analysis is in the “text” column. We filtered the data based on unique complaint id’s (“cr_id”) and unique text content for the “Initial / Intake Allegation” category in “column_name” and corrected anomalies such as the allegation being repeated twice

_Supervised_

- Naive-Bayes-EM.ipynb: This notebook implements both traditional multinomial Naive Bayes and semi-supervised expectation maximization (EM) algorithms on our limited set of labelled data. It draws heavily from the examples demonstrated in this github repository of which a submodule is included in order to implement a semi-supervised EM classifier. Our originally classified data contained multiple classifications for each set of text. In our data cleaning, we reassigned to a single classification and removed categories that had very few assignments. We used TF-IDF features and added common stop words in police complaints.For us, the traditional multinomial Naive-Bayes classifier performed with higher accuracy than the semi-supervised model. However, we believe this is largely due to the need to treat some of the labeled data as unlabeled to run the semi-supervised model. At the end of the document, we include word clouds representative of each category for both models. Lines of code: ~150 lines

- Text_Classification_Using_EM_And_Semisupervied_Learning: This submodule is a separate project which implements a comparison between traditional multinomial NB classifiers and semi-supervised EM classifiers. We include it here because we draw on some of their code in our implementation.

- training_with_text.csv: This is a dataset of ~300 allegations which were manually categorized by Invisible Institute’s volunteers. Our supervised models draw on this data as a training set. The text is contained in the ‘text_content’ field and categories are assigned in 13 columns which are binaries.

- cleaning.ipynb: Jupyter notebook containing code that cleans the data on “training_with_text.csv” and produces “plain_text.csv”, which is ready to be processed on supervised_pipeline.ipynb and mlp_pipeline.ipynb. Lines of code ~ 29 lines.

- mlp_pipeline.ipynb: Jupyter notebook that was run on an AWS EMR cluster with a pyspark kernel. The file uses pyspark to read the text data from an S3 bucket, tokenizes the text of the complaints, removes stop words and generates a Word2Vec embedding. It then uses this embedding to train Multilayer Perceptron models with different layer parameters and chooses the best performing model for each complaint category. It then saves the information about the best performing model on a csv file named “mlp_results.csv”. Lines of code: ~ 109 lines.

- supervised_pipeline.ipynb: Jupyter notebook that was run on an AWS EMR cluster with a pyspark kernel. The file uses pyspark to read the text data from an S3 bucket, tokenizes the text of the complaints, removes stop words and generates a Word2Vec embedding. It then uses this embedding to train Logistic Regression models with different regularization parameters and chooses the best performing model for each complaint category. It then saves the information about the best performing model on a csv file named “results.csv”. Lines of code: ~ 125 lines.

- plain_text.csv: data generated by cleaning “training_with_text.csv” running the code on cleaning.ipynb.

- results.csv: data file containing the results of the Logistic Regression models trained and tested on supervised_pipeline.ipynb.

- mlp_results.csv: data file containing the results of the Logistic Regression models trained and tested on mlp_pipeline.ipynb.
