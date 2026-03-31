README
======

Privacy-Utility Evaluation with Mondrian k-Anonymity on Adult Dataset
---------------------------------------------------------------------

This project studies the trade-off between **privacy protection** and **machine learning utility** using the **Adult dataset**.The experiment applies **Mondrian k-anonymity** to anonymize quasi-identifiers and then evaluates multiple machine learning models on both original and anonymized data.

Features
--------

*   Mondrian k-anonymity for **numerical quasi-identifiers**
    
*   Mixed-QI setting with:
    
    *   numerical QI anonymized by Mondrian
        
    *   categorical QI generalized using taxonomy-style mappings
        
*   Machine learning models:
    
    *   Logistic Regression
        
    *   Random Forest
        
    *   SVM
        
    *   MLP
        
    *   Naive Bayes
        
    *   1D CNN (PyTorch)
        
*   Evaluation metrics:
    
    *   Accuracy
        
    *   Misclassification Rate
        
    *   Precision
        
    *   Recall
        
    *   AUC
        
*   CSV result export
    
*   PNG plot export
    

Project Structure
-----------------

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   .├── adult_mondrian_privacy_ml.py├── adult.data├── requirements.txt└── results/   `

Dataset
-------

This project uses the **Adult dataset** from UCI.

Place the dataset file in the project folder and name it:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   adult.data   `

Installation
------------

Create a virtual environment first if you want:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python -m venv .venvsource .venv/bin/activate   `

Install dependencies:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   pip install -r requirements.txt   `

If you do not have a requirements.txt, install manually:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   pip install pandas numpy scikit-learn torch matplotlib   `

How to Run
----------

Run the full experiment with:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python adult_mondrian_privacy_ml.py --data_path adult.data --output_dir results   `

Example with custom parameters:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python adult_mondrian_privacy_ml.py \  --data_path adult.data \  --output_dir results \  --k_values 2 5 10 20 50 100 \  --cnn_epochs 10 \  --cnn_batch_size 128 \  --cnn_learning_rate 0.001   `

Command Line Arguments
----------------------

ArgumentDescriptionDefault--data\_pathPath to adult.datarequired--output\_dirDirectory to save resultsresults--k\_valuesList of k values for k-anonymity2 5 10 20 50 100--seedRandom seed42--cnn\_epochsNumber of training epochs for PyTorch CNN10--cnn\_batch\_sizeBatch size for CNN128--cnn\_learning\_rateLearning rate for CNN0.001

Output Files
------------

After running, the script will generate files like:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   results/├── adult_cleaned.csv├── adult_full_mondrian_numeric_qi_k2.csv├── adult_full_mondrian_numeric_qi_k5.csv├── adult_full_mondrian_numeric_qi_k10.csv├── adult_full_mondrian_numeric_qi_k20.csv├── adult_full_mondrian_numeric_qi_k50.csv├── adult_full_mondrian_numeric_qi_k100.csv├── adult_full_mondrian_mixed_qi_k2.csv├── adult_full_mondrian_mixed_qi_k5.csv├── adult_full_mondrian_mixed_qi_k10.csv├── adult_full_mondrian_mixed_qi_k20.csv├── adult_full_mondrian_mixed_qi_k50.csv├── adult_full_mondrian_mixed_qi_k100.csv├── partition_summary_numeric_qi.csv├── partition_summary_mixed_qi.csv├── results_numeric_qi_classical.csv├── results_numeric_qi_cnn_pytorch.csv├── results_mixed_qi_classical.csv├── results_mixed_qi_cnn_pytorch.csv├── all_results.csv├── accuracy_table.csv├── auc_table.csv├── experiment_metadata.json└── plots/   `

Saved Plots
-----------

PNG plots are stored in:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   results/plots/   `

Examples:

*   Accuracy vs k
    
*   AUC vs k
    
*   Precision vs k
    
*   Recall vs k
    
*   Misclassification Rate vs k
    
*   Per-model comparison:
    
    *   Numeric\_QI vs Mixed\_QI
        

Experiment Settings
-------------------

### Sensitive Attribute

*   income
    

### Numerical Quasi-Identifiers

*   age
    
*   fnlwgt
    
*   education-num
    
*   capital-gain
    
*   capital-loss
    
*   hours-per-week
    

### Categorical Quasi-Identifiers

*   workclass
    
*   education
    
*   marital-status
    
*   occupation
    
*   relationship
    
*   race
    
*   sex
    
*   native-country
    

Notes
-----

*   k = 0 in result tables refers to the **original dataset**
    
*   Numeric-QI means only numerical QI are anonymized
    
*   Mixed-QI means:
    
    *   numerical QI are anonymized with Mondrian
        
    *   categorical QI are generalized globally
        
*   In this implementation, Mondrian partitioning is based only on numerical QI
    

Reproducibility
---------------

The experiment uses:

*   fixed random seed: 42
    
*   fixed train/test split:
    
    *   80% training
        
    *   20% testing
        
    *   stratified by target label
        

The same split is reused across all models and privacy settings for fair comparison.

Example Results
---------------

The final results are saved in:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   results/all_results.csv   `

This file contains:

*   model name
    
*   setting
    
*   k value
    
*   accuracy
    
*   misclassification rate
    
*   precision
    
*   recall
    
*   AUC
    

Citation / Reference
--------------------

If you use this project in a report, describe it as:

> Mondrian k-anonymity was applied to anonymize quasi-identifiers in the Adult dataset. Machine learning performance was then evaluated on original and anonymized data using Logistic Regression, Random Forest, SVM, MLP, Naive Bayes, and a PyTorch-based 1D CNN.

Troubleshooting
---------------

### 1\. Missing dataset

Make sure adult.data exists and the path is correct:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python adult_mondrian_privacy_ml.py --data_path adult.data   `

### 2\. PyTorch GPU not detected

The script automatically uses CUDA if available. Otherwise it runs on CPU.

### 3\. Long CNN training time

Reduce epochs:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python adult_mondrian_privacy_ml.py --data_path adult.data --cnn_epochs 5   `

License
-------

Add your preferred license here.
