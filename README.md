# Privacy-Utility Evaluation with Mondrian k-Anonymity on Adult Dataset

This project studies the trade-off between privacy protection and machine learning utility using the Adult dataset.
The experiment applies Mondrian k-anonymity to anonymize quasi-identifiers and then evaluates multiple machine learning models on both original and anonymized data.

## Course Name: Data Privacy and Security (1142CS5164701)
### Group Name: SecureBytes
Member:
1. Alim Misbullah D11415803	
2. Laina Farsiah D11415802
3. Stenly Ibrahim Adam D11215809
4. Aurelio Naufal Effendy M11415802

## Features
* Mondrian k-anonymity for numerical quasi-identifiers
* Mixed-QI setting with:
  * numerical QI anonymized by Mondrian
  * categorical QI generalized using taxonomy-style mappings
* Machine learning models:
  * Logistic Regression
  * Random Forest
  * SVM
  * MLP
  * 1D CNN (PyTorch)
* Evaluation metrics:
  * Accuracy
  * Misclassification Rate
  * Precision
  * Recall
  * AUC
* CSV result export
* PNG plot export

## Project Structure
```
.
├── adult_mondrian_privacy_ml.py
├── adult.data
├── requirements.txt
└── results/
```
## Dataset
This project uses the Adult dataset from UCI. Place the dataset file in the project folder and name it:
```
adult.data
```

## Installation
Create a virtual environment first if you want:
```
python -m venv .venv
source .venv/bin/activate
```
Install dependencies:
```
pip install -r requirements.txt
```
If you do not have a requirements.txt, install manually:
```
pip install pandas numpy scikit-learn torch matplotlib
```
## How to Run
Run the full experiment with:
```
python adult_mondrian_privacy_ml.py --data_path adult.data --output_dir results
```
Example with custom parameters:
```
python adult_mondrian_privacy_ml.py \
  --data_path adult.data \
  --output_dir results \
  --k_values 2 5 10 20 50 100 \
  --cnn_epochs 10 \
  --cnn_batch_size 128 \
  --cnn_learning_rate 0.001
```

## Command Line Arguments

| Argument | Description | Default |
| --- | --- | --- |
| --data_path | Path to adult.data | required |
| --output_dir | Directory to save results | results |
| --k_values | List of k values for k-anonymity | 2 5 10 20 50 100 |
| --seed | Random seed |	42 |
| --cnn_epochs | Number of training epochs for PyTorch CNN | 10 |
| --cnn_batch_size |	Batch size for CNN | 128 |
| --cnn_learning_rate | Learning rate for CNN | 0.001 |

## Output Files
After running, the script will generate files like:

```
results/
├── adult_cleaned.csv
├── adult_full_mondrian_numeric_qi_k2.csv
├── adult_full_mondrian_numeric_qi_k5.csv
├── adult_full_mondrian_numeric_qi_k10.csv
├── adult_full_mondrian_numeric_qi_k20.csv
├── adult_full_mondrian_numeric_qi_k50.csv
├── adult_full_mondrian_numeric_qi_k100.csv
├── adult_full_mondrian_mixed_qi_k2.csv
├── adult_full_mondrian_mixed_qi_k5.csv
├── adult_full_mondrian_mixed_qi_k10.csv
├── adult_full_mondrian_mixed_qi_k20.csv
├── adult_full_mondrian_mixed_qi_k50.csv
├── adult_full_mondrian_mixed_qi_k100.csv
├── partition_summary_numeric_qi.csv
├── partition_summary_mixed_qi.csv
├── results_numeric_qi_classical.csv
├── results_numeric_qi_cnn_pytorch.csv
├── results_mixed_qi_classical.csv
├── results_mixed_qi_cnn_pytorch.csv
├── all_results.csv
├── accuracy_table.csv
├── auc_table.csv
├── experiment_metadata.json
└── plots/
```

## Saved Plots
Saved Plots
```
results/plots/
```

Examples:
* Accuracy vs k
* AUC vs k
* Precision vs k
* Recall vs k
* Misclassification Rate vs k
* Per-model comparison:
  * Numeric_QI vs Mixed_QI

## Experiment Settings
### Sensitive Attribute
* income
### Numerical Quasi-Identifiers
* age
* fnlwgt
* education-num
* capital-gain
* capital-loss
* hours-per-week
### Categorical Quasi-Identifiers
* workclass
* education
* marital-status
* occupation
* relationship
* race
* sex
* native-country

## Notes
* `k = 0` in result tables refers to the original dataset
* Numeric-QI means only numerical QI are anonymized
* Mixed-QI means:
  * numerical QI are anonymized with Mondrian
  * categorical QI are generalized globally
* In this implementation, Mondrian partitioning is based only on numerical QI

## Reproducibility
The experiment uses:
* fixed random seed: `42`
* fixed train/test split:
  * 80% training
  * 20% testing
  * stratified by target label
The same split is reused across all models and privacy settings for fair comparison.

## Example Results
The final results are saved in:
```
results/all_results.csv
```
This file contains:
* model name
* setting
* k value
* accuracy
* misclassification rate
* precision
* recall
* AUC

## References

```
  LeFevre, K., DeWitt, D. J., &#38; Ramakrishnan, R. (2006). Mondrian Multidimensional K-Anonymity. <i>International Conference on Data Engineering</i>, 25. https://doi.org/10.1109/ICDE.2006.101

```

## Troubleshooting
1. Missing dataset
Make sure adult.data exists and the path is correct:
```
python adult_mondrian_privacy_ml.py --data_path adult.data
```

2. PyTorch GPU not detected
The script automatically uses CUDA if available. Otherwise it runs on CPU.

3. Long CNN training time
Reduce epochs:
```
python adult_mondrian_privacy_ml.py --data_path adult.data --cnn_epochs 5
```

## References

1. K. LeFevre, D. J. DeWitt, and R. Ramakrishnan, *“Mondrian Multidimensional K-Anonymity,”* Proceedings of the IEEE International Conference on Data Engineering (ICDE), 2006.

2. H. Wimmer and L. Powell, *“A Comparison of the Effects of K-Anonymity on Machine Learning Algorithms,”* Proceedings of the Conference for Information Systems Applied Research (CONISAR), vol. 2167, 2014.

3. J. Brownlee, *“PyTorch Tutorial: Develop Deep Learning Models,”* Machine Learning Mastery. [Link](https://machinelearningmastery.com/pytorch-tutorial-develop-deep-learning-models/)

4. J. Brownlee, *“Tutorial: First Neural Network with Keras,”* Machine Learning Mastery. [Link](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/)

5. DataDrivenInvestor, *“Building Neural Network Using Keras for Classification,”* Medium. [Link](https://medium.com/datadriveninvestor/building-neural-network-using-keras-for-classification-3a3656c726c1)

