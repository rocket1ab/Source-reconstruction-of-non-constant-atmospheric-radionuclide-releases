# Source reconstruction of non-constant atmospheric radionuclide releases
**Update 30/10/2023**: Added example data and rearrange the code.

## Requirements
* graphviz==0.14.2
* hyperopt==0.2.5
* lightgbm==3.1.1
* matplotlib==3.1.3
* numpy==1.21.5
* pandas==1.1.5
* scikit_learn==0.23.2
* scipy==1.7.3
* xgboost==1.2.1

## Example data
The example data for running the code is stored in the datafile.mat.

## Installation
We recommend installing relevant dependencies in a virtual environment of the Anaconda. If so, you could create a virtual environment by running the following code in the command line of Anaconda Prompt:
```
conda create -n your_env_name python=3.7.6
```
Then you can finish the installation by the following code. If you didn't install the git, you can also directly download the code in the link: https://github.com/rocket1ab/Source-reconstruction-of-non-constant-atmospheric-radionuclide-releases 
```
git clone https://github.com/rocket1ab/Source-reconstruction-of-non-constant-atmospheric-radionuclide-releases
cd Source-reconstruction-of-non-constant-atmospheric-radionuclide-releases
pip install -r requirements.txt
```
## Usage
### Step1 Generating Training_datasets
```
cd Source-reconstruction-of-non-constant-atmospheric-radionuclide-releases
python Build_feature_training_dataset.py
```
### Step2 Generating Testing_datasets
```
cd Source-reconstruction-of-non-constant-atmospheric-radionuclide-releases
python Build_feature_testing_dataset.py
```
### Step3 Training and output the source localization results
```
cd Source-reconstruction-of-non-constant-atmospheric-radionuclide-releases
ipython -c "%run Oct4-source-localization.ipynb"
```

## Results
The training datasets and testing datasets will be stored in the directory './Training_datasets' and './Testing_datasets', respectively.
The source localization results are stored in the directory './Reconstruction_results'.
