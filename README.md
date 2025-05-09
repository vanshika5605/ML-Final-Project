# Machine Learning Model Comparison

This project evaluates and compares Random Forest, K-Nearest Neighbors (KNN), and Neural Network algorithms across five datasets using custom implementations and hyperparameter tuning.

---

## Environment Setup Instructions

### 1. Clone or Download the Repository
Download the ZIP file which contains all code files, datasets, and the `requirements.txt` file.

### 2. Create and Activate a Virtual Environment

#### On macOS/Linux:
```bash
python -m venv myenv
source myenv/bin/activate
```

#### On Windows:
```bash
python -m venv myenv
myenv\Scripts\activate
```

### 3. Install Required Packages
```bash
pip install -r requirements.txt
```

### 4. Prepare the Datasets
Place all datasets inside a folder named `data` at the same level as your code file directories:

```
project/
│
├── data/
│   ├── dataset1.csv
│   ├── dataset2.csv
│   └── ...
├── knn
├── random-forest
├── ...
```

---

## Running the Algorithms

### Random Forest
Open `main.py` and uncomment the lines corresponding to the dataset you want to run:

- **Dataset 1:** Lines 105–106  
- **Dataset 2:** Lines 108–109  
- **Dataset 3:** Lines 111–112  
- **Dataset 4:** Lines 114–116  
- **Dataset 5:** Lines 118–134  

Then run:
```
python main.py
```

### K-Nearest Neighbors (KNN)
Open `KNN.py` and uncomment the relevant block:

- **Dataset 1:** Lines 211–216  
- **Dataset 2:** Lines 218–222  
- **Dataset 3:** Lines 224–228  
- **Dataset 4:** Lines 230–238  
- **Dataset 5:** Lines 240–250  

Then run:
```
python KNN.py
```

### Neural Network

---

## Datasets Evaluated

1. Handwritten Digits Recognition  
2. Parkinson’s Disease Detection  
3. Rice Grains Classification  
4. Credit Approval Prediction  
5. Student Performance Categorization

