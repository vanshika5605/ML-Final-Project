## Instructions to setup the environment

#### Download the zip (includes the code, data and the requirements file)

#### Create a virtual environment using below commands

```python -m venv myenv```

#### Activate the virtual environment 
```source myenv/bin/activate```  # On macOS/Linux
```myenv\Scripts\activate```  # On Windows

#### Install the requirements

```pip install -r requirements.txt```

#### Place the datasets

Put the datasets in a folder named "data" at the same level as the code files.

## Instructions to run the classifier for different datasets

### WDBC Dataset

Uncomment line 101 in main.py, and then run ```python main.py```

### Loan Dataset

Uncomment line 103-104 in main.py, and then run ```python main.py```

### Raisin Dataset

Uncomment line 106 in main.py, and then run ```python main.py```

### Titanic Dataset

Uncomment line 108-109 in main.py, and then run ```python main.py```

## Instructions to run the classifier for different datasets for different depths for hyperparameter tuning of maximal_depth

### WDBC Dataset

Uncomment line 113 in main.py, and then run ```python main.py```

### Loan Dataset

Uncomment line 115-116 in main.py, and then run ```python main.py```

### Raisin Dataset

Uncomment line 118 in main.py, and then run ```python main.py```

### Titanic Dataset

Uncomment line 120-121 in main.py, and then run ```python main.py```