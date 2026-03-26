# Mobile Phone Price Prediction

## Overview
This project aims to predict the price of mobile phones based on various features using machine learning algorithms. The model analyzes the given features and predicts the price category of the mobile phones.

## Features
- Price prediction based on specifications
- User-friendly interface for inputting mobile features
- Data visualization for better understanding
- Ability to train and test models

## Installation Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/tugcesi/mobil-phone-price-prediction.git
   ```
2. Navigate into the project directory:
   ```bash
   cd mobil-phone-price-prediction
   ```
3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Prepare your dataset in the specified format.
2. Run the main script to train the model:
   ```bash
   python train.py
   ```
3. Use the prediction script to estimate prices:
   ```bash
   python predict.py --input your_input_file.csv
   ```

## Models
The project utilizes various machine learning models including:
- Linear Regression
- Decision Trees
- Random Forest
- Support Vector Machines (SVM)

Users can experiment with different models to see which yields the best results for their dataset.

## Structure
```
mobil-phone-price-prediction/
│
├── data/                     # Directory for datasets
├── models/                   # Directory for machine learning models
├── src/                      # Source code
│   ├── train.py              # Training script
│   └── predict.py            # Prediction script
├── requirements.txt          # Required libraries
└── README.md                 # Project documentation
```

## Technologies
- Python 3.x
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Jupyter Notebook

## Acknowledgments
- Special thanks to the contributors who helped in building this project.
- Influential research papers and documentation that guided the model creation.
- Open-source community for shared datasets and tools.