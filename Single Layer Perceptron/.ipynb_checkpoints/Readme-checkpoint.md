## Student Placement Prediction
This project uses a foundational deep learning model, a single-layer perceptron, to predict student placement status based on their CGPA (Cumulative Grade Point Average) and IQ scores. The goal is to explore how a simple linear classifier can separate and classify student data.

## Project Structure

* `Untitled.ipynb`: This Jupyter Notebook contains the complete analysis, from data loading and exploration to model training and evaluation.

* `Placement.csv`: The dataset used for this project, containing student information including CGPA, IQ, and placement status (1 for placed, 0 for not placed).

## Technologies Used

* Python

* Jupyter Notebook

* Pandas

* Matplotlib

* Seaborn

* mlxtend

## Project Workflow

1. **Data Loading and Exploration**: The `Placement.csv` file is loaded into a pandas DataFrame. Basic data exploration is performed to understand the structure and content of the dataset.

2. **Data Visualization**: Visualizations are created to analyze the relationship between the features (CGPA, IQ) and the target variable (placement). This helps in understanding the data distribution and potential correlations.

3. **Model Training**: A machine learning model is trained on the dataset to learn the patterns and predict placement. The `mlxtend` library is used to visualize the decision boundary of the model.

4. **Results**: The project demonstrates how to build and evaluate a predictive model for a simple classification task. The visualizations provide insight into how the model makes its predictions based on the input features.

## Getting Started

To run this project, you will need to have Python and the necessary libraries installed. You can install the dependencies using pip:

```pip install pandas matplotlib seaborn mlxtend jupyter```

Then, you can open and run the Jupyter Notebook:

```jupyter notebook Untitled.ipynb```

## Contributing

Feel free to fork this repository and contribute to the project. Suggestions for improvements, bug fixes, or new features are welcome.
"""