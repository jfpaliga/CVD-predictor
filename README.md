# CVD Predictor - A Predictive Classification Model for Determining Risk of Heart Disease

Live link here - PLACEHOLDER

CVD Predictor is a machine-learning (ML) project using a publically available dataset to determine whether a ML pipeline could be built in order to predict whether a patient is at risk of heart disease. This was achieved by using a classification task, using the HeartDisease attribute from the dataset as the target and the remaining attributes as features.

## Table of Contents

- [Dataset Content](#dataset-content)
- [Business Requirements](#business-requirements)
- [Hypothesis](#hypothesis-and-how-to-validate)
- [Mapping Business Requirements to Data Visualisation and ML Tasks](#the-rationale-to-map-the-business-requirements-to-the-data-visualizations-and-ml-tasks)
- [ML Business Case](#ml-business-case)
- [Epics and User Stories](#epics-and-user-stories)
- [Dashboard Design](#dashboard-design)
- [Technologies Used](#technologies-used)
- [Testing](#testing)
- [Unfixed Bugs](#unfixed-bugs)
- [Deployment](#deployment)
- [Credits](#credits)
- [Acknowledgements](#acknowledgements)


## Dataset Content

* The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/data). Each row represents a patient and each column contains a patient attribute. The dataset includes information about:
    - patient age and sex
    - patient medical information such as blood pressure, heart rate and cholesterol levels
    - whether or not the patient had heart disease

| Attribute      | Information                               | Units                                                                                       |
|----------------|-------------------------------------------|---------------------------------------------------------------------------------------------|
| Age            | age of the patient                        | years                                                                                       |
| Sex            | sex of the patient                        | M: Male, F: Female                                                                          |
| ChestPainType  | chest pain type                           | TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic          |
| RestingBP      | resting blood pressure                    | mm Hg                                                                                       |
| Cholesterol    | serum cholesterol                         | mm/dl                                                                                       |
| FastingBS      | fasting blood sugar                       | 1: if FastingBS > 120 mg/dl, 0: otherwise                                                   |
| RestingECG     | resting electrocardiogram results         | Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria |
| MaxHR          | maximum heart rate achieved               | Numeric value between 60 and 202                                                            |
| ExerciseAngina | exercise-induced angina                   | Y: Yes, N: No                                                                               |
| Oldpeak        | oldpeak = ST depression induced by exercise relative to rest | Numeric value measured in depression                                     |
| ST_Slope       | the slope of the peak exercise ST segment | Up: upsloping, Flat: flat, Down: downsloping                                                |
| HeartDisease   | output class                              | 1: heart disease, 0: Normal                                                                 |


## Business Requirements
* Cardiovascular diseases are the number 1 cause of death globally, accounting for 31% of all deaths worldwide. People with cardiovascular disease or who are at high risk of disease need early detection and management. A fictional organisation has requested a data practitioner to analyse a dataset of patients from a number of different hospitals in order to determine what factors can be attributed to a high risk of disease and whether patient data can accurately predict risk of heart disease.

* Business Requirement 1 - The client is interested in which attributes correlate most closely with heart disease, ie what are the most common risk factors?
* Business Requirement 2 - The client is interested in using patient data to predict whether or not a patient is at risk of heart disease.


## Hypothesis and how to validate?
* Hypothesis 1:
    - We suspect that the highest risk factors involved in heart disease are cholesterol and maximum heart rate.
    - **Validation**: a correlation analysis that indicates a strong relationship between the above features and the target 'HeartDisease'.

* Hypothesis 2:
    - We suspect that a successful prediction will rely on a large number of parameters.
    - **Validation**: analysis of the feature importance from the ML pipeline after hyperparameter optimisation will indicate at least 5/11 of the features are necessary for a prediction.

* Hypothesis 3:
    - We suspect that men over 50 with high cholesterol are the most at-risk patient group.
    - **Validation**: analysis and visualisation using a parallel plot of the dataset to determine a 'typical' heart disease patient profile


## The rationale to map the business requirements to the Data Visualizations and ML tasks
* **Business Requirement 1**: Data Visualisation and Correlation study
    - We need to perform a correlation study to determine which features correlate most closely to the target.
    - A Pearson's correlation will indicate linear relationships between numerical variables.
    - A Spearman's correlation will measure the monotonic relationships between variables.
    - A Predictive Power Score study can also be used to determine relationships between attributes regardless of data type (6/11 features are categorical).

* **Business Requirement 2**: Classification Model
    - We need to predict whether a patient is at risk of heart disease or not.
    - Therefore we need to build a binary classification model.
    - A conventional machine learning pipeline will be able to map the relationships between the features and target.
    - Extensive hyperparameter optimisation will give us the best chance at a highly accurate prediction.


## ML Business Case
**Classification Model**
* We want a ML model to predict whether a patient is at risk of heart disease based upon previously gathered patient data. The target variable, 'HeartDisease', is categorical and contains two classes: 0 (no heart disease) and 1 (heart disease).
* We will consider a **classification model**, a supervised model with a two-class, single-label output that matches the target.
* The model success metrics are:
    - at least 75% recall for heart disease on the train and test sets
* The model will be considered a failure if:
    - the model fails to achieve 75% recall for heart disease
    - the model fails to achieve 70% precision for no heart disease (falsely indicating patients are at risk)
* The model output is defined as a flag, indicating if a patient will have heart disease or not and the associated probability of heart disease.
* The training data to fit the model comes from: [Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/data)
    - The dataset contains: 918 observations and 12 attributes.
    - Target: HeartDisease; Features: all other attributes.

## Epics and User Stories
* The project was split into 5 Epics based upon the Data Visualisation and Machine Learning tasks and within each of these, user stories were set out to enable an agile methodology.

### Epic - Information Gathering and Data Collection
* **User Story** - As a data analyst, I can import the dataset from Kaggle so that I can save the data in a local directory.
* **User Story** - As a data analyst, I can load a saved dataset so that I can analyse the data to gain insights on what further tasks may be required.

### Epic - Data Visualization, Cleaning, and Preparation
* **User Story** - As a data scientist, I can visualise the dataset so that I can interpret which attributes correlate most closely with heart disease (Business Requirement 1).
* **User Story** - As a data analyst, I can evaluate the dataset to determine what data cleaning tasks need to be carried out.
* **User Story** - As a data analyst, I can impute or drop missing data to prepare the dataset for a ML model.
* **User Story** - As a data analyst, I can determine whether the target requires balancing in order to ensure the ML is not fed imbalanced data.
* **User Story** - As a data scientist, I can carry out feature engineering to best transform the data for the ML model.

### Epic - Model Training, Optimization and Validation
* **User Story** - As a data scientist, I can split the data into a train and test set to prepare it for the ML model.
* **User Story** - As a data engineer, I can fit a ML pipeline with all the data to prepare the ML model for deployment.
* **User Story** - As a data engineer, I can determine the best algorithm for predicting heart disease to use in the ML model.
* **User Story** - As a data engineer, I can carry out an extensive hyperparameter optimisation to ensure the ML model gives the best results.
* **User Story** - As a data scientist, I can determine the best features from the ML pipeline to determine whether the ML model can be optimised further.
* **User Story** - As a data scientist, I can evaluate the ML model's performance to determine whether it can successfully predict heart disease (Business Requirement 2).

### Epic - Dashboard Planning, Designing, and Development
* **User Story** - As a non-technical user, I can view a project summary that describes the project, dataset and business requirements to understand the project at a glance.
* **User Story** - As a non-technical user, I can view the project hypotheses and validations to determine what the project was trying to achieve and whether it was successful.
* **User Story** - As a non-technical user, I can enter unseen data into the model and receive a prediction.
* **User Story** - As a technical user, I can view the correlation analysis to see how the outcomes were reached.
* **User Story** - As a technical user, I can view all the data to understand the model performance and see statistics related to the model.
* **User Story** - As a non-technical user, I can view the project conclusions to see whether the model was successful and if the business requirements were met.

### Epic - Dashboard Deployment and Release
* **User Story** - As a user, I can view the project dashboard on a live deployed website.
* **User Story** - As a technical user, I can follow instructions in the readme to fork the repository and deploy the project for myself.


## Dashboard Design
### Page 1: Project Summary
* **Section 1 - Summary**
    * Introduction to project
    * Description of dataset, where was it sourced?
    * Link to readme
* **Section 2 - Business Requirements**
    * Description of business requirements

### Page 2: Project Hypotheses
* Outline the three project hypothesis
* Present validation of each hypothesis

### Page 3: Feature Correlation Study
* State business requirement 1
* Overview of dataset - display first 5 rows of data and describe dataset shape
* Display correlation results and PPS heatmap
* Display distributions of correlated features against target
* Conclusions

### Page 4: Heart Disease Prediction
* State business requirement 2
* Widget inputs for prediction
* "Run prediction" button to run inputted data through the ML model and output a prediction and % chance

### Page 5: Classification Performance Metrics
* Summary of model performance and metrics
* Model pipeline, features used to train the model and how they were selected
* Documentation of model performance on train and test sets


## Technologies Used

The technologies used throughout the development are listed below:

### Languages

* [Python](https://www.python.org/)

### Python Packages

* [Pandas](https://pandas.pydata.org/docs/index.html)

    Open source library for data manipulation and analysis.

* [Numpy](https://numpy.org/doc/stable/index.html)

    Adds support for large, multi-dimensional arrays and matrices, and high-level mathematical functions.

* [YData Profiling](https://docs.profiling.ydata.ai/latest/)

    For data profiling and exploratory data analysis.

* [Matplotlib](https://matplotlib.org/)

    Comprehensive library for creating static, animated and interactive visualisations.

* [Seaborn](https://seaborn.pydata.org/)

    Another data visualisation library for drawing attractive and informative statistical graphics.

* [Pingouin](https://pingouin-stats.org/build/html/index.html)

    Open source statistical package for simple yet exhaustive stats functions.

* [Feature-engine](https://feature-engine.trainindata.com/en/latest/)

    Library with multiple transformers to engineer and select features for machine learning models.

* [ppscore](https://pypi.org/project/ppscore/)

    Library for detecting linear or non-linear relationships between two features.

* [scikit-learn](https://scikit-learn.org/stable/)

    Open source machine learning library that features various algorithms for training a ML model.

* [SciPy](https://scipy.org/)

    Library used for scientific computing and technical computing.

* [XGBoost](https://xgboost.readthedocs.io/en/stable/)

    Optimised distributed gradient boosting library.

* [Imbalanced-learn](https://imbalanced-learn.org/stable/)

    Provides tools for dealing with classification problems with imbalanced classes.

* [Joblib](https://joblib.readthedocs.io/en/stable/)

    Provides tools for lightweight pipelining, e.g. caching output values.

## Testing
* testing


## Issues

* Describe v1 development and issues with deployment to Heroku


## Unfixed Bugs
* You will need to mention unfixed bugs and why they were not fixed. This section should include shortcomings of the frameworks or technologies used. Although time can be a significant variable to consider, paucity of time and difficulty understanding implementation is not a valid reason to leave bugs unfixed.


## Deployment
### Heroku

* The App live link is: https://YOUR_APP_NAME.herokuapp.com/ 
* Set the runtime.txt Python version to a [Heroku-20](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
* The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click now the button Open App on the top of the page to access your App.
6. If the slug size is too large then add large files not required for the app to the .slugignore file.


## Credits 

* In this section, you need to reference where you got your content, media and extra help from. It is common practice to use code from other repositories and tutorials, however, it is important to be very specific about these sources to avoid plagiarism. 
* You can break the credits section up into Content and Media, depending on what you have included in your project. 

### Content 

- The text for the Home page was taken from Wikipedia Article A
- Instructions on how to implement form validation on the Sign-Up page was taken from [Specific YouTube Tutorial](https://www.youtube.com/)
- The icons in the footer were taken from [Font Awesome](https://fontawesome.com/)

- The code for the histogram/QQ plots and PPS score heatmaps were taken from the Code Institute "Churnometer" walkthrough project.
- Other bits of code used from Code Institute, check all notebooks.
- The parameters to optimise and related values were taken from Abhishek Thakur's ["Approaching (Almost) Any Machine Learning Problem"](https://www.linkedin.com/pulse/approaching-almost-any-machine-learning-problem-abhishek-thakur/) post on LinkedIn and Jason Brownlee's [Machine Learning Mastery](https://machinelearningmastery.com/hyperparameters-for-classification-machine-learning-algorithms/) website.

### Media

- The photos used on the home and sign-up page are from This Open-Source site
- The images used for the gallery page were taken from this other open-source site



## Acknowledgements
* Thank the people that provided support through this project.

