# CVD Predictor - A Predictive Classification Model for Determining Risk of Heart Disease

[CVD Predictor](https://cvd-predictor-a8ce111af1d1.herokuapp.com/) is a machine-learning (ML) project using a publically available dataset to determine whether a ML pipeline could be built in order to predict whether a patient is at risk of heart disease. This was achieved by using a classification task, using the HeartDisease attribute from the dataset as the target and the remaining attributes as features.

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

[Back to top](#table-of-contents)

## Business Requirements
* Cardiovascular diseases are the number 1 cause of death globally, accounting for 31% of all deaths worldwide. People with cardiovascular disease or who are at high risk of disease need early detection and management. A fictional organisation has requested a data practitioner to analyse a dataset of patients from a number of different hospitals in order to determine what factors can be attributed to a high risk of disease and whether patient data can accurately predict risk of heart disease.

* Business Requirement 1 - The client is interested in which attributes correlate most closely with heart disease, ie what are the most common risk factors?
* Business Requirement 2 - The client is interested in using patient data to predict whether or not a patient is at risk of heart disease.

[Back to top](#table-of-contents)

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

[Back to top](#table-of-contents)

## The rationale to map the business requirements to the Data Visualizations and ML tasks
* **Business Requirement 1**: Data Visualisation and Correlation study
    - We need to perform a correlation study to determine which features correlate most closely to the target.
    - A Pearson's correlation will indicate linear relationships between numerical variables.
    - A Spearman's correlation will measure the monotonic relationships between variables.
    - A Predictive Power Score study can also be used to determine relationships between attributes regardless of data type (6/11 features are categorical).
    - This will be carried out during the **Data Visualization, Cleaning, and Preparation** Epic (see Epics & User Stories).

* **Business Requirement 2**: Classification Model
    - We need to predict whether a patient is at risk of heart disease or not.
    - Therefore we need to build a binary classification model.
    - A conventional machine learning pipeline will be able to map the relationships between the features and target.
    - Extensive hyperparameter optimisation will give us the best chance at a highly accurate prediction.
    - This will be carried out during the **Model Training, Optimization and Validation** Epic (see Epics & User Stories).

[Back to top](#table-of-contents)

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

[Back to top](#table-of-contents)

## Epics and User Stories
* The project was split into 5 Epics based upon the Data Visualisation and Machine Learning tasks and within each of these, user stories were set out to enable an agile methodology.

### Epic - Information Gathering and Data Collection
* **User Story** - As a data analyst, I can import the dataset from Kaggle so that I can save the data in a local directory.
* **User Story** - As a data analyst, I can load a saved dataset so that I can analyse the data to gain insights on what further tasks may be required.

### Epic - Data Visualization, Cleaning, and Preparation
* **User Story** - As a data scientist, I can visualise the dataset so that I can interpret which attributes correlate most closely with heart disease (**Business Requirement 1**).
* **User Story** - As a data analyst, I can evaluate the dataset to determine what data cleaning tasks need to be carried out.
* **User Story** - As a data analyst, I can impute or drop missing data to prepare the dataset for a ML model.
* **User Story** - As a data analyst, I can determine whether the target requires balancing in order to ensure the ML is not fed imbalanced data.
* **User Story** - As a data scientist, I can carry out feature engineering to best transform the data for the ML model.

### Epic - Model Training, Optimization and Validation
* **User Story** - As a data scientist, I can split the data into a train and test set to prepare it for the ML model.
* **User Story** - As a data engineer, I can fit a ML pipeline with all the data to prepare the ML model for deployment.
* **User Story** - As a data engineer, I can determine the best algorithm for predicting heart disease to use in the ML model (**Business Requirement 2**).
* **User Story** - As a data engineer, I can carry out an extensive hyperparameter optimisation to ensure the ML model gives the best results (**Business Requirement 2**).
* **User Story** - As a data scientist, I can determine the best features from the ML pipeline to determine whether the ML model can be optimised further (**Business Requirement 2**).
* **User Story** - As a data scientist, I can evaluate the ML model's performance to determine whether it can successfully predict heart disease (**Business Requirement 2**).

### Epic - Dashboard Planning, Designing, and Development
* **User Story** - As a non-technical user, I can view a project summary that describes the project, dataset and business requirements to understand the project at a glance.
* **User Story** - As a non-technical user, I can view the project hypotheses and validations to determine what the project was trying to achieve and whether it was successful.
* **User Story** - As a non-technical user, I can enter unseen data into the model and receive a prediction (**Business Requirement 2**).
* **User Story** - As a technical user, I can view the correlation analysis to see how the outcomes were reached (**Business Requirement 1**).
* **User Story** - As a technical user, I can view all the data to understand the model performance and see statistics related to the model (**Business Requirement 2**).
* **User Story** - As a non-technical user, I can view the project conclusions to see whether the model was successful and if the business requirements were met.

### Epic - Dashboard Deployment and Release
* **User Story** - As a user, I can view the project dashboard on a live deployed website.
* **User Story** - As a technical user, I can follow instructions in the readme to fork the repository and deploy the project for myself.

[Back to top](#table-of-contents)

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

[Back to top](#table-of-contents)

## Technologies Used

The technologies used throughout the development are listed below:

### Languages

* [Python](https://www.python.org/)

### Python Packages

* [Pandas](https://pandas.pydata.org/docs/index.html) - Open source library for data manipulation and analysis.
* [Numpy](https://numpy.org/doc/stable/index.html) - Adds support for large, multi-dimensional arrays and matrices, and high-level mathematical functions.
* [YData Profiling](https://docs.profiling.ydata.ai/latest/) - For data profiling and exploratory data analysis.
* [Matplotlib](https://matplotlib.org/) - Comprehensive library for creating static, animated and interactive visualisations.
* [Seaborn](https://seaborn.pydata.org/) - Another data visualisation library for drawing attractive and informative statistical graphics.
* [Pingouin](https://pingouin-stats.org/build/html/index.html) - Open source statistical package for simple yet exhaustive stats functions.
* [Feature-engine](https://feature-engine.trainindata.com/en/latest/) - Library with multiple transformers to engineer and select features for machine learning models.
* [ppscore](https://pypi.org/project/ppscore/) - Library for detecting linear or non-linear relationships between two features.
* [scikit-learn](https://scikit-learn.org/stable/) - Open source machine learning library that features various algorithms for training a ML model.
* [SciPy](https://scipy.org/) - Library used for scientific computing and technical computing.
* [XGBoost](https://xgboost.readthedocs.io/en/stable/) - Optimised distributed gradient boosting library.
* [Imbalanced-learn](https://imbalanced-learn.org/stable/) - Provides tools for dealing with classification problems with imbalanced classes.
* [Joblib](https://joblib.readthedocs.io/en/stable/) - Provides tools for lightweight pipelining, e.g. caching output values.

### Other Technologies

* [Git](https://git-scm.com/) - For version control
* [GitHub](https://github.com/) - Code repository and GitHub projects was used as a Kanban board for Agile development
* [Heroku](https://heroku.com) - For application deployment
* [VSCode](https://code.visualstudio.com/) - IDE used for development

[Back to top](#table-of-contents)

## Testing
### Manual Testing

#### User Story Testing
* Dashboard was manually tested using user stories as a basis for determining success.
* Jupyter notebooks were reliant on consecutive functions being successful so manual testing against user stories was deemed irrelevant.

*As a non-technical user, I can view a project summary that describes the project, dataset and business requirements to understand the project at a glance.*

| Feature | Action | Expected Result | Actual Result |
| --- | --- | --- | --- |
| Project summary page | Viewing summary page | Page is displayed, can move between sections on page | Functions as intended |

---

*As a non-technical user, I can view the project hypotheses and validations to determine what the project was trying to achieve and whether it was successful.*

| Feature | Action | Expected Result | Actual Result |
| --- | --- | --- | --- |
| Project hypotheses page | Navigate to page | Clicking on navbar link in sidebar navigates to correct page | Functions as intended |

---

*As a non-technical user, I can enter unseen data into the model and receive a prediction (Business Requirement 2).*

| Feature | Action | Expected Result | Actual Result |
| --- | --- | --- | --- |
| Prediction page | Navigate to page | Clicking on navbar link in sidebar navigates to correct page | Functions as intended |
| Enter live data | Interact with widgets | All widgets are interactive, respond to user input | Functions as intended |
| Live prediction | Click on 'Run Predictive Analysis' button | Clicking on button displays message on page with prediction and % chance | Functions as intended |

---

*As a technical user, I can view the correlation analysis to see how the outcomes were reached (Business Requirement 1).*

| Feature | Action | Expected Result | Actual Result |
| --- | --- | --- | --- |
| Correlation Study page | Navigate to page | Clicking on navbar link in sidebar navigates to correct page | Functions as intended |
| Correlation data | Tick correlation results checkbox | Correlation data is displayed on dashboard | Functions as intended |
| PPS Heatmap | Tick PPS heatmap checkbox | Heatmap is displayed on dashboard | Functions as intended |
| Feature Correlation | Select feature from dropdown box | Relevant countplot is displayed | Functions as intended |
| Parallel Plot | Tick parallel plot checkbox | Parallel plot is displayed on dashboard, is interactive | Functions as intended |

---

*As a technical user, I can view all the data to understand the model performance and see statistics related to the model (Business Requirement 2)*

| Feature | Action | Expected Result | Actual Result |
| --- | --- | --- | --- |
| Model performance page | Navigate to page | Clicking on navbar link in sidebar navigates to correct page | Functions as intended |
| Success metrics | View page | Success metrics outlined in business case are displayed | Functions as intended |
| ML Pipelines | View page | Both ML Pipelines from Jupyter notebooks are displayed | Functions as intended |
| Feature Importance | View page | Most important features are plotted and displayed | Functions as intended |
| Model Performance | View page | Confusion matrix for train and test sets are displayed | Functions as intended |

---

### Validation
All code in the app_pages and src directories was validated as conforming to PEP8 standards using CodeInstitute's PEP8 Linter.
* Some files had warnings due to 'line too long', however these were related to long strings when writing to the dashboard.
* These warnings were ignored as it did not effect the readability of any functions.

### Automated Unit Tests
No automated unit tests have been carried out at this time.

[Back to top](#table-of-contents)

## Issues
### Heroku Slug Size and XGBoost
* Development of the classification model initially took place using XGBoost version 2.0.3 (see outputs/ml_pipeline/classification_model/v1).
* The model gave excellent results with 93% recall for heart disease on both train and test sets, and precision on no heart disease of 90% for the train set and 85% for the test set.
* Unfortunately, upon deployment of the dashboard on Heroku using this model failed as the slug size was too large.

![Image showing failed Heroku deployment due to too large a slug size](assets/images/slug_size_too_large.png)

* A number of attempts were made to reduce the slug size, including adding all unnecessary files for deployment to the ```.slugignore``` file, removing requirements from ```requirements.txt``` so that only packages necessary for deployment were present and purging the build cache using the Heroku CLI.
* These attempts were unsuccessful, and the slug size remained too large.
* During further analysis of the build log, I noticed that the size of XGBoost was very large:

![Image showing size of the xgboost package downloaded for deployment](assets/images/xgboost_download.png)

* Therefore, I attempted to roll back to an older version of XGBoost (version 1.7.6) in order to reduce it's size.
* This was successful, and the app was now able to be deployed to Heroku, however my model no longer gave the same performance.
* I carried out hyperparameter optimisation again, resulting in v2 of the model, however the performance was still not as good as achieved in v1.
* I had to accept this trade-off in performance, due to the limitations of Heroku deployment.

[Back to top](#table-of-contents)

## Unfixed Bugs
* At the time of writing, there are no unfixed bugs within the project.

[Back to top](#table-of-contents)

## Deployment
### Heroku

* The App live link is: [CVD Predictor](https://cvd-predictor-a8ce111af1d1.herokuapp.com/)

The project was deployed to Heroku using the following steps:

1. Within your working directory, ensure there is a setup.sh file containing the following:
```
mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```
2. Within your working directory, ensure there is a runtime.txt file containing a [Heroku-20](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack supported version of Python.
```
python-3.10.12
```
3. Within your working directory, ensure there is a Procfile file containing the following:
```
web: sh setup.sh && streamlit run app.py
```
4. Ensure your requirements.txt file contains all the packages necessary to run the streamlit dashboard.
5. Update your .gitignore and .slugignore files with any files/directories that you do not want uploading to GitHub or are unnecessary for deployment.
6. Log in to [Heroku](https://id.heroku.com/login) or create an account if you do not already have one.
7. Click the **New** button on the dashboard and from the dropdown menu select "Create new app".
8. Enter a suitable app name and select your region, then click the **Create app** button.
9. Once the app has been created, navigate to the Deploy tab.
10. At the Deploy tab, in the Deployment method section select **GitHub**.
11. Enter your repository name and click **Search**. Once it is found, click **Connect**.
12. Navigate to the bottom of the Deploy page to the Manual deploy section and select main from the branch dropdown menu.
13. Click the **Deploy Branch** button to begin deployment.
14. The deployment process should happen smoothly if all deployment files are fully functional. Click the button **Open App** at the top of the page to access your App.
15. If the build fails, check the build log carefully to troubleshoot what went wrong.

[Back to top](#table-of-contents)

## Forking and Cloning
If you wish to fork or clone this repository, please follow the instructions below:

### Forking
1. In the top right of the main repository page, click the **Fork** button.
2. Under **Owner**, select the desired owner from the dropdown menu.
3. **OPTIONAL:** Change the default name of the repository in order to distinguish it.
4. **OPTIONAL:** In the **Description** field, enter a description for the forked repository.
5. Ensure the 'Copy the main branch only' checkbox is selected.
6. Click the **Create fork** button.

### Cloning
1. On the main repository page, click the **Code** button.
2. Copy the HTTPS URL from the resulting dropdown menu.
3. In your IDE terminal, navigate to the directory you want the cloned repository to be created.
4. In your IDE terminal, type ```git clone``` and paste the copied URL.
5. Hit Enter to create the cloned repository.

### Installing Requirements
**WARNING:** The packages listed in the requirements.txt file are limited to those necessary for the deployment of the dashboard to Heroku, due to the limit on the slug size.

In order to ensure all the correct dependencies are installed in your local environment, run the following command in the terminal:

    pip install -r full-requirements.txt

[Back to top](#table-of-contents)

## Credits 

### Content 

#### Exploratory Data Analysis Notebook
* The code for the histogram/QQ plots were taken from the Code Institute "Churnometer" walkthrough project.
* The code for the PPS heatmap function was taken from the Code Institute "Exploratory Data Analysis Tools" module.

#### Data Cleaning Notebook
* The custom function for checking the effect of data cleaning on distribution was taken from the Code Institute "Data Analytics Packages - ML: feature-engine" module.

#### Feature Engineering Workbook
* The custom function for analysing transformations during feature engineering was taken from the Code Institute "Data Analytics Packages - ML: feature-engine" module.

#### Modelling And Evaluation Notebook
* Abhishek Thakur's ["Approaching (Almost) Any Machine Learning Problem"](https://www.linkedin.com/pulse/approaching-almost-any-machine-learning-problem-abhishek-thakur/) post on LinkedIn and Jason Brownlee's [Machine Learning Mastery](https://machinelearningmastery.com/hyperparameters-for-classification-machine-learning-algorithms/) website were both used to help define the hyperparameter values used for optimisation.
* The custom function for carrying out hyperparameter optimisation was taken from the Code Institute "Data Analytics Packages - ML: Scikit-learn" module.
* The custom function for displaying the confusion matrix and analysing model performance was taken from the Code Institute "Data Analytics Packages - ML: Scikit-learn" module.

#### Streamlit Dashboard
* The multi-page class was taken from the Code Institute "Data Analysis & Machine Learning Toolkit" streamlit lessons.

[Back to top](#table-of-contents)

## Acknowledgements
* Thanks to my mentor Mo Shami, for his support and guidance on the execution of the project

[Back to top](#table-of-contents)
