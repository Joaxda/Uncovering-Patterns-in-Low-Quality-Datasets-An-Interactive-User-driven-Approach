# Master_thesis_project

To Run:

python -m venv .venv                            # create the virtual environment \
.venv\Scripts\pip install -r requirements.txt   # install all packages \
.venv\Scripts\pip install new_package           # install a new package \
.venv\Scripts\pip freeze > requirements.txt     # update requirements with new packages \

cd ./Src                                        # Change directory to /Src \
python app.py                                   # run the app and open it in the browser (localhost) \

ABSTRACT:

This thesis presents an interactive, two-stage pipeline designed to address the challenges of analysing complex, real-world datasets characterised by data quality issues
such as mixed data types and significant missing values. The first stage focuses on essential data preparation steps, including handling missing values using automated imputation, detecting and mitigating outliers, encoding mixed-type data, and normalisation. 
The second stage facilitates data analysis through visualisation, incorporating feature relevance assessment, dimensionality reduction, and cluster analysis to
identify patterns and attribute relationships that impact a user-chosen target variable. The evaluation included two case studies on datasets with practical data quality issues and a user study employing the User Experience Questionnaire - Short (UEQ-S),
which demonstrated the pipeline’s effectiveness in uncovering meaningful structures and providing an accessible workflow for users. The user study results highlight its strength in pragmatic usability, contributing to enhanced decision-making processes.
