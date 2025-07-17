Medical Insurance Cost Estimator

MY goal: Predict the medical insurance charge based on the patient data includes age, smoking, BMI etc.

Data Collection : I have used the dataset from Kaggle insurance.csv for my analysis. The dataset includes attributes like age, bmi, sex, children, smoker, region, charges.

Data Cleaning : clean the dataset, if the column has null or missing values in this we identity and clean the data.

Data Analysis(EDA) : used python packages pandas, seaborn and matplotlib for analysis and plots to compare how does each feature will affect the medical charges.

preprocess the insurance dataset step by step so it's clean and ready for modeling.

Feature Engineering: create interaction features these will help how the effect of age or bmi on charges will differ between smokers and non smokers.

Model Training: In this step I have compared between regression models to find efficient method to build the model by comparing the RMSE, MAE AND R2 Score. So, from all the compared regression models Random forest algorithm has less RMSE & MAE values and highest R2 score. So I have considered Random forest regression model for predicting the medical insurance charges.

Analysis results:
               Model         RMSE          MAE  R2 Score
1   Ridge Regression  4562.451452  2760.083595  0.865919
4      Random Forest  4562.864104  2549.970430  0.865894
2   Lasso Regression  4574.035822  2757.833674  0.865237
0  Linear Regression  4574.123734  2757.759204  0.865232
3      Decision Tree  4856.010615  2872.286134  0.848109

Run the Model : I have used FastAPI to run the model.

Install Docker to run this API on docker, run the below command
docker build -t insurance-api .
Here, I built a Docker image with the tag insurance-api
-- once built run the container using this command;
docker run -p 8000:8000 insurance-api
