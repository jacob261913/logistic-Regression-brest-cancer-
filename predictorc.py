import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib

data =pd.read_csv('cancer_data.csv')
# Preprocess the data
label_encoder = LabelEncoder()

data["diagnosis_encoded"] = label_encoder.fit_transform(data["diagnosis"])
X =data[["radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"]]
Y = data["diagnosis_encoded"]

model = LogisticRegression(max_iter=10000)
model.fit(X, Y)
print("Coeff=", model.coef_)
print("Intercept=", model.intercept_)

#save the model
joblib.dump(model, 'cancer_prediction_model.pkl')
print("Model saved successfully as 'cancer_prediction_model.pkl'")