import joblib 
import pandas as pd

model = joblib.load('model.pkl')
columns = joblib.load('columns.pkl')

importances = pd.Series(model.feature_importances_, index=columns)
top_features = importances.sort_values(ascending=False).head(5)

print(top_features)

import ollama

def explain_results(top_features, accuracy):
    prompt = f"""
    I built a machine learning model to predict customer churn.
    Accuracy: {accuracy:.2%}
    Top 5 most important  features: {top_features.to_dict()}
    
    Explain these results in plain English for a business audience.
    Suggest 2-3 actions the business could take based on this.
    """

    response = ollama.chat(model='llama3.2', messages=[
        {'role': 'user', 'content': prompt}
    ])

    return response['message']['content']

explanation = explain_results(top_features, accuracy=0.785)
print("\n==== AI EXPLANATION ====")
print(explanation)

