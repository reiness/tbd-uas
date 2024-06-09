from mrjob.job import MRJob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def fill_na_with_mean(df, column):
    for potability in df['Potability'].unique():
        mean_value = df[df['Potability'] == potability][column].mean(skipna=True)
        df.loc[(df['Potability'] == potability) & (df[column].isna()), column] = mean_value

class WaterPotabilityRegressor(MRJob):
    
    def mapper(self, _, line):
        if line.startswith('ph'):  # Assuming the first column name in the CSV is 'ph'
            return
        
        data = line.strip().split(',')
        
        # Parsing the input values from the line
        ph = float(data[0]) if data[0] else np.nan
        hardness = float(data[1]) if data[1] else np.nan
        solids = float(data[2]) if data[2] else np.nan
        chloramines = float(data[3]) if data[3] else np.nan
        sulfate = float(data[4]) if data[4] else np.nan
        conductivity = float(data[5]) if data[5] else np.nan
        organic_carbon = float(data[6]) if data[6] else np.nan
        trihalomethanes = float(data[7]) if data[7] else np.nan
        turbidity = float(data[8]) if data[8] else np.nan
        potability = int(data[9])
        
        features = [ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]
        yield None, (potability, *features)

    def reducer(self, _, target_features_pairs):
        df = pd.DataFrame(target_features_pairs, columns=[
            'Potability', 'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
            'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'
        ])
        
        # Handling missing values using the provided fill_na_with_mean function
        columns_to_fill = ['ph', 'Sulfate', 'Trihalomethanes']
        for column in columns_to_fill:
            fill_na_with_mean(df, column)
        
        # Separating features and target
        y = df['Potability']
        X = df.drop(columns=['Potability'])
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Standardize features
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        
        # Train and evaluate RandomForestClassifier
        model = RandomForestClassifier(random_state=2)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, pred)
        
        # Emit cross-validation scores and test accuracy
        yield None, {
            'Cross-validation scores': cv_scores.tolist(),
            'Mean cross-validation score': np.mean(cv_scores),
            'Test Accuracy': accuracy
        }

if __name__ == '__main__':
    WaterPotabilityRegressor.run()
