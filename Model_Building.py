import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


class model_class:
    def model_build(self):
        df = pd.read_excel("customer_churn_large_dataset.xlsx")
        
        #handling Categorical featurews
        dummy_cols = pd.get_dummies(df[['Gender','Location']],drop_first=True)

        df = pd.concat([df, dummy_cols], axis=1)
        df.drop(["Gender",'Location'], axis=1, inplace=True)

        # Customerid and name features are not required because all the records are unique
        df.drop(["CustomerID","Name"], axis=1,inplace=True)
        
        #spliting the data
        X = df.drop(['Churn'],axis=1)
        Y = df.loc[:,["Churn"]]
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

        # Hyperparameter tuning

        '''param_grid = {"n_estimators": [10, 50, 100, 130], "criterion": ['gini', 'entropy'],
                               "max_depth": range(2, 4, 1), "max_features": ['auto', 'log2']}
        
        grid = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=2,  verbose=3)
        grid.fit(X_train, y_train)'''

        model_rf=RandomForestClassifier()
        model_rf.fit(X_train,y_train)

        return model_rf



    