from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

class ModelBuilder:
    def get_available_models(self, task_type):
        if task_type == "Klasyfikacja":
            return ["Logistic Regression", "Decision Tree", "Random Forest", "SVM", "KNN"]
        else:
            return ["Linear Regression", "Decision Tree", "Random Forest"]

    def get_model_and_params(self, model_name, task_type):
        model = None
        params = {}

        if task_type == "Klasyfikacja":
            if model_name == "Logistic Regression":
                model = LogisticRegression()
                params = {'C': [0.1, 1, 10]}
            elif model_name == "Decision Tree":
                model = DecisionTreeClassifier()
                params = {'max_depth': [None, 5, 10, 20], 'min_samples_split': [2, 5]}
            elif model_name == "Random Forest":
                model = RandomForestClassifier()
                params = {'n_estimators': [50, 100], 'max_depth': [None, 10]}
            elif model_name == "SVM":
                model = SVC(probability=True)
                params = {'C': [0.1, 1], 'kernel': ['linear', 'rbf']}
            elif model_name == "KNN":
                model = KNeighborsClassifier()
                params = {'n_neighbors': [3, 5, 7]}
                
        elif task_type == "Regresja":
            if model_name == "Linear Regression":
                model = LinearRegression()
                params = {}
            elif model_name == "Decision Tree":
                model = DecisionTreeRegressor()
                params = {'max_depth': [None, 5, 10], 'min_samples_split': [2, 5]}
            elif model_name == "Random Forest":
                model = RandomForestRegressor()
                params = {'n_estimators': [50, 100], 'max_depth': [None, 10]}
        
        return model, params

    def train_model(self, X_train, y_train, task_type, model_name, use_grid_search=False):
        
        model, params = self.get_model_and_params(model_name, task_type)
        
        if use_grid_search and params:
            grid = GridSearchCV(model, params, cv=3, n_jobs=-1)
            grid.fit(X_train, y_train)
            return grid.best_estimator_, grid.best_params_
        else:
            model.fit(X_train, y_train)
            return model, {}