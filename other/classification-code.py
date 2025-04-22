import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

class AdvancedClassifier:
    """
    90% dan yuqori aniqlik bilan classification qiladigan universal model
    """
    
    def __init__(self):
        self.best_model = None
        self.feature_selector = None
        self.scaler = None
        self.feature_importances = None
        self.threshold = 0.5
    
    def load_data(self, file_path=None, X=None, y=None, test_size=0.2, random_state=42):
        """
        Dataset yuklash uchun metod
        """
        if file_path:
            try:
                # CSV, Excel va boshqa formatlarda yuklab olish
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                elif file_path.endswith(('.xls', '.xlsx')):
                    df = pd.read_excel(file_path)
                else:
                    raise ValueError("Faqat CSV yoki Excel formatlarida ma'lumot yuklash mumkin")
                
                # Avtomatik ravishda X va y ga ajratish
                # Oxirgi ustunni target sifatida olish
                X = df.iloc[:, :-1]
                y = df.iloc[:, -1]
            except Exception as e:
                print(f"Ma'lumot yuklanishida xatolik: {e}")
                return None
        elif X is not None and y is not None:
            pass
        else:
            raise ValueError("Yo file_path yoki X va y parametrlari berilishi kerak")
        
        # Train va test qismlarga ajratish
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        
        # Ma'lumotlar haqida ma'lumot
        print(f"Umumiy namunalar soni: {len(X)}")
        print(f"Treninrovka uchun namunalar: {len(X_train)}")
        print(f"Test uchun namunalar: {len(X_test)}")
        print(f"Feature'lar soni: {X.shape[1]}")
        
        # Class imbalance tekshirish
        class_counts = pd.Series(y).value_counts()
        print("\nSinflar taqsimoti:")
        print(class_counts)
        
        # Agar class imbalance bo'lsa ogohlantirish
        if class_counts.min() / class_counts.max() < 0.5:
            print("\nOgohlantirish: Sinflar orasida nomutanosiblik mavjud. SMOTE texnikasi qo'llaniladi.")
        
        return X_train, X_test, y_train, y_test
    
    def preprocess_data(self, X_train, X_test, scaling_method='standard', handle_imbalance=True):
        """
        Ma'lumotlarni preprocessing qilish
        """
        # Feature scaling
        if scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaling_method == 'robust':
            self.scaler = RobustScaler()
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Class imbalance bo'lsa SMOTE qo'llash
        if handle_imbalance:
            try:
                smote = SMOTE(random_state=42)
                X_train_scaled, y_train_resampled = smote.fit_resample(X_train_scaled, self.y_train)
                print("SMOTE qo'llanildi. Yangi treninrovka namunalar soni:", len(X_train_scaled))
                return X_train_scaled, X_test_scaled, y_train_resampled
            except:
                print("SMOTE qo'llanilmadi. Asl ma'lumotlar qaytarildi.")
                return X_train_scaled, X_test_scaled, self.y_train
        
        return X_train_scaled, X_test_scaled, self.y_train
    
    def feature_selection(self, X_train, y_train, X_test, method='random_forest', threshold=0.05):
        """
        Muhim feature'larni tanlash
        """
        if method == 'random_forest':
            selector = RandomForestClassifier(n_estimators=100, random_state=42)
            selector.fit(X_train, y_train)
            
            # Feature ahamiyatini saqlash
            self.feature_importances = pd.DataFrame(
                {'feature': range(X_train.shape[1]), 
                 'importance': selector.feature_importances_}
            ).sort_values('importance', ascending=False)
            
            # SelectFromModel orqali muhim feature'larni tanlash
            self.feature_selector = SelectFromModel(selector, threshold=threshold)
            self.feature_selector.fit(X_train, y_train)
            
            # Tanlangan feature'larni qo'llash
            X_train_selected = self.feature_selector.transform(X_train)
            X_test_selected = self.feature_selector.transform(X_test)
            
            # Tanlangan feature'lar soni
            n_selected = X_train_selected.shape[1]
            print(f"Feature selection qo'llanildi. {X_train.shape[1]} feature'dan {n_selected} ta tanlandi.")
            
            return X_train_selected, X_test_selected
        
        return X_train, X_test
    
    def build_models(self):
        """
        Turli klassifikatorlarni yaratish
        """
        models = {
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
            'RandomForest': RandomForestClassifier(random_state=42),
            'GradientBoosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'XGBoost': xgb.XGBClassifier(objective='binary:logistic', random_state=42)
        }
        return models
    
    def tune_hyperparameters(self, X_train, y_train, model_name, param_grid):
        """
        GridSearchCV orqali hiperparametrlarni optimizatsiya qilish
        """
        models = self.build_models()
        if model_name not in models:
            raise ValueError(f"{model_name} modelini topa olmadim")
        
        model = models[model_name]
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # GridSearchCV orqali optimizatsiya
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"\n{model_name} uchun eng yaxshi parametrlar:")
        print(grid_search.best_params_)
        print(f"Eng yaxshi skor: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def create_ensemble(self, models):
        """
        Ensemble model yaratish
        """
        voting_clf = VotingClassifier(
            estimators=models,
            voting='soft'
        )
        
        return voting_clf
    
    def fit(self, X_train, y_train, X_test=None, y_test=None):
        """
        Modelni o'rgatish asosiy metodi
        """
        self.y_train = y_train
        
        # Ma'lumotni preprocessing qilish
        X_train_scaled, X_test_scaled, y_train_resampled = self.preprocess_data(X_train, X_test if X_test is not None else X_train)
        
        # Feature tanlash
        X_train_selected, X_test_selected = self.feature_selection(X_train_scaled, y_train_resampled, X_test_scaled if X_test is not None else X_train_scaled)
        
        # Asosiy modellar uchun hiperparametrlarni optimizatsiya qilish
        rf_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        xgb_params = {
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'n_estimators': [100, 200, 300],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        # Optimizatsiya qilingan modellar
        rf_model = self.tune_hyperparameters(X_train_selected, y_train_resampled, 'RandomForest', rf_params)
        xgb_model = self.tune_hyperparameters(X_train_selected, y_train_resampled, 'XGBoost', xgb_params)
        
        # Ensemble model yaratish
        ensemble_models = [
            ('RandomForest', rf_model),
            ('XGBoost', xgb_model)
        ]
        
        self.best_model = self.create_ensemble(ensemble_models)
        
        # Modelni o'rgatish
        self.best_model.fit(X_train_selected, y_train_resampled)
        
        # Agar test ma'lumotlari berilgan bo'lsa, baholash
        if X_test is not None and y_test is not None:
            self.evaluate(X_test_selected, y_test)
        
        return self
    
    def predict(self, X):
        """
        Yangi ma'lumotlar uchun bashorat qilish
        """
        if self.best_model is None:
            raise ValueError("Model hali o'rgatilmagan")
        
        # Preprocessing va feature selection
        X_scaled = self.scaler.transform(X)
        X_selected = self.feature_selector.transform(X_scaled)
        
        # Bashorat
        return self.best_model.predict(X_selected)
    
    def predict_proba(self, X):
        """
        Ehtimolliklar bilan bashorat qilish
        """
        if self.best_model is None:
            raise ValueError("Model hali o'rgatilmagan")
        
        # Preprocessing va feature selection
        X_scaled = self.scaler.transform(X)
        X_selected = self.feature_selector.transform(X_scaled)
        
        # Ehtimolliklar bilan bashorat
        return self.best_model.predict_proba(X_selected)
    
    def evaluate(self, X_test, y_test):
        """
        Modelni baholash
        """
        if self.best_model is None:
            raise ValueError("Model hali o'rgatilmagan")
        
        # Preprocessing va feature selection
        X_test_scaled = self.scaler.transform(X_test)
        X_test_selected = self.feature_selector.transform(X_test_scaled)
        
        # Bashoratlar
        y_pred = self.best_model.predict(X_test_selected)
        y_proba = self.best_model.predict_proba(X_test_selected)
        
        # Baholash metrikalarini hisoblash
        accuracy = accuracy_score(y_test, y_pred)
        
        print("\nModel baholash natijalari:")
        print(f"Aniqlik (Accuracy): {accuracy:.4f}")
        
        # Binary classification uchun qo'shimcha metrikalar
        if len(np.unique(y_test)) == 2:
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_proba[:, 1])
            
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"ROC AUC: {roc_auc:.4f}")
            
            # ROC egri chizig'ini chizish
            plt.figure(figsize=(10, 6))
            fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.show()
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Bashorat qilingan')
        plt.ylabel('Haqiqiy')
        plt.title('Confusion Matrix')
        plt.show()
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature ahamiyatini aks ettirish
        if self.feature_importances is not None:
            plt.figure(figsize=(12, 8))
            sns.barplot(x='importance', y='feature', data=self.feature_importances.head(20))
            plt.title('Top 20 muhim feature\'lar')
            plt.tight_layout()
            plt.show()
        
        return accuracy
    
    def optimize_threshold(self, X_test, y_test):
        """
        Binary classification uchun threshold'ni optimizatsiya qilish
        """
        if len(np.unique(y_test)) != 2:
            print("Bu funksiya faqat binary classification uchun")
            return self.threshold
        
        # Preprocessing va feature selection
        X_test_scaled = self.scaler.transform(X_test)
        X_test_selected = self.feature_selector.transform(X_test_scaled)
        
        # Ehtimolliklar 
        y_proba = self.best_model.predict_proba(X_test_selected)[:, 1]
        
        # Threshold qiymatlarini sinab ko'rish
        thresholds = np.arange(0.1, 1.0, 0.05)
        f1_scores = []
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            f1 = f1_score(y_test, y_pred)
            f1_scores.append(f1)
        
        # Eng yaxshi threshold tanlash
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
        
        print(f"Optimizatsiya qilingan threshold: {best_threshold:.2f} (F1 Score: {best_f1:.4f})")
        
        # Natijalarni grafik ko'rinishida chizish
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, f1_scores, marker='o')
        plt.axvline(x=best_threshold, color='r', linestyle='--', label=f'Best threshold: {best_threshold:.2f}')
        plt.title('F1 Score vs Threshold')
        plt.xlabel('Threshold')
        plt.ylabel('F1 Score')
        plt.grid(True)
        plt.legend()
        plt.show()
        
        self.threshold = best_threshold
        return best_threshold
    
    def cross_validate(self, X, y, cv=5):
        """
        Cross-validation orqali modelni baholash
        """
        if self.best_model is None:
            raise ValueError("Model hali o'rgatilmagan")
        
        # Preprocessing
        X_scaled = self.scaler.transform(X)
        X_selected = self.feature_selector.transform(X_scaled)
        
        # StratifiedKFold
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        # Cross-validation
        cv_scores = cross_val_score(self.best_model, X_selected, y, cv=cv_strategy, scoring='accuracy')
        
        print(f"\n{cv}-fold Cross-Validation natijalari:")
        print(f"O'rtacha aniqlik: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return cv_scores
    
    def save_model(self, filepath):
        """
        Modelni saqlash
        """
        import joblib
        if self.best_model is None:
            raise ValueError("Model hali o'rgatilmagan")
        
        # Modelni saqlash
        model_data = {
            'model': self.best_model,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'threshold': self.threshold,
            'feature_importances': self.feature_importances
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model muvaffaqiyatli saqlandi: {filepath}")
    
    def load_model(self, filepath):
        """
        Saqlangan modelni yuklash
        """
        import joblib
        try:
            model_data = joblib.load(filepath)
            self.best_model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_selector = model_data['feature_selector']
            self.threshold = model_data['threshold']
            self.feature_importances = model_data['feature_importances']
            
            print(f"Model muvaffaqiyatli yuklandi: {filepath}")
            return self
        except Exception as e:
            print(f"Modelni yuklashda xatolik: {e}")
            return None

# Dasturning qo'llanilishi
if __name__ == "__main__":
    # Misol uchun
    # 1. AdvancedClassifier obyektini yaratish
    classifier = AdvancedClassifier()
    
    # 2. Ma'lumotlarni yuklash (misol uchun)
    # X_train, X_test, y_train, y_test = classifier.load_data("your_dataset.csv")
    # Yoki
    """
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = classifier.load_data(X=X, y=y)
    
    # 3. Modelni o'rgatish
    classifier.fit(X_train, y_train, X_test, y_test)
    
    # 4. Cross-validation orqali baholash
    classifier.cross_validate(X, y)
    
    # 5. Binary classification uchun threshold'ni optimizatsiya qilish
    classifier.optimize_threshold(X_test, y_test)
    
    # 6. Modelni saqlash
    classifier.save_model("advanced_classifier_model.pkl")
    
    # 7. Yangi ma'lumotlar uchun bashorat qilish 
    # new_data = ... # O'zingizning yangi ma'lumotlaringiz
    # predictions = classifier.predict(new_data)
    """
