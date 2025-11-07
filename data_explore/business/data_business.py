from sklearn.compose import ColumnTransformer
from data_explore.data.data_explore import DataExplore
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np


class DataBusiness:

    def __init__(self):
        self.explore = DataExplore()

    def normalization_data(self, df):
        print("\n========== INÍCIO DA NORMALIZAÇÃO ==========\n")

        # 1) Corrigir inconsistências
        print("🔧 Etapa 1: Correção de inconsistências...")
        df = self.data_inconsistencies(df)
        print("✅ Inconsistências corrigidas. Dimensões:", df.shape)

        # 2) Tratar valores faltantes
        print("\n🧼 Etapa 2: Tratamento de valores faltantes...")
        df = self.data_missing(df)
        print("✅ Missing values tratados. Dimensões:", df.shape)

        # 3) Separar features e target
        print("\n🎯 Etapa 3: Separando features e target...")
        X, y = self.split_features_and_target(df)
        print("✅ Features (X):", X.shape, " | Target (y):", y.shape)

        # 4) Split treino e teste
        print("\n✂️ Etapa 4: Realizando train_test_split...")
        X_train, X_test, y_train, y_test = self.split_train_test(X, y)
        print("✅ X_train:", X_train.shape, "| X_test:", X_test.shape)
        print("✅ y_train:", y_train.shape, "| y_test:", y_test.shape)

        # 5) Escalar treino e teste
        print("\n📏 Etapa 5: Escalonando dados (train e test)...")
        X_train_scaled, X_test_scaled, scaler = self.scale_train_test(
            X_train, X_test)
        print("✅ Escalonamento concluído!")
        print("   - X_train_scaled:", X_train_scaled.shape)
        print("   - X_test_scaled :", X_test_scaled.shape)
        print("   - Scaler usado  :",
              scaler.__class__.__name__ if scaler else None)

        print("\n========== FIM DA NORMALIZAÇÃO ==========\n")

        return (X_train_scaled, X_test_scaled), (y_train, y_test), scaler

    def data_inconsistencies(self, df):
        # Valores inconsistentes, correção
        colsNames = self.explore.name_of_colluns(df)
        for colName in set(colsNames):
            # Senão for valores binários (0 e 1)
            valores_unicos = set(df[colName].dropna().unique())
            if not valores_unicos.issubset({0, 1}):
                data: pd.DataFrame = self.explore.identify_zeros(df, colName)

                if not data.empty:
                    linhasWithZeros = data.shape[0]
                    recordAccount = df.shape[0]
                    percent = (linhasWithZeros / recordAccount) * 100
                    # meanCol
                    if percent < 5:
                        df = self.explore.set_mean_collumn(df, colName)
                    elif percent >= 5 and percent <= 10:
                        df = self.explore.delete_rows(df, data)
                    else:
                        df = self.explore.delete_coll(df, colName)
        return df

    def data_missing(self, df):
        # Valores Faltantes/Nan/Null, correção
        valueNull = self.explore.values_exist(df)
        if not valueNull.empty:
            for colName in set(valueNull):
                linhasWithNull = valueNull.shape[0]
                recordAccount = df.shape[0]
                percent = (linhasWithNull / recordAccount) * 100
                # meanCol
                if percent < 5:
                    df = self.explore.set_mean_collumn(df, colName)
                elif percent >= 5 and percent <= 10:
                    df = self.explore.delete_rows(df, valueNull)
                else:
                    df = self.explore.delete_coll(df, valueNull)
        return df

    def split_features_and_target(self, df):
        target_col = df.columns[-1]
        X = df.drop(columns=[target_col])
        y = df[target_col]
        return X, y

    def split_train_test(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y if y.nunique() <= 20 else None
        )
        return X_train, X_test, y_train, y_test

    def scale_train_test(self, X_train, X_test):
        numeric_cols = X_train.select_dtypes(
            include=['int64', 'float64']).columns.tolist()

        numeric_cols = [c for c in numeric_cols if 'id' not in c.lower()]

        if len(numeric_cols) == 0:
            return X_train.copy(), X_test.copy(), None

        # Detecta outliers
        has_outliers = (X_train[numeric_cols].skew().abs() > 2).any()
        scaler = RobustScaler() if has_outliers else StandardScaler()

        # Aplica scaling
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()

        X_train_scaled[numeric_cols] = scaler.fit_transform(
            X_train[numeric_cols])
        X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

        return X_train_scaled, X_test_scaled, scaler
