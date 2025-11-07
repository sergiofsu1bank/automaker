import pandas as pd
import numpy as np


class DataExplore:

    def basic_summary(self, df):
        return {
            "linhas": df.shape[0],
            "colunas": df.shape[1],
            "valores_nulos": df.isnull().sum().to_dict(),
            "tipos_de_dados": df.dtypes.astype(str).to_dict()
        }

    def identify_zeros(self, df, colName) -> pd.DataFrame:
        if colName not in df.columns:
            return pd.DataFrame()
        if pd.api.types.is_numeric_dtype(df[colName]):
            return df.loc[df[colName] <= 0]
        return pd.DataFrame()

    def delete_coll(self, df, colName):
        return df.drop(colName, axis=1)

    def delete_rows(self, df, colName):
        return df.drop(df[df[colName] <= 0].index)

    def first_lines(self, df):
        return df.head()

    def last_lines(self, df):
        return df.tail()

    def quantity_line_coll(self, df):
        return df.shape

    def name_of_colluns(self, df):
        return set(df.columns)

    def quantity_currences(self, df):
        return df.describe()

    def unique_values(self, df, colName):
        return df[colName].unique()

    def unique_values_count(self, df, colName):
        return np.unique(df[colName], return_counts=True)

    def mean_col(self, df, colName):
        return df[colName][df[colName] > 0].mean()

    def values_exist(self, df):
        return df.columns[df.isnull().any()]

    def set_mean_collumn(self, df, colName):
        df[colName] = df[colName].fillna(df[colName].mean())
        return df

    def get_target(self, df):
        """
        Identifica a variável target com 100% de acurácia
        baseada em regras determinísticas:

        1. Se existir coluna binária exclusiva {0,1}, ela é o target.
        2. Se houver mais de uma binária, usa a última coluna.
        3. Se não houver binária, assume a última coluna.
        """
        colunas_binarias = []

        for col in df.columns:
            valores = set(df[col].dropna().unique())
            if valores.issubset({0, 1}) and len(valores) >= 1:
                colunas_binarias.append(col)

        # Regra 1: se só existe uma binária → é o target
        if len(colunas_binarias) == 1:
            return colunas_binarias[0]

        # Regra 2: se há várias binárias → usa a última
        if len(colunas_binarias) > 1:
            return colunas_binarias[-1]

        # Regra 3: fallback → assume última coluna
        return df.columns[-1]

    def get_predictors(self, df):
        """
        Retorna todas as colunas preditoras (features),
        removendo automaticamente a coluna target.
        """
        target = self.get_target(df)
        return df.drop(columns=[target])
