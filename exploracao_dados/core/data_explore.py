
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

    def firstLines(self, df):
        return df.head()

    def lastLines(self, df):
        return df.tail()

    def quantityLineColl(self, df):
        return df.shape

    def nameOfColluns(self, df):
        return {df.columns}

    def quantityCurrences(self, df):
        return df.describe()

    def uniqueValues(self, df, colName):
        return df[colName].unique()

    def uniqueValuesCount(self, df, colName):
        return np.unique(df[colName], return_counts=True)
