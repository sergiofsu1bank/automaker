import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


from data_explore.data.data_loader import DataLoader
from data_explore.data.data_explore import DataExplore
from data_explore.data.data_graphic import DataGraphic
from data_explore.business.data_business import DataBusiness


class Main:

    def __init__(self):
        self.loader = DataLoader()
        self.explore = DataExplore()
        self.graphic = DataGraphic()
        self.business = DataBusiness()

    def run(self):
        dataset = "credit_data.csv"
        df = self.loader.load(dataset)
        return df

    def explore_data(self, df):
        print('-' * 50)
        data = self.explore.first_lines(df)
        print(data)
        print('-' * 50)
        data = self.explore.last_lines(df)
        print(data)
        print('-' * 50)
        data = self.explore.quantity_currences(df)
        print(data)
        print('-' * 50)
        data = self.explore.quantity_line_coll(df)
        print(data)
        print('-' * 50)
        data = self.explore.basic_summary(df)
        colName = "default"
        data = self.explore.unique_values(df, colName)
        print(data)
        print('-' * 50)
        data = self.explore.unique_values_count(df, colName)
        print(data)

    def explore_graphic(self, df):
        print('-' * 50)
        colName = "default"
        data = self.graphic.count_plot(df, colName)
        plt.show()
        print('-' * 50)

        colName = "age"
        data = self.graphic.histogram_col(df, colName)
        plt.show()
        print('-' * 50)

        cols = ["income", "age", "loan", "default"]
        data = self.graphic.matrix_compare(df, cols)
        data.show()
        print('-' * 50)


if __name__ == "__main__":
    app = Main()
    df = app.run()
    # app.exploreData(df)
    # app.exploreGraphic(df)
    X_train_norm, X_test_norm, y_train, y_test = app.business.normalization_data(
        df)
