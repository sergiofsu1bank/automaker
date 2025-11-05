import matplotlib.pyplot as plt
import pandas as pd

from exploracao_dados.core.data_loader import DataLoader
from exploracao_dados.core.data_explore import DataExplore
from exploracao_dados.core.data_graphic import DataGraphic


class Main:
    def __init__(self):
        self.loader = DataLoader()
        self.explore = DataExplore()
        self.graphic = DataGraphic()

    def run(self):
        dataset = "credit_data.csv"
        df = self.loader.load(dataset)
        return df

    def exploreData(self, df):
        print('-' * 50)
        data = self.explore.firstLines(df)
        print(data)
        print('-' * 50)
        data = self.explore.lastLines(df)
        print(data)
        print('-' * 50)
        data = self.explore.quantityCurrences(df)
        print(data)
        print('-' * 50)
        data = self.explore.quantityLineColl(df)
        print(data)
        print('-' * 50)
        data = self.explore.basic_summary(df)
        colName = "default"
        data = self.explore.uniqueValues(df, colName)
        print(data)
        print('-' * 50)
        data = self.explore.uniqueValuesCount(df, colName)
        print(data)

    def exploreGraphic(self, df):
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
    app.exploreGraphic(df)
