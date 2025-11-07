import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


class DataGraphic:

    def count_plot(self, df, colName):
        return sns.countplot(data=df, x=colName)

    def histogram_col(self, df, colName):
        return plt.hist(x=df[colName])

    def matrix_compare(self, df, cols):
        return px.scatter_matrix(df, dimensions=cols)
