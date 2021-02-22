import matplotlib.pyplot as plt
import seaborn as sns

def barplot_res(dataframe, percentage, metric, y_name):
    selector = dataframe.apply(lambda x: percentage in x["tags"], axis=1)
    data_percentage=dataframe.loc[selector, :]
    data_percentage["tags"]=data_percentage["tags"].apply(lambda x: str(x))
    res_percentage=data_percentage.groupby(by='tags')[metric].mean().reset_index()
    res_percentage['model']=res_percentage['tags'].apply(lambda x: eval(x)[1])
    ax=sns.barplot(data=res_percentage, x='model', y=metric)
    plt.ylim((0.5, 0.9))
    ax.set(xlabel='Model', ylabel=y_name, title =percentage)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=60)
    return ax, res_percentage

def lineplot_res(dataframe, y_name, metric, gbc_result):
    data=dataframe.copy()
    data['percentage']=data['tags'].apply(lambda x: x[0])
    data=data[~data['percentage'].apply(lambda x: 'd' in str(x))]
    data['model']=data['tags'].apply(lambda x: x[1])
    result=data.groupby(by=['model','percentage'])[metric].mean().reset_index()
    ax=sns.lineplot(data=result, x='percentage', y=metric, hue='model')
    ax.set(xlabel='Training set size', ylabel=y_name)
    plt.axhline(y=gbc_result, color='brown', linestyle='--')
    return ax