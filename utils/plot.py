import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm


def plot_conf_mat(conf_mat):
    fig = plt.figure(figsize=(5,4))
    df = pd.DataFrame(conf_mat, index=['True', 'False'], columns=['True', 'False'])
    sns.set(font_scale=1.2)
    ax = sns.heatmap(df, annot=True,  cmap=cm.Blues, cbar=False, fmt='g')
    ax.set_title('Confusion matrix', fontweight='bold', fontsize='large')
    ax.set_xlabel('Actual',fontweight='bold')
    ax.set_ylabel('Predicted', fontweight='bold')
    plt.tight_layout()
    return fig
