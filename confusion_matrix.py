from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import torch
import numpy as np


def confusionMatrix(y_pred,y_true,save_path='confusion_matrix.png'):
    # classes = ('Anger', 'Disgust', 'Fear', 'Happy', 'Neutral','Sad', 'Surprise') #RAF
    classes = ('Surprise','Fear','Disgust','Happiness','Sadness','Anger','Neutral')  #AffectNet
    y_pred = torch.cat(y_pred).cpu().numpy()
    y_true = torch.cat(y_true).cpu().numpy()
    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred,normalize='true')
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *len(classes), index = [i for i in classes],
                        columns = [i for i in classes])
    plt.figure(figsize = (7,7))
    sn.heatmap(df_cm, annot=True,cmap="Blues")
    plt.ylabel('True Label')
    plt.xlabel('Prediced Label') 
    plt.savefig(save_path)
