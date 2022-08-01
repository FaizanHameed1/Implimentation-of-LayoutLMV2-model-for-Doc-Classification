from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import parser
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

train_config=parser.train_config

##classes
class_path=train_config["class_path"]

with open(class_path,'r') as file:
    classes=file.read().split('\n')

class Results():
    def __init__(self,y_pred,y_labels):
        self.y_pred = y_pred
        self.y_labels = y_labels

    def tensors_to_int_lists(self):

        #extracting the labels from from tensors and making a list
        label0=[]
        label1=[]
        labels=[] #extracted label

        for val0 in self.y_labels:
        #converted to numpy array
            label0.append(val0.cpu().numpy())
        for val2 in label0:
            label1.append(val2.tolist())#converting from array to list of list of labels
        for val3 in label1:#converting in list of labels
            for val4 in val3:
                labels.append(val4)




        #extracting the predictions from tensors and making a list
        pred0=[]
        pred1=[]
        predictions=[] #extracted label

        for val0 in self.y_pred:
            #converted to numpy array
            pred0.append(val0.cpu().numpy())
        for val2 in pred0:
            pred1.append(val2.tolist())#converting from array to list of list of labels
        for val3 in pred1:#converting in list of labels
            for val4 in val3:
                predictions.append(val4)
        return predictions, labels

    def confusion_matrix(self):
        predictions, labels = self.tensors_to_int_lists()
        cm = confusion_matrix(labels, predictions)
        # Creating a dataframe for a array-formatted Confusion matrix,so it will be easy for plotting.
        cm_df = pd.DataFrame(cm,index = [classes], columns = [classes])

        print("confusion metrix: ".format(cm))
        #Plotting the confusion matrix
        plt.figure(figsize=(5,4))
        sns.heatmap(cm_df, annot=True)
        plt.title('Confusion Matrix')
        plt.ylabel('Actual Values')
        plt.xlabel('Predicted Values')
        plt.show()
        
    def classification_report(self):
        predictions, labels = self.tensors_to_int_lists()
        #classification report
        target_names = classes
        print(classification_report(labels, predictions, target_names=target_names))

    def accuracy_score(self):
        predictions, labels = self.tensors_to_int_lists()
        print(f"Accuracy Score: {(metrics.accuracy_score(labels, predictions)) * 100} %" )

