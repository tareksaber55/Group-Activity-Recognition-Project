import torch
from sklearn.metrics import accuracy_score, confusion_matrix , classification_report , f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import os
class report:
    def __init__(self,all_labels,all_preds,team_level = True):
        self.all_labels = all_labels
        self.all_preds = all_preds
        self.team_level = team_level
        self.team_labels = {
            0:'l_pass',     1:'r_pass',
            2:'l_spike',    3:'r_spike',
            4:'l_set',      5:'r_set',
            6:'l_winpoint', 7:'r_winpoint'
        }
        self.player_labels = {
            0:'waiting', 1:'setting', 2:'digging', 
            3:'falling', 4:'spiking', 5:'blocking',
            6:'jumping', 7:'moving', 8:'standing'
        }
    def make_report(self,output_path):
        labels = []
        if self.team_level:
            labels = [self.team_labels[k] for k in sorted(self.team_labels.keys())]
        else:
            labels = [self.player_labels[k] for k in sorted(self.player_labels.keys())]
        # confusion matrix
        cm = confusion_matrix(self.all_labels,self.all_preds)
        cm_precent = cm.astype('float') / cm.sum(axis=1,keepdims=True)
        sns.heatmap(cm_precent,annot=True,fmt='.2f',xticklabels=labels,yticklabels=labels)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")

        plt.savefig(os.path.join(output_path,'confusion_matrix.png'),dpi=300)

        # classification report + f1_score
        cr = classification_report(
            self.all_labels,
            self.all_preds,
            target_names=labels,
            zero_division=0
        )

        f1_macro = f1_score(self.all_labels, self.all_preds, average='macro')
        f1_weighted = f1_score(self.all_labels, self.all_preds, average='weighted')
        acc = accuracy_score(self.all_labels, self.all_preds)

        final_report = f"""
        ================ Classification Report ================

        {cr}

        ================ Global Metrics =======================

        Accuracy     : {acc:.4f}
        F1 Macro     : {f1_macro:.4f}
        F1 Weighted  : {f1_weighted:.4f}

        =======================================================
        """

        with open(os.path.join(output_path, 'classification_report.txt'), 'w') as f:
            f.write(final_report)
                


