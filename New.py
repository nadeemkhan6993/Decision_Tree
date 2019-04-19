import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
import time
import graphviz
import pydotplus
import io
from scipy import misc

data = pd.read_csv('indian_liver_patient.csv')
start = time.perf_counter()
train, test = train_test_split(data, test_size= 0.15)
print("Training size: {}; Test size: {}".format(len(train), len(test)))

c = DecisionTreeClassifier(min_samples_split=50)
features = ["Age","Total_Bilirubin","Direct_Bilirubin","Alkaline_Phosphotase","Alamine_Aminotransferase","Aspartate_Aminotransferase","Total_Protiens","Albumin"]
X_train = train[features]
y_train = train["Dataset"]

X_test = test[features]
y_test = test["Dataset"]
dt = c.fit(X_train, y_train)


#Uncomment the portin below to see the tree
'''
def show_tree(tree, features, path):
    f = io.StringIO()
    export_graphviz(tree, out_file= f, feature_names= features)
    pydotplus.graph_from_dot_data(f.getvalue()).write_png(path)
    
    img = misc.imread(path)
    
    plt.rcParams["figure.figsize"] = (20,20)
    plt.imshow(img)

show_tree(dt, features,'dec_tree_01.png')
'''
y_pred = c.predict(X_test)

score= accuracy_score(y_test, y_pred)*100

print("Accuracy : ", round(score,1), "%")
print("Took %f secs" % (time.perf_counter() - start))
