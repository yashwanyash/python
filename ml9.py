import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.tree import export_text
import matplotlib.pyplot as plt

x=np.array([[150,0],[170,1],[120,0],[140,1],[200,1],[130,0]])
y=np.array(['Apple','Orange','apple','Orange','Melon','Apple'])

clf=DecisionTreeClassifier(random_state=42)
clf.fit(x,y)

tree_rules=export_text(clf,feature_names=['Weight','Texture'])
print("Decision Tree Classifier Rules:/n",tree_rules)

plt.figure(figsize=(10,6))
plot_tree(clf,filled=True,feature_names=['Weight','Texture'],class_names=np.unique(y))
plt.show()