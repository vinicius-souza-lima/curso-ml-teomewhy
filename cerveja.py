# %%
import pandas as pd
from sklearn import tree
# %%
df = pd.read_excel("dados/dados_cerveja.xlsx")
df.head()
#%%
target = 'classe'
features = df.columns[1:-1]

X = df[features]
y = df[target]

X = X.replace({
    "mud":1,"pint":2,
    "sim":1,"não":0,
    "clara":0,"escura":1
})

# %%
X
# %%
model = tree.DecisionTreeClassifier(random_state=42)
model.fit(X,y)
# %%
import matplotlib.pyplot as plt

plt.figure(dpi=400)

tree.plot_tree(model,
               feature_names=features,
               class_names=model.classes_,
               filled=True
               )
