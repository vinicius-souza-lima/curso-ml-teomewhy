# %%
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
# %%
df = pd.read_parquet("dados/dados_clones.parquet")
df.head()
# %%
target = df.columns[-1]
features = [
    "Massa(em kilos)","Estatura(cm)",
    "Distância Ombro a ombro",
    "Tamanho do crânio","Tamanho dos pés",
    "Tempo de existência(em meses)"
]

X = df[features]
y = df[target]

X = X.replace({
    "Yoda":0,"Shaak Ti":1,"Obi-Wan Kenobi":2,"Mace Windu":3,"Aayla Secura":4,
    "Tipo 1":1,"Tipo 2":2,"Tipo 3":3,"Tipo 4":4,"Tipo 5":5
})
# %%
y
# %%
model = tree.DecisionTreeClassifier(random_state=42)
model.fit(X,y)
# %%
plt.figure(dpi=400)

tree.plot_tree(model,
               feature_names=features,
               class_names=model.classes_,
               filled=True,
               max_depth=3
            )