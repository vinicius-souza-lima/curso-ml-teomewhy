# %%
import pandas as pd
from sklearn import linear_model
from sklearn import tree
import matplotlib.pyplot as plt

from frutas import arvore
# %%
df = pd.read_excel("dados/dados_cerveja_nota.xlsx")
df.head()
# %%
X = df[["cerveja"]]
y = df[["nota"]]
reg = linear_model.LinearRegression()
reg.fit(X, y)
a,b = reg.intercept_,reg.coef_[0]
print(a,b)
predict_reg = reg.predict(X.drop_duplicates())

arvore_full = tree.DecisionTreeRegressor(random_state=42)
arvore_full.fit(X,y)

arvore_d2 = tree.DecisionTreeRegressor(random_state=42,max_depth=2)
arvore_d2.fit(X,y)

predict_arvore_full = arvore_full.predict(X.drop_duplicates())
predict_arvore_d2 = arvore_d2.predict(X.drop_duplicates())
# %%
fig,ax = plt.subplots(1,1)
ax.plot(X.drop_duplicates(), predict_reg, color ='red')
ax.scatter(X,y)
ax.grid(True)
plt.title("Relação Cerveja x Nota")
plt.xlabel("Cerveja")
plt.ylabel("Nota")

plt.plot(X.drop_duplicates(),predict_arvore_full,color = 'green')
plt.plot(X.drop_duplicates(),predict_arvore_d2, color = 'magenta')

plt.legend(["Observado", f"y = {a[0]:.3f} + {b[0]:.3f}x","Árvore Full","Árvore D2"])
# %%
df.mean()
# %%
plt.figure(dpi=400)
tree.plot_tree(arvore_d2,feature_names=['cerveja'],class_names=['nota'],filled=True)