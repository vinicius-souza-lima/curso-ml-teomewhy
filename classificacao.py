# %%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model,tree,naive_bayes
# %%

df = pd.read_excel("dados/dados_cerveja_nota.xlsx")
# %%
df.head()
# %%
df["aprovado"] = (df['nota'] > 5).astype(int)
# %%
df
# %%
y = df['aprovado']
X = df[['cerveja']]
plt.plot(df['cerveja'], y, 'o', color ='royalblue')
plt.grid(True)
plt.title('Cerveja x Aprovação')
plt.xlabel("Cerveja")
plt.ylabel('Aprovado')
# %%
reg = linear_model.LogisticRegression(fit_intercept=True)
reg.fit(X, y)
reg_predict = reg.predict(X.drop_duplicates())
reg_proba = reg.predict_proba(X.drop_duplicates())[:,1]
# %%
arvore_full = tree.DecisionTreeClassifier(random_state=42)
arvore_full.fit(X, y)
arvore_full_predict = arvore_full.predict(X.drop_duplicates())
arvore_full_proba = arvore_full.predict_proba(X.drop_duplicates())[:,1]
# %%
arvore_d2 = tree.DecisionTreeClassifier(random_state=42,max_depth=2)
arvore_d2.fit(X, y)
arvore_d2_predict = arvore_d2.predict(X.drop_duplicates())
arvore_d2_proba = arvore_d2.predict_proba(X.drop_duplicates())[:,1]
# %%
nb = naive_bayes.GaussianNB()
nb.fit(X, y)
nb_predict = nb.predict(X.drop_duplicates())
nb_proba = nb.predict_proba(X.drop_duplicates())[:,1]
# %%
plt.plot(X, y, 'o', color ='royalblue')
plt.grid(True)
plt.title('Cerveja x Aprovação')
plt.xlabel("Cerveja")
plt.ylabel('Aprovado')

plt.plot(X.drop_duplicates(), reg_predict, color='tomato')
plt.plot(X.drop_duplicates(), reg_proba, color='red')

plt.plot(X.drop_duplicates(), arvore_full_predict, color='tomato')
plt.plot(X.drop_duplicates(), arvore_full_proba, color='red')

plt.plot(X.drop_duplicates(), arvore_d2_predict, color='tomato')
plt.plot(X.drop_duplicates(), arvore_d2_proba, color='red')

plt.plot(X.drop_duplicates(), nb_predict, color='blue')
plt.plot(X.drop_duplicates(), nb_proba, color='green')

plt.hlines(0.5,linestyles='--',xmin=1,xmax=9,colors='black')
plt.legend(['Observação','Reg Predict','Reg Proba','Naive predict', 'Naive Proba'])