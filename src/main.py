from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parents[1]
for folder in ('data', 'screenshots', 'results'):
    (ROOT / folder).mkdir(exist_ok=True)
rng = np.random.default_rng(42)
n = 1200
monthly = rng.uniform(20, 120, n)
tenure = rng.integers(1, 73, n)
contract = rng.choice(['Month-to-month', 'One year', 'Two year'], n, p=[0.55, 0.28, 0.17])
support = rng.poisson(1.5, n)
logit = -0.9 + 0.022*(monthly-70) - 0.035*(tenure-24) + 0.48*support + 0.85*(contract=='Month-to-month') - 0.75*(contract=='Two year')
prob = 1/(1+np.exp(-logit))
churn = rng.binomial(1, prob)
df = pd.DataFrame({'monthly_charges': monthly.round(2), 'tenure_months': tenure, 'support_tickets': support, 'contract_type': contract, 'churn': churn})
df.to_csv(ROOT/'data/customers_demo.csv', index=False)
X, y = df.drop(columns='churn'), df['churn']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)
pre = ColumnTransformer([('numeric',StandardScaler(),['monthly_charges','tenure_months','support_tickets']),('categorical',OneHotEncoder(handle_unknown='ignore'),['contract_type'])])
model = Pipeline([('preprocess',pre),('classifier',LogisticRegression(max_iter=1000,random_state=42))])
model.fit(X_train,y_train)
pred = model.predict(X_test)
score = model.predict_proba(X_test)[:,1]
metrics = {'accuracy':float(accuracy_score(y_test,pred)),'precision':float(precision_score(y_test,pred,zero_division=0)),'recall':float(recall_score(y_test,pred,zero_division=0)),'f1':float(f1_score(y_test,pred,zero_division=0)),'roc_auc':float(roc_auc_score(y_test,score)),'note':'Synthetic demonstration data only'}
(ROOT/'results/metrics.json').write_text(json.dumps(metrics,indent=2))
fig,axes=plt.subplots(1,2,figsize=(11,4.5))
ConfusionMatrixDisplay.from_predictions(y_test,pred,ax=axes[0],colorbar=False,cmap='Blues')
axes[0].set_title('Churn classification • synthetic test set')
fpr,tpr,_=roc_curve(y_test,score)
axes[1].plot(fpr,tpr,label=f'AUC = {metrics["roc_auc"]:.2f}',linewidth=2)
axes[1].plot([0,1],[0,1],'--',color='gray')
axes[1].set(xlabel='False positive rate',ylabel='True positive rate',title='ROC curve • synthetic test set')
axes[1].legend();fig.tight_layout();fig.savefig(ROOT/'screenshots/churn_evaluation.png',dpi=170);plt.close(fig)
print('Synthetic demonstration metrics:',json.dumps(metrics,indent=2))
print('Sample churn probability:',round(float(model.predict_proba(X_test.iloc[[0]])[0,1]),3))
