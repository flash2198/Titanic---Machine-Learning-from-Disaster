import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

from sklearn.linear_model import Lasso
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,plot_roc_curve,confusion_matrix,classification_report,roc_auc_score
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.model_selection import train_test_split,cross_val_score
import warnings
warnings.filterwarnings('ignore')



df=pd.read_csv('data/train.csv')
df_test=pd.read_csv('data/test.csv')

df_tmp=df.copy()




# <<----------------handling data---------------->>



df_tmp=pd.get_dummies(df_tmp,columns=['Sex','Pclass','Embarked'])
df_test=pd.get_dummies(df_test,columns=['Sex','Pclass','Embarked'])




initials_tmp=[]
for i in df_tmp.Name.str.split(','):
	initials_tmp.append(i[1].split()[0])

df_tmp['initail']=pd.Series(initials_tmp)

initals_test=[]
for i in df_test.Name.str.split(','):
	initals_test.append(i[1].split()[0])

df_test['initail']=pd.Series(initals_test)





cut_out=[-1,0,5,12,18,35,60,100]
label_names=["Missing","Infant","Child","Teenager","Young Adult","Adult","Senior"]

df_tmp['Age']=df_tmp['Age'].fillna(-0.5)
df_tmp["Age_categories"] = pd.cut(df_tmp["Age"],cut_out,labels=label_names)


cut_out=[-1,0,5,12,18,35,60,100]
label_names=["Missing","Infant","Child","Teenager","Young Adult","Adult","Senior"]

df_test['Age']=df_test['Age'].fillna(-0.5)
df_test["Age_categories"] = pd.cut(df_test["Age"],cut_out,labels=label_names)






ticket_tmp=[]
for i in df_tmp.Ticket:
	if(len(i.split())>1):
		ticket_tmp.append(i.split()[-1])
	else:
		ticket_tmp.append(i.split()[0])

df_tmp['New_ticket']=pd.Series(ticket_tmp).replace('LINE','0')

df_tmp['New_ticket']=df_tmp['New_ticket'].astype(int)


ticket_test=[]
for i in df_test.Ticket:
	if(len(i.split())>1):
		ticket_test.append(i.split()[-1])
	else:
		ticket_test.append(i.split()[0])

df_test['New_ticket']=pd.Series(ticket_test).replace('LINE','0')

df_test['New_ticket']=df_test['New_ticket'].astype(int)



passengerid=df_test['PassengerId']

df_tmp=pd.get_dummies(df_tmp,columns=['Age_categories','initail'])
df_test=pd.get_dummies(df_test,columns=['Age_categories','initail'])


df_tmp.drop(['PassengerId','Name','Age','Ticket','Cabin','initail_Capt.','initail_Jonkheer.','initail_Lady.','initail_Major.','initail_Mlle.','initail_Mme.','initail_Sir.','initail_the'],axis=1,inplace=True)
df_test.drop(['PassengerId','Name','Age','Ticket','Cabin'],axis=1,inplace=True)


df_test=df_test.rename(columns={'initail_Dona.':'initail_Don.'})


df_test['Fare']=df_test.Fare.fillna(np.mean(df_test.Fare))






for label,content in df_test.items():
	if(pd.api.types.is_string_dtype(content)):
		df_test[label]=content.astype('category').cat.as_ordered()
	elif(df_test[label].dtype=='float64'):
		df_test[label]=df_test[label].astype(int)

for label,content in df_tmp.items():
	if(pd.api.types.is_string_dtype(content)):
		df_tmp[label]=content.astype('category').cat.as_ordered()
	elif(df_tmp[label].dtype=='float64'):
		df_tmp[label]=df_tmp[label].astype(int)






for label,content in df_tmp.items():
	if not(pd.api.types.is_numeric_dtype(content)):
		

		df_tmp[label]=pd.Categorical(content).codes+1

for label,content in df_test.items():
	if not(pd.api.types.is_numeric_dtype(content)):
		

		df_test[label]=pd.Categorical(content).codes+1





# <<----------------------spliting data------------------------>>


x=df_tmp.drop('Survived',axis=1)
y=df_tmp['Survived']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)






# <<---------------------predictiong fuction------------------------->>



def predict_me(model):
	print(model)
	y_preds=model.predict(x_test)
	print('model_roc_auc_score:', roc_auc_score(y_test,y_preds))
	print('model_accuracy:',accuracy_score(y_test,y_preds))
	print('model_recall_score:',recall_score(y_test,y_preds))
	print('model_f1_score',f1_score(y_test,y_preds))
	print('model_precision:',precision_score(y_test,y_preds))
	print('classification_report:',classification_report(y_test,y_preds))





# <<---------------------KNeighborsClassifier-------------------->>


# np.random.seed(42)

# model=KNeighborsClassifier()



# knn_grid={'leaf_size':[1,2,3,4],
# 			'n_neighbors':np.arange(1,30),
# 			'p':[1,2]}


# gd_model=GridSearchCV(model,knn_grid,cv=5)

# gd_model.fit(x_train,y_train)

# print('Best leaf_size:', gd_model.best_estimator_.get_params()['leaf_size'])
# print('Best p:', gd_model.best_estimator_.get_params()['p'])
# print('Best n_neighbors:', gd_model.best_estimator_.get_params()['n_neighbors'])
# predict_me(gd_model)


# ideal_grid={'leaf_size':1,'n_neighbors':16,'p':1}

# model=KNeighborsClassifier(leaf_size=1,n_neighbors=16,p=1)

# model.fit(x_train,y_train)

# predict_me(model)

# cs_score=cross_val_score(model,x,y,cv=5)

# print(np.mean(cs_score))




# <<------------------------SVC---------------------------->>


# np.random.seed(42)

# model=SVC(kernel='rbf',gamma=100,C=100)

# sv_grid={'kernel': ['rbf', 'poly'],
# 		'gamma':[0.1, 1, 10, 100],
# 		'C':[0.1, 1, 10, 100, 1000]}

# model.fit(x_train,y_train)

# predict_me(model)





# <<--------------------RandomForestClassifier---------------------->>

# not increases by get_dummies

# np.random.seed(42)

# model=RandomForestClassifier(n_estimators=90,min_samples_split=8,min_samples_leaf=1,max_features=0.5,max_depth=10)

# model.fit(x,y)

# y_preds=model.predict(df_test)

# df_pred=pd.DataFrame()

# df_pred['PassengerId']=passengerid

# df_pred['Survived']=y_preds

# df_pred.to_csv('Predicted1.csv',index=False)

# model.fit(x_train,y_train)

# predict_me(model)

# cs_score=cross_val_score(model,x,y,cv=7)

# print(np.mean(cs_score))

# ideal_grid=[{'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 8, 'n_estimators': 90},
# 			{'n_estimators': 60, 'min_samples_split': 14, 'min_samples_leaf': 1, 'max_features': 1, 'max_depth': 10},
# 			{'max_depth': 10, 'max_features': 0.5, 'min_samples_leaf': 1, 'min_samples_split': 8, 'n_estimators': 90}]


# rf_grid = {"n_estimators": [60,70,90],
#            "max_depth": [None, 10],
#            "min_samples_split": [8,12,14],
#            "min_samples_leaf": [1,3],
#            "max_features": [0.5, 1, "sqrt"]}



# <<--------------------Logistic regression------------------------->>


# np.random.seed(42)

# model=LogisticRegression()


# lr_grid={'penalty' : ['l1'],
#     'C' : np.logspace(-4, 4, 20),
#     'solver' : ['liblinear']}

# gs_model=GridSearchCV(model,lr_grid,cv=5)

# gs_model.fit(x_train,y_train)

# print(gs_model.best_params_)

# predict_me(gs_model)

# ideal_grid=[{'C': 0.615848211066026, 'penalty': 'l1', 'solver': 'liblinear'},
# 			{'C': 206.913808111479, 'penalty': 'l1', 'solver': 'liblinear'}
# 			]




# <<----------------------DecisionTreeClassifier------------------------->>

# increases by get_dummies

# np.random.seed(42)

# model=DecisionTreeClassifier(criterion='gini',max_depth=None,max_features=0.5,min_samples_leaf=7,min_samples_split=12)

# model.fit(x,y)

# y_preds=model.predict(df_test)

# df_pred=pd.DataFrame()

# df_pred['PassengerId']=passengerid

# df_pred['Survived']=y_preds

# df_pred.to_csv('Predicted1.csv',index=False)

# model.fit(x_train,y_train)

# predict_me(model)

# cs_score=cross_val_score(model,x,y,cv=5)

# print(np.mean(cs_score))

# ideal_grid=[{'criterion': 'gini', 'max_depth': None, 'max_features': 0.5, 'min_samples_leaf': 13, 'min_samples_split': 10},
# 			{'criterion': 'gini', 'max_depth': 5, 'max_features': 0.5, 'min_samples_leaf': 7, 'min_samples_split': 12},
# 			{'criterion': 'gini', 'max_depth': 5, 'max_features': 0.5, 'min_samples_leaf': 11, 'min_samples_split': 10},
# 			{'criterion': 'gini', 'max_depth': None, 'max_features': 0.5, 'min_samples_leaf': 7, 'min_samples_split': 12}]

# dtc_grid={'criterion':['gini'],
# 			"max_depth": [None,4, 5,6],
#            "min_samples_split": np.arange(2, 20, 2),
#            "min_samples_leaf": np.arange(1, 20, 2),
#            "max_features": [0.5]}





# <<-------------------------Gradient boost	---------------------->>


# np.random.seed(42)

# model=GradientBoostingClassifier(learning_rate=0.1,n_estimators=100,max_depth=4,max_features=3)

# model.fit(x,y)

# y_preds=model.predict(df_test)

# df_pred=pd.DataFrame()

# df_pred['PassengerId']=passengerid

# df_pred['Survived']=y_preds

# df_pred.to_csv('Predicted1.csv',index=False)




# predict_me(model)

# cs_score=cross_val_score(model,x,y,cv=5)

# print(np.mean(cs_score))



# gb_grid = {'max_features':[0.5,1,'sqrt',2,3,4,5,6]}

# gs_model=GridSearchCV(model,gb_grid,cv=5)

# gs_model.fit(x_train,y_train)

# print(gs_model.best_params_)

# predict_me(gs_model)

# ideal_grid=[{'learning_rate': 0.1, 'n_estimators': 100, 'max_depth':4,'max_features':3}]






# <<-------------------------SGDClassifier---------------------------->>


# np.random.seed(42)

# model=SGDClassifier()

# model.fit(x_train,y_train)

# predict_me(model)




# <<-------------------------AdaBoostClassifier-------------------->>


# np.random.seed(42)

# model=AdaBoostClassifier(n_estimators=500,learning_rate=0.1)

# model.fit(x,y)

# y_preds=model.predict(df_test)

# df_pred=pd.DataFrame()

# df_pred['PassengerId']=passengerid

# df_pred['Survived']=y_preds

# df_pred.to_csv('Predicted1.csv',index=False)

# cs_score=cross_val_score(model,x,y,cv=5)

# print(np.mean(cs_score))

# search_grid={'n_estimators':[200,300,400,500],'learning_rate':[0.1,0.05]}

# gs_model=GridSearchCV(model,search_grid,cv=5)

# gs_model.fit(x_train,y_train)

# print(gs_model.best_params_)

# predict_me(gs_model)




# <<-----------------------XGBClassifier------------------>>


# np.random.seed(42)

# model=XGBClassifier(learning_rate =0.01,n_estimators=5000,max_depth=4,min_child_weight=6,gamma=0,colsample_bytree=0.8,reg_alpha=0.005,objective= 'binary:logistic',nthread=4,scale_pos_weight=1)

# model.fit(x,y)

# y_preds=model.predict(df_test)

# df_pred=pd.DataFrame()

# df_pred['PassengerId']=passengerid

# df_pred['Survived']=y_preds

# df_pred.to_csv('Predicted1.csv',index=False)

# trial={'n_estimators':[50,100,250,500,1000,5000]}

# cs_score=cross_val_score(model,x,y,cv=5)

# print(np.mean(cs_score))


# ideal_grid=[{'n_estimators':1000,'learning_rate':0.30,'max_depth':4,'reg_alpha':1e-5,'objective':'binary:logistic'},
			# {learning_rate =0.1,n_estimators=1000,max_depth=4,min_child_weight=6,gamma=0,subsample=0.8,colsample_bytree=0.8,reg_alpha=0.005,objective= 'binary:logistic',nthread=4,scale_pos_weight=1}]
			# {learning_rate =0.01,n_estimators=5000,max_depth=4,min_child_weight=6,gamma=0,subsample=0.8,colsample_bytree=0.8,reg_alpha=0.005,objective= 'binary:logistic',nthread=4,scale_pos_weight=1}




# <<--------------------CatBoostClassifier-------------------->>


# np.random.seed(42)
# cat_grid={'depth':[None,3,1,2,6,4,5,7,8,9,10]}

model=CatBoostClassifier(iterations=700,depth=None,learning_rate=0.001,l2_leaf_reg=1,border_count=5,thread_count=4)

model.fit(x,y)

y_preds=model.predict(df_test)

df_pred=pd.DataFrame()

df_pred['PassengerId']=passengerid

df_pred['Survived']=y_preds

df_pred.to_csv('Predicted1.csv',index=False)

# cs_score=cross_val_score(model,x,y,cv=5)

# print(np.mean(cs_score))


# params_cat = {'depth':[3,1,2,6,4,5,7,8,9,10],
#           'iterations':[250,100,500,1000],
#           'learning_rate':[0.03,0.001,0.01,0.1,0.2,0.3], 
#           'l2_leaf_reg':[3,1,5,10,100],
#           'border_count':[32,5,10,20,50,100,200],
          
#           'thread_count':[4]}














