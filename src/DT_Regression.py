import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.metrics import mean_squared_error

#manage data
all_demands = pd.read_csv('Daily_Demand_Forecasting_Orders.csv')
all_demands.head()
current_columns = all_demands.columns.tolist()
current_columns[0] = 'WeekOfMonth'
current_columns[1] = 'DayOfWeek'
current_columns[-1] = 'Total orders'
all_demands.columns = current_columns
print(all_demands.head())

X = all_demands.drop(['Total orders'],axis=1)
y = all_demands['Total orders']
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

#decision tree
#manual K_Fold
kf = KFold(n_splits=5)
results = []
scores = []
for train, test in kf.split(Xs,y):
    tree_reg = DecisionTreeRegressor(max_depth=8,random_state=0)
    tree_reg.fit(Xs[train], y[train])
    regressor_accuracy = tree_reg.score(Xs[test], y[test])
    scores.append(regressor_accuracy)
    # calculate mse
    predictions = tree_reg.predict(Xs[test])
    print(f"predictions:, {predictions}\n,true value:{y[test].values}")
    mse = mean_squared_error(y[test], predictions)
    results.append(mse)
    print('The R² score is {:03.2f}'.format(regressor_accuracy))
    print('The mse is {:03.2f}'.format(mse))
    # tree.plot_tree(tree_clf)
    # plt.show()

avg_score = np.mean(scores)*100
std_dev = np.std(scores)*100 / np.sqrt(len(scores))
mse_mean = np.mean(results)
print(f"average mse is {mse_mean:03.2f},Average R² score and standard deviation: (%.2f +- %.3f)%%" % (avg_score, std_dev))
