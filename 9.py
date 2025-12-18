import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")

# 1. 读数据
train_df = pd.read_excel("回归预测.xlsx", sheet_name=0, header=None)
test_df  = pd.read_excel("回归预测.xlsx", sheet_name=1, header=None)

X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values
X_test  = test_df.iloc[:, :-1].values
y_test  = test_df.iloc[:, -1].values

drop_col = 30   # 药品名在第31列
X_train = np.delete(X_train, drop_col, axis=1)
X_test  = np.delete(X_test,  drop_col, axis=1)

# 2. 基学习器与待调超参
base_learners = {
    "rf": (RandomForestRegressor(random_state=42),
           {"n_estimators":[200,400], "max_depth":[None,10,20], "min_samples_split":[2,5]}),
    "et": (ExtraTreesRegressor(random_state=42),
           {"n_estimators":[200,400], "max_depth":[None,10,20], "min_samples_split":[2,5]}),
    "gb": (GradientBoostingRegressor(random_state=42),
           {"n_estimators":[200,400], "max_depth":[3], "learning_rate":[0.05,0.1]}),
    "xgb":(XGBRegressor(random_state=42, n_jobs=-1),
           {"n_estimators":[200,400], "max_depth":[3,5], "learning_rate":[0.05,0.1]})
}

# 3. 交叉验证调参
kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_models = {}
for name, (model, param) in base_learners.items():
    gs = GridSearchCV(model, param, cv=kf, scoring="neg_mean_squared_error", n_jobs=-1)
    gs.fit(X_train, y_train)
    best_models[name] = gs.best_estimator_
    print(f"{name} best params: {gs.best_params_}")

# 4. 生成第二层特征
meta_features = np.zeros((X_train.shape[0], len(best_models)))
for i, (name, mdl) in enumerate(best_models.items()):
    # 用 out-of-fold 预测作为第二层输入
    oof_pred = np.zeros_like(y_train)
    for tr_idx, va_idx in kf.split(X_train):
        mdl.fit(X_train[tr_idx], y_train[tr_idx])
        oof_pred[va_idx] = mdl.predict(X_train[va_idx])
    meta_features[:, i] = oof_pred
    mdl.fit(X_train, y_train)

# 5. 训练第二层线性回归
stacker = LinearRegression()
stacker.fit(meta_features, y_train)

# 6. 测试集预测
test_meta = np.column_stack([m.predict(X_test) for m in best_models.values()])
y_pred = stacker.predict(test_meta)

# 7. 评估
sq_relative_errors = ((y_pred - y_test) / y_test) ** 2
mean_sre = float(np.mean(sq_relative_errors))
var_sre  = float(np.var(sq_relative_errors))

print("\n测试集结果")
print("平方相对误差均值:", mean_sre)
print("平方相对误差方差:", var_sre)