"""LightGBM quantile forecaster for OD750 24 h ahead.
Split BY CULTURE (never by random rows) so the future cannot leak into training."""
import joblib, pandas as pd, lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from algae.sim.simulator import simulate
from algae.forecast.features import make_features, FEATURES

def _build(ids):
    return pd.concat([make_features(simulate(30, f'pbr{k}', seed=k)) for k in ids]).dropna()

def run(save=True) -> dict:
    train, val, test = _build(range(0, 14)), _build(range(14, 17)), _build(range(17, 20))
    specs = {'p50': dict(objective='regression_l1'),
             'p10': dict(objective='quantile', alpha=0.1),
             'p90': dict(objective='quantile', alpha=0.9)}
    models = {}
    for name, kw in specs.items():
        m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=15,
                              min_child_samples=30, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, verbose=-1, **kw)
        m.fit(train[FEATURES], train['y'], eval_set=[(val[FEATURES], val['y'])],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        models[name] = m
    p50 = models['p50'].predict(test[FEATURES])
    lo, hi = models['p10'].predict(test[FEATURES]), models['p90'].predict(test[FEATURES])
    res = {'mae_persistence': mean_absolute_error(test['y'], test['od750']),
           'mae_lgbm': mean_absolute_error(test['y'], p50),
           'coverage_80': float(((test['y'] >= lo) & (test['y'] <= hi)).mean())}
    if save:
        joblib.dump(models, 'data/models/forecast.joblib')
    print(res)
    return res

if __name__ == '__main__':
    run()
