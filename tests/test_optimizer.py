import optuna
from algae.optimize.bo import sim_objective

def test_optimizer_finds_the_simulator_optimum():
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=0, n_startup_trials=8))
    study.optimize(sim_objective, n_trials=40)
    p = study.best_params
    assert 26 <= p['temp_set'] <= 33
    assert 6.8 <= p['ph_set'] <= 8.0
