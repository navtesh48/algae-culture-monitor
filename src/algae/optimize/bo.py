"""Bayesian optimization of culture settings with Optuna (TPE).

On simulated data the objective is final biomass after a 10-day run. With a real reactor, run the
suggested setpoints yourself and report the measured yield back through suggest()/report()."""
import optuna
from algae.sim.simulator import simulate

def make_study():
    return optuna.create_study(direction='maximize', study_name='algae',
        sampler=optuna.samplers.TPESampler(seed=0, n_startup_trials=8),
        storage='sqlite:///data/optuna.db', load_if_exists=True)

def sim_objective(trial):
    p = dict(temp_set=trial.suggest_float('temp_set', 22, 36),
             ph_set=trial.suggest_float('ph_set', 6.5, 9.0),
             par_scale=trial.suggest_float('par_scale', 0.4, 1.2))
    return float(simulate(10, seed=trial.number, events=False, **p).biomass_gl.iloc[-1])

def suggest():                       # real-world loop, step 1
    study = make_study(); t = study.ask()
    return t.number, dict(temp_set=t.suggest_float('temp_set', 22, 36),
                          ph_set=t.suggest_float('ph_set', 6.5, 9.0),
                          par_scale=t.suggest_float('par_scale', 0.4, 1.2))

def report(trial_number: int, measured_yield: float):   # real-world loop, step 2
    make_study().tell(trial_number, measured_yield)

if __name__ == '__main__':
    make_study().optimize(sim_objective, n_trials=60)
    print(make_study().best_params)
