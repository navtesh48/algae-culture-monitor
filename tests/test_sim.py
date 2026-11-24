from algae.sim.simulator import simulate

def test_deterministic():
    a = simulate(seed=1); b = simulate(seed=1)
    assert a.od750.equals(b.od750)

def test_no_nans_and_grows_without_events():
    df = simulate(seed=2, events=False)
    assert not df.isna().any().any()
    assert df.biomass_gl.iloc[-1] > 3 * df.biomass_gl.iloc[0]

def test_contamination_lowers_biomass():
    for s in range(50):
        df = simulate(seed=s)
        if (df.event == 'contamination').any():
            idx = df.index[df.event == 'contamination']
            assert df.biomass_gl[idx[-1]] < df.biomass_gl[idx[0]]
            return
    raise AssertionError('no contamination event in 50 seeds')
