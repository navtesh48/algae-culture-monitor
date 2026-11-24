"""Central place for tunables shared across modules."""
HORIZON_HOURS = 24
ANOMALY_PERCENTILE = 99.5   # threshold percentile on fault-free training scores
ANOMALY_CONSECUTIVE = 3     # consecutive hours above threshold before alerting
