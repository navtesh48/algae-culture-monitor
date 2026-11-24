"""The model only restates what is in the numbers and offers hypotheses. It never gets to invent measurements."""
import pandas as pd
from pydantic import BaseModel
from algae.llm import structured

class Explanation(BaseModel):
    summary: str
    likely_causes: list[str]     # hypotheses, max 3
    checks: list[str]            # concrete things to check or do, max 3
    confidence: str              # low | medium | high

SYSTEM = ('You are a cautious algae cultivation process assistant. Use ONLY the numbers given. '
          'Every cause is a hypothesis, never a fact. Max 3 causes and 3 checks. Plain English, no jargon '
          'without a short definition. If the data cannot distinguish causes, say so and set confidence low.')

def explain(alert: dict, recent: pd.DataFrame) -> Explanation:
    r = recent.tail(48)
    stats = r[['temp_c', 'ph', 'do_mgl', 'par', 'od750']].describe().loc[['min', 'mean', 'max']].round(2)
    change = (r.iloc[-1][['temp_c', 'ph', 'do_mgl', 'od750']] - r.iloc[0][['temp_c', 'ph', 'do_mgl', 'od750']]).round(2)
    user = (f'Alert: {alert}\n\nLast 48h stats:\n{stats.to_string()}\n\n'
            f'Change first-to-last hour:\n{change.to_string()}\n\nExplain this alert.')
    return structured(Explanation, SYSTEM, user)
