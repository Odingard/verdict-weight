from verdict_weight import VerdictWeight, ContextType
vw = VerdictWeight()
r = vw.score(source_reliability=0.88, n_corroborating_sources=4, age_value=30,
             correct_predictions=90, total_predictions=100,
             context=ContextType.HEALTHCARE_DIAGNOSTIC)
print(f"[DIAGNOSTIC] {r.action_tier} | CW={r.consequence_weight:.3f}")
print(f"             {r.interpretation}")
