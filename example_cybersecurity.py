from verdict_weight import VerdictWeight, ContextType
vw = VerdictWeight()

scenarios = [
    ("APT",     dict(source_reliability=0.95, n_corroborating_sources=3, age_value=1, correct_predictions=48, total_predictions=50, context=ContextType.CYBERSECURITY_APT)),
    ("DISINFO", dict(source_reliability=0.91, n_corroborating_sources=0, age_value=2, correct_predictions=20, total_predictions=50, context=ContextType.CYBERSECURITY_DISINFO)),
    ("ZERODAY", dict(source_reliability=0.80, n_corroborating_sources=1, age_value=0.25, correct_predictions=15, total_predictions=20, context=ContextType.CYBERSECURITY_ZERODAY)),
]
for label, kwargs in scenarios:
    r = vw.score(**kwargs)
    print(f"[{label:8s}] {r.action_tier:8s} | CW={r.consequence_weight:.3f} | {r.interpretation[:60]}")
