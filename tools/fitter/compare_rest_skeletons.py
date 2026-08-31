import numpy as np
# mm. MHR mean read from its rest skeleton state (cm x10); ours from chained rest translations.
rows = [
 ("shoulder width", 352, 540, 346.4, 363.1),
 ("pelvis -> neck", 518, 580, 576.6, 513.3),
 ("hip width",      164, 180, 207.1, 214.5),
 ("upper arm",      257, 260, 287.2, 277.2),
 ("forearm",        270, 240, 268.9, 258.0),
 ("thigh",          420, 430, 399.8, 402.0),
 ("shin",           421, 420, 396.5, 405.3),
]
print(f"{'segment':<16}{'MHR mean':>9}{'our rig':>9}{'perf A':>9}{'perf B':>9} | {'MHR err':>9}{'ours err':>9}")
print("-"*76)
em, eo = [], []
for name, mhr, ours, a, b in rows:
    for perf in (a, b):
        em.append(abs(mhr-perf)); eo.append(abs(ours-perf))
    print(f"{name:<16}{mhr:>9.0f}{ours:>9.0f}{a:>9.1f}{b:>9.1f} | "
          f"{(abs(mhr-a)+abs(mhr-b))/2:>8.0f} {(abs(ours-a)+abs(ours-b))/2:>8.0f}")
print("-"*76)
print(f"{'MEAN ABS ERROR':<16}{'':>27}{'':>9} | {np.mean(em):>8.0f} {np.mean(eo):>8.0f}")
print()
print(f"MHR's UNTUNED mean body is {np.mean(eo)/np.mean(em):.1f}x closer to these two")
print("performers than our canonical rig -- before fitting a single parameter.")
