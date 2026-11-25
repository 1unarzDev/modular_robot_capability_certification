Checkable claims. Every claim is verified by code against simulator ground truth, including held-out quantities
you do not see. Make 2-5 claims per case, each specific and falsifiable; prefer claims that carry the mechanism.
Use exactly these kinds (target = wheel name, leg name, probe name, module name or latent name; op in lt, gt, eq;
value a number; mode only for wheel_mode, else ""):
- wheel_mode(target wheel, mode)                  the true mode of that wheel
- wheel_gain(target wheel, op, value)             true g of the wheel (1.0 if not weak)
- wheel_stall / wheel_noload(target wheel, op, value)   true per-wheel motor multipliers
- spread_masked(target wheel, value 1|0)          the wheel is weak with g inside the motor-spread band (>= ~0.85)
- wheel_in_candidate(target wheel, value 1|0)
- candidate_has_fault(value 1|0)                  the candidate contains a truly faulted wheel
- module_mu(target module, op, value)             per-module friction factor
- latent(target global latent name, op, value)    A5 global latents (mu, s0, c_roller, ...)
- leg_truth(target leg, op, value)                the true label on that leg (A5: the T2 value)
- leg_t1(target leg, op, value)                   A5 only: the T1 value
- leg_margin(target leg, op, value)               |true label - threshold| (A5: |T1 - tau|)
- leg_shift(target leg, op, value)                A5 only: T2 - T1 on that leg
- post_mean / post_sd / p_leg(target leg, op, value)    the v2 posterior's mean, sd, P(rho >= r) on that leg
- set_min / set_max(target leg, op, value)        A2 only: min / max of rho over the consistent set
- truth_mass(op, value)                           posterior mass on the true fault hypothesis (held out)
- truth_in_screen(value 1|0)                      the true hypothesis survived the hypothesis screen (held out)
- probe_resolves(target clamp|moddiff|tug|split, value 1|0)   held out: after that single extra probe the decision
                                                  is unambiguous and on the correct side (ambiguous cases only)
- condition_flip(target base|compliance|load_transfer|aniso|aniso_mu|all-aniso_mu, value 1|0)   held out, A5 only:
                                                  the flip also occurs in T2 with only that effect set (base = the
                                                  independent kernel alone; all-aniso_mu = every effect except the
                                                  per-wheel roller-friction scatter)
Give each claim a short `text` in words.
