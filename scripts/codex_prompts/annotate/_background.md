Background (CRANE study cfcap; simulation only).
- A module is a square mecanum-wheel robot (4 wheels, named m<k>_<fl|fr|rl|rr> by module k and body-frame corner;
  wheels of the merge partner are o<k>_..). Modules dock rigidly into polyomino assemblies. A candidate is a body the
  team could form next: the whole parent, a connected child (sub-assembly) or a merge with another assembly.
- Wheel modes: ok; weak(g) = the motor's drive voltage is scaled by g, i.e. a speed ceiling g*omega0 (not a torque
  deficit); coast = unpowered, free-wheeling; brake = shorted, resists motion. Per-wheel motor spread: stall and
  no-load multipliers U(0.92, 1.08) (half-width 0.08). A weak wheel with g >= (1-0.08)/(1+0.08) ~ 0.85 is the same plant
  as a healthy wheel with slower motor constants, so it cannot be identified by any probe.
- Capability label rho_dec per decision leg (surge+-, sway+-, yaw+-, diag++, diag+-, diag-+, diag--): the closed-loop
  achieved twist ratio on a ~3 s leg divided by the nominal healthy body's value; ~1 = as good as a healthy body.
  A task requires rho_dec >= r on some legs (and a minimum number of modules).
- Observation: the parent runs the v2 routine (six body-twist legs at low load: per-wheel duty and encoder speed and one
  IMU). Extra probes: clamp (ramps until wheels hit the duty clamp / friction limit), moddiff (module-differential
  twists), tug (per-wheel antagonistic tug-of-war at body rest), split (undock the child and run its own v2).
- Posterior: weighted fault hypotheses (<= 2 faults) x nuisance draws, pushed to every candidate. The certificate
  commits a (candidate, task) pair iff lower = posterior mean - q*(posterior sd + 0.02) >= r on every required leg, with
  q fitted on other episodes (sim0.1); raw uses q = 0; `all` = after v2 + clamp + moddiff + tug.
- Case kinds: unsafe_cert (certified but truly below r), ambiguous (posterior P(capable) in (0.05, 0.95), or for A2 the
  oracle consistent set straddles tau), refused_capable (not certified although truly capable), refused_incapable,
  flip_up / flip_down (A5: the truth label rho_dec >= tau under the training plant T1 differs from the independent
  stress plant T2, T2 says capable / incapable).
