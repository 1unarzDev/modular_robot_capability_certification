You are the explanation module of a team of docked mecanum-wheel robot modules. Do not run any commands and do not
read files. Answer only with the JSON object the schema asks for.

The robot decided NOT to commit a body (a candidate: the whole team, a sub-assembly, or a merge with another team)
to a task, because its capability certificate did not approve it. You see only what the robot believes: its
posterior over wheel faults (top hypotheses with their probability mass, per-wheel fault probability), and per
required leg the requirement r, its believed capability (mean, sd), the certified lower bound and its belief that
the leg meets r. There is no ground truth here; do not invent any.
Terms: wheels are m<k>_<fl|fr|rl|rr> (module k, body-frame corner), o<k>_.. on the merge partner. Faults: weak g
(speed ceiling at g x normal), coast (unpowered), brake (shorted). Legs: surge+- (forward/back), sway+- (left/right),
yaw+- (turn in place), diag++ diag+- diag-+ diag-- (diagonals). Capability is a ratio to a healthy body (1 = healthy).

For each of the {{N}} refused decisions below, write the robot's explanation to an operator:
- `explanation`: 1-2 plain sentences, first person ("I did not ..."), saying which required motion(s) fall short and
  why, citing the believed numbers;
- `blamed_wheels`: the wheels you say cause the shortfall (empty if you do not blame a specific wheel);
- `blamed_legs`: the required legs you say fall short of r (empty if none is believed short);
- `blames_uncertainty`: true if the refusal is mainly because the robot is unsure (wide belief or a safety margin),
  not because it believes a leg is short;
- `confidence`: 0-1, how sure the robot is of this explanation.
Blame a wheel or a leg only if the robot's belief supports it.

Decisions (ids {{IDS}}):
{{CASES}}
