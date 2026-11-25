You read job descriptions for a team of docked mecanum-wheel robot modules and work out the motions each job needs.
Do not run any commands and do not read files. Answer only with the JSON object the schema asks for.

The robots. Each module is a 0.2 m x 0.2 m square mecanum base (12.9 kg with its deck plate) that can drive forward/
back (surge), sideways (sway), diagonally, and turn in place (yaw). Modules dock rigidly into 1-8-module bodies. A
healthy body reaches about 0.27 m/s on straight and diagonal legs and about 50 deg/s turning in place. Assume about
5 kg payload per module carried comfortably.

Task. For each of the {{N}} jobs below, read only the title, description, environment and payload, and derive:
- `team_modules`: the body size the job needs (1-8); `payload_kg`: the payload mass (0 if none);
- `motion`: the peak speed the job needs on each axis, 0 if not needed: `surge_mps`, `sway_mps`, `diag_mps` (m/s),
  `yaw_dps` (deg/s), and the directions used: `surge_dirs` (surge+ forward, surge- reverse), `sway_dirs` (sway+ left,
  sway- right), `diag_dirs` (diag++ front-left, diag+- front-right, diag-+ rear-left, diag-- rear-right), `yaw_dirs`
  (yaw+ counter-clockwise, yaw- clockwise). Empty list when the motion is not needed. A motion is needed only if the
  job cannot be done without it. Derive speeds from the stated distances, times and precision.

Jobs (ids {{IDS}}):
{{CASES}}
