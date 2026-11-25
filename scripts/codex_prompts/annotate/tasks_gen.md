You write realistic job scenarios for a team of docked mecanum-wheel robot modules. Do not run any commands and do
not read files. Answer only with the JSON object the schema asks for.

The robots. Each module is a 0.2 m x 0.2 m square mecanum base (12.9 kg with its deck plate) that can drive in any
planar direction: forward/back (surge), sideways (sway), diagonally, and turn in place (yaw). Modules dock rigidly
edge to edge into 1-8-module bodies (a pair is 0.4 x 0.2 m, a 2x3 block 0.6 x 0.4 m). A healthy body reaches about
0.27 m/s on straight and diagonal legs and about 50 deg/s turning in place; bodies larger than 4 modules turn a little
slower. They work on flat indoor floors only (no stairs, no ramps steeper than a few degrees). A payload sits on the
deck; assume about 5 kg per module carried comfortably.

Task. For each of the {{N}} generation slots below (one JSON object per line: domain, setting, emphasis, team_hint),
write ONE distinct, concrete, plausible job that such a team would be sent on in that setting with that emphasis.
`team_hint` is a suggested body size; change it if the job needs another. Make the slots differ from each other: vary
the layout, the payload, the motions, and how demanding they are (many jobs need only a fraction of full speed; some
need near-full speed on one axis; some need careful slow alignment).
Fields:
- `title`: 3-8 words. `description`: 2-4 plain sentences a site operator would write: what is moved where, the layout,
  the tight spots, and how the body must move (say explicitly if it must slide sideways, turn in place, go diagonally
  or reverse).
- `environment`: `floor` (one of the enum values), `aisle_width_m` (the narrowest passage the body must use, 0 if
  open floor), `constraints` (2-4 short phrases).
- `payload_kg`, `payload_desc` ("none" if no payload), `team_modules` (1-8), `duration_s` (typical job time).
- `motion`: the peak speed the job needs on each axis, 0 if that motion is not needed: `surge_mps`, `sway_mps`,
  `diag_mps` (m/s), `yaw_dps` (deg/s), and the directions used: `surge_dirs` (surge+ forward, surge- reverse),
  `sway_dirs` (sway+ left, sway- right), `diag_dirs` (diag++ front-left, diag+- front-right, diag-+ rear-left,
  diag-- rear-right), `yaw_dirs` (yaw+ counter-clockwise, yaw- clockwise). Empty list when the motion is not needed.
  Speeds must follow from the description (aisle lengths, cycle times, alignment precision), not be round guesses.
- `motion_rationale`: one sentence linking each needed motion and its speed to the description.

Slots (ids {{IDS}}):
{{CASES}}
