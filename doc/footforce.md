That indicates the Go2 foot_force values are probably raw sensor units, not Newtons.

  Your latest stationary simulation recording averages:

  - Feet: 48.4, 42.5, 28.0, 34.0 N
  - Total: 152.8 N

  The simulated robot mass is approximately 16 kg, so expected weight is about 157 N. Therefore, the
  simulation total is physically correct. Different feet carry different loads because the center of
  mass is not centered.

  Real readings of 22–26 total only around 96. If interpreted as Newtons, that implies a robot mass
  near 9.8 kg, which is unrealistic.

  For sim-to-real, calibrate hardware readings rather than forcing every simulated foot to 22–26:

  [
  F_{\text{sim-equivalent},i}

  (F_{\text{hardware},i}-offset_i)\times gain_i
  ]

  A preliminary global gain is:

  [
  gain \approx 153/96 \approx 1.6
  ]

  So hardware 24 would become approximately 38 N. Ideally, determine separate offset and gain for
  each foot using unloaded and standing measurements.

  Also, unequal simulated values such as 22–55 N are normal if their sum stays close to robot
  weight. No files were changed.