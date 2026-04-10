# Expfit: Fit exponentials to noisy time series

Expfit is a lightweight Python package to fit exponentials to noisy time 
series, typically with the aim of extracting time constants.

It aims to meet the following goals:

- Fit scaled and vertically transposed exponentials ``y = a + b * exp(c * x)``
  without requiring initial parameter estimates.
- Fit double, triple, and quadruple exponentials ``y = a + b_i * exp(c_i * x)``
  where each exponential term is decaying (``c_i < 0``) and all ``b_i`` have
  the same sign.
- Fit multiple decaying exponentials in data with multiple ``b_i`` signs.
- Be lightweight: use good initial strategies and properties of exponentials to
  simplify the optimisation problem.

Although a relatively simple task, ``expfit`` has unit tests, and reported
failures will be added to its test suite to create a reliable tool for this
sometimes fiddly operation.


## State 2026-04-10

Does single exponentials well

Building on initial guesses for single, also started adding double which it does OK



