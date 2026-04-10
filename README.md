# Expfit: fit noisy time series with 1 to 4 exponentials

- With guessing of initial parameters
- With tests
- A place to try out a few methods
  - Cobble something together that fits 1, then 2, then 3 etc.
  - Or just chuck it in a global opt, but with sensible constraints e.g. tau1 > tau2 > tau3 etc (unless that's not sensible, cause if it finds tau1=tau2 it'd be stuck until they are both near equal and the other can move again? might be better to order after finding?)
- Ideally with minimal dependencies (because for low number of exponentials (1--4) it shouldn't be super hard.

## State 2026-04-10

Does single exponentials well

Building on initial guesses for single, also started adding double which it does OK

Not much testing for doubles yet.



## Future?

Maybe stand-alone. Maybe in pcpostprocess or datkit

For now, need somewhere to play and document.
