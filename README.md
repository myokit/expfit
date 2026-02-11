# Expfit: fit noisy signals with 1 to 4 exponentials

- With guessing of initial parameters
- With some tests
- A place to try out a few methods
  - Cobble something together that fits 1, then 2, then 3 etc.
  - Or just chuck it in a global opt, but with sensible constraints e.g. tau1 > tau2 > tau3 etc (unless that's not sensible, cause if it finds tau1=tau2 it'd be stuck until they are both near equal and the other can move again? might be better to order after finding?)
- Ideally with minimal dependencies (because for low number of exponentials (1--4) it shouldn't be super hard.

## Name

The name may be subobtimal, as its easy to type/think expkit (like myokit, datkit), and an [expkit-core](https://pypi.org/project/expkit-core/) already exists in pypi

## Future?

Maybe stand-alone. Maybe in pcpostprocess or datkit

For now, need somewhere to play and document.
