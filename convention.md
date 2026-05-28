# Conventions

The following are internal or external conventions within the code:
* states are pure jax arrays, stricter than pytrees
    * `states.shape = (# trials, **param_shape)`
    * this means that one can call `np.array(state)`
    * We pad partially polarized states to deal with this
* Orbitals are tuples
    * UHF have leading dim of size 2, GHF has leading dim of 1
    * This means `jax.vmap(orbFunc)(states)` produces a tuple with each element being the mapped index
* internally energy terms expect 1rdms in the format `(r_uu,r_dd)` or `(r_uu,r_dd,r_ud,r_du)`
    * however `makeRDMs` produces a (2,N,N) or (2N,2N) 1rdm

### External API
* `solve_hf` produces two dictionaries, one that can be pickled and one of jax functions
* In order to load a pickle, we need access to `Es`,`nelec`, `phi0` potentially, `state`/`states`
* To convert to SD we need `orbitals` and `all_orbitals` from `"dump_trials":True`
