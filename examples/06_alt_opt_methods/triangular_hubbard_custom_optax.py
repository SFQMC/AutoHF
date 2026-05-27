import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
from jax import config

config.update("jax_enable_x64", True)

import numpy as np
# import matplotlib.pyplot as plt

import autohf
import jax
import optax
from typing import NamedTuple, Union
import optax.tree_utils as otu

hops = (np.array([ 0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,
                        2,  3,  3,  3,  3,  3,  3,  4,  4,  4,  4,  4,  4,  5,  5,  5,  5,
                        5,  5,  6,  6,  6,  6,  6,  6,  7,  7,  7,  7,  7,  7,  8,  8,  8,
                        8,  8,  8,  9,  9,  9,  9,  9,  9, 10, 10, 10, 10, 10, 10, 11, 11,
                       11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 14,
                       14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16,
                       17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19,
                       19, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22,
                       22, 22, 23, 23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 25, 25, 25,
                       25, 25, 25, 26, 26, 26, 26, 26, 26, 27, 27, 27, 27, 27, 27, 28, 28,
                       28, 28, 28, 28, 29, 29, 29, 29, 29, 29, 30, 30, 30, 30, 30, 30, 31,
                       31, 31, 31, 31, 31, 32, 32, 32, 32, 32, 32, 33, 33, 33, 33, 33, 33,
                       34, 34, 34, 34, 34, 34, 35, 35, 35, 35, 35, 35]),
         np.array([ 1,  5,  6, 11, 30, 31,  0,  2,  6,  7, 31, 32,  1,  3,  7,  8, 32,
                        33,  2,  4,  8,  9, 33, 34,  3,  5,  9, 10, 34, 35,  0,  4, 10, 11,
                        30, 35,  0,  1,  7, 11, 12, 17,  1,  2,  6,  8, 12, 13,  2,  3,  7,
                         9, 13, 14,  3,  4,  8, 10, 14, 15,  4,  5,  9, 11, 15, 16,  0,  5,
                         6, 10, 16, 17,  6,  7, 13, 17, 18, 23,  7,  8, 12, 14, 18, 19,  8,
                         9, 13, 15, 19, 20,  9, 10, 14, 16, 20, 21, 10, 11, 15, 17, 21, 22,
                         6, 11, 12, 16, 22, 23, 12, 13, 19, 23, 24, 29, 13, 14, 18, 20, 24,
                        25, 14, 15, 19, 21, 25, 26, 15, 16, 20, 22, 26, 27, 16, 17, 21, 23,
                        27, 28, 12, 17, 18, 22, 28, 29, 18, 19, 25, 29, 30, 35, 19, 20, 24,
                        26, 30, 31, 20, 21, 25, 27, 31, 32, 21, 22, 26, 28, 32, 33, 22, 23,
                        27, 29, 33, 34, 18, 23, 24, 28, 34, 35,  0,  5, 24, 25, 31, 35,  0,
                         1, 25, 26, 30, 32,  1,  2, 26, 27, 31, 33,  2,  3, 27, 28, 32, 34,
                         3,  4, 28, 29, 33, 35,  4,  5, 24, 29, 30, 34])
        )  # fmt: skip


#### Define Parameters
Nx, Ny = 6, 6
U = 8
Ne = 18
steps = 500
batch_size = 4

hf_settings = {
  "verbose": False,
  "steps": steps,
  "opt_method": "lbfgs",
  "ansatz": "SD_ROT",
  "batch_size": batch_size,
  "gpu": False,
  "seed": 42,
  "nelec": (Ne, Ne),
  "noncollinear": True,
  "state0_scale": 0.001,  # start close to U=0
}

#### Setup calculation
N = Nx * Ny
assert Nx == 6 and Ny == 6  # harded coded here
T = np.zeros((N, N))
T[hops] = -1

H = autohf.AutoHFHamiltonian(T=(T, T), U=U)

## Standard lbfgs to get a sense of things
data, data_f = autohf.solve_hf(
  H,
  settings=hf_settings,
)
E_lbfgs = data["E"]

# Gradient Descent via optax
opt = optax.chain(
  # Sets the parameters of Adam. Note the learning_rate is not here.
  optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8),
  # Puts a minus sign to *minimize* the loss.
  optax.scale(-0.01),
)

data, data_f = autohf.solve_hf(
  H,
  settings=hf_settings | dict(opt_method="grad"),
  optimizer=opt,
)
E_grad = data["E"]


# Now Construct custom optimization ala Optax examples
def run_opt(init_params, fun, opt, max_iter, tol):
  value_and_grad_fun = optax.value_and_grad_from_state(fun)

  def step(carry):
    params, state = carry
    value, grad = value_and_grad_fun(params, state=state)
    updates, state = opt.update(grad, state, params, value=value, grad=grad, value_fn=fun)
    params = optax.apply_updates(params, updates)
    return params, state

  def continuing_criterion(carry):
    _, state = carry
    iter_num = otu.tree_get(state, "count")
    grad = otu.tree_get(state, "grad")
    err = otu.tree_l1_norm(grad)
    return (iter_num == 0) | ((iter_num < max_iter) & (err >= tol))

  init_carry = (init_params, opt.init(init_params))
  final_params, final_state = jax.lax.while_loop(continuing_criterion, step, init_carry)
  return final_params, final_state


class InfoState(NamedTuple):
  iter_num: Union[np.number, np.ndarray, jax.Array]


def print_info():
  def init_fn(params):
    del params
    return InfoState(iter_num=0)

  def update_fn(updates, state, params, *, value, grad, **extra_args):
    del params, extra_args

    jax.debug.print(
      "Iteration: {i}, Value: {v}, Gradient norm: {e}",
      i=state.iter_num,
      v=value,
      e=otu.tree_l1_norm(grad),
    )
    return updates, InfoState(iter_num=state.iter_num + 1)

  return optax.GradientTransformationExtraArgs(init_fn, update_fn)


data, data_f = autohf.solve_hf(
  H,
  settings=hf_settings | dict(opt_method="grad", steps=-1),
)

rng = np.random.default_rng(hf_settings["seed"])
fun = data_f["energy_func"]
init_params = data["state"]
init_params += rng.normal(scale=0.001, size=init_params.shape)
opt = optax.chain(print_info(), optax.lbfgs())
print(
  f"Initial value: {fun(init_params):.2e} "
  f"Initial gradient norm: {otu.tree_l2_norm(jax.grad(fun)(init_params)):.2e}"
)
final_params, _ = run_opt(init_params, fun, opt, max_iter=hf_settings["steps"], tol=0.001)
E_final = fun(final_params)
print(
  f"Final value: {E_final:.2e}, "
  f"Final gradient norm: {otu.tree_l2_norm(jax.grad(fun)(final_params)):.2e}"
)


print("=================")
print("LBFGS:       \t", E_lbfgs)
print("Grad:        \t", E_grad)
print("custom lbfgs:\t", E_final)
