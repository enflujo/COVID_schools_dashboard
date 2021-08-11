import jax.numpy as np
from jax import random
from jax.ops import index_add, index_update, index
from jax import jit, random
import functools
from tqdm import tqdm
from models import model

# Bogota data

cum_cases = 632532
cum_rec = 593329
mild_house = 17595
hosp_beds = 5369
ICU_beds = 1351
deaths = 13125
# Model parameter values


DurMildInf = 6  # Duration of mild infections, days
DurSevereInf = 6  # Duration of hospitalization (severe infection), days
# Time from ICU admission to death/recovery (critical infection), days
DurCritInf = 8

# Standard deviations
std_IncubPeriod = 4  # Incubation period, days
std_DurMildInf = 2  # Duration of mild infections, days
std_DurSevereInf = 4.5  # Duration of hospitalization (severe infection), days
# Time from ICU admission to death/recovery (critical infection), days
std_DurCritInf = 6

FracSevere = 0.15  # Fraction of infections that are severe
FracCritical = 0.05  # Fraction of infections that are critical
CFR = 0.02  # Case fatality rate (fraction of infections resulting in death)
FracMild = 1 - FracSevere - FracCritical  # Fraction of infections that are mild


# Define transition probabilities

# Define probability of recovering (as opposed to progressing or dying) from each state
recovery_probabilities = np.array(
    [0.0, 0.0, FracMild, FracSevere / (FracSevere + FracCritical), 1.0 - CFR / FracCritical, 0.0, 0.0]
)
# Means
IncubPeriod = 5  # Incubation period, days
# Get gamma distribution parameters
mean_vec = np.array([1.0, IncubPeriod, DurMildInf, DurSevereInf, DurCritInf, 1.0, 1.0])
std_vec = np.array([1.0, std_IncubPeriod, std_DurMildInf, std_DurSevereInf, std_DurCritInf, 1.0, 1.0])

# This will contain scale values for each state
scale_vec = (std_vec ** 2) / mean_vec
# Define relative infectivity of each state
infection_probabilities = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
# This will contain shape values for each state
shape_vec = (mean_vec / std_vec) ** 2


@functools.partial(jit, static_argnums=(3,))
def _discrete_gamma(key, alpha, beta, shape=()):
    samples = np.round(random.gamma(key, alpha, shape=shape) / beta)
    return samples.astype(np.int32)


def discrete_gamma(key, alpha, beta, shape=()):
    shape_ = shape
    if shape_ == ():
        try:
            shape_ = alpha.shape
        except:
            shape_ = ()
    return _discrete_gamma(key, alpha, beta, shape_)


def simulate(args, total_steps, pop, total_pop_BOG, ws, time_intervals):
    @jit
    def state_length_sampler(key, new_state):
        """Duration in transitional state. Must be at least 1 time unit."""
        alphas = shape_vec[new_state]
        betas = args.delta_t / scale_vec[new_state]
        key, subkey = random.split(key)
        # Time must be at least 1.
        lengths = 1 + discrete_gamma(subkey, alphas, betas)
        # Makes sure non-transitional states are returning 0.
        return key, lengths * model.is_transitional(new_state)

    BOG_E = int(pop * (cum_cases - cum_rec - mild_house - deaths) / total_pop_BOG)
    # Assuming that 30% of population is already recovered
    BOG_R = int(pop * 0.3)
    BOG_I1 = int(pop * mild_house / total_pop_BOG)
    BOG_I2 = int(pop * hosp_beds / total_pop_BOG)
    BOG_I3 = int(pop * ICU_beds / total_pop_BOG)
    BOG_D = int(pop * deaths / total_pop_BOG)

    soln = np.zeros((args.number_trials, total_steps, 7))
    soln_cum = np.zeros((args.number_trials, total_steps, 7))
    soln_ind=np.zeros((args.number_trials,len(time_intervals),pop), dtype=np.int8)

    for key in tqdm(range(args.number_trials), total=args.number_trials):

        # Initial condition
        init_ind_E = random.uniform(random.PRNGKey(key), shape=(BOG_E,), maxval=pop).astype(np.int32)
        init_ind_I1 = random.uniform(random.PRNGKey(key), shape=(BOG_I1,), maxval=pop).astype(np.int32)
        init_ind_I2 = random.uniform(random.PRNGKey(key), shape=(BOG_I2,), maxval=pop).astype(np.int32)
        init_ind_I3 = random.uniform(random.PRNGKey(key), shape=(BOG_I3,), maxval=pop).astype(np.int32)
        init_ind_D = random.uniform(random.PRNGKey(key), shape=(BOG_D,), maxval=pop).astype(np.int32)
        init_ind_R = random.uniform(random.PRNGKey(key), shape=(BOG_R,), maxval=pop).astype(np.int32)
        init_state = np.zeros(pop, dtype=np.int32)
        init_state = index_update(init_state, init_ind_E, np.ones(BOG_E, dtype=np.int32) * 1)  # E
        init_state = index_update(init_state, init_ind_I1, np.ones(BOG_I1, dtype=np.int32) * 2)  # I1
        init_state = index_update(init_state, init_ind_I2, np.ones(BOG_I2, dtype=np.int32) * 3)  # I2
        init_state = index_update(init_state, init_ind_I3, np.ones(BOG_I3, dtype=np.int32) * 4)  # I3
        init_state = index_update(init_state, init_ind_D, np.ones(BOG_D, dtype=np.int32) * 5)  # D
        init_state = index_update(init_state, init_ind_R, np.ones(BOG_R, dtype=np.int32) * 6)  # R

        _, init_state_timer = state_length_sampler(random.PRNGKey(key), init_state)

        # Run simulation
        _, state, _, states_evolution, total_history = model.simulate_intervals(
            ws,
            time_intervals,
            state_length_sampler,
            infection_probabilities,
            recovery_probabilities,
            init_state,
            init_state_timer,
            key=random.PRNGKey(key),
            epoch_len=1,
        )

        # This unpacks current state counts
        history = np.array(total_history)[:, 0, :]
        soln = index_add(soln, index[key, :, :], history)
        soln_ind=index_add(soln_ind,index[key,:, :],states_evolution)

        # cumulative_history = np.array(total_history)[:, 1, :]
        # soln_cum = index_add(soln_cum, index[key, :, :], cumulative_history)

    return history, soln, soln_ind, soln_cum
