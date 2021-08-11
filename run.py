#!
import modules.simulate as sim
from modules.results import save
from modules.graphs import create_graph_matrix
from modules.dynamics import create_dynamics
import jax.numpy as np
from timeit import default_timer as timer
from datetime import timedelta


async def process(args):
    pop = args.population
    Tmax = args.Tmax
    days_intervals = [1] * Tmax
    delta_t = args.delta_t
    step_intervals = [int(x / delta_t) for x in days_intervals]
    total_steps = sum(step_intervals)
    tvec = np.linspace(0, Tmax, total_steps)
    time_intervals = 2400
    hvec = np.arange(4,sum(time_intervals)+4,4)

    total_start = timer()
    print("Creating graphs...")
    start = timer()
    nodes, total_pop, multilayer_matrix = create_graph_matrix(args)
    end = timer()
    print("Graphs created in: {0}".format(str(timedelta(seconds=(end - start)))))

    print("Creating dynamics...")
    start = timer()
    time_intervals, ws = create_dynamics(args, multilayer_matrix, Tmax, total_steps)
    end = timer()
    print("Dynamics created in: {0}".format(str(timedelta(seconds=(end - start)))))

    print("Simulating...")
    start = timer()
    history, soln, soln_ind, soln_cum = sim.simulate(args, total_steps, pop, total_pop, ws, time_intervals)
    end = timer()
    print("Simulation created in: {0}".format(str(timedelta(seconds=(end - start)))))

    print("Saving results...")
    start = timer()
    save(args, tvec, hvec, soln, soln_cum, history, soln_ind, pop, nodes)
    end = timer()
    print("Saved results in: {0}".format(str(timedelta(seconds=(end - start)))))

    total_end = timer()
    print("Done, total process in: {0}".format(str(timedelta(seconds=(total_end - total_start)))))
