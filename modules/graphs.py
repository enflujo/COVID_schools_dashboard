import jax.numpy as np
import numpy as np2
from numpy.lib.function_base import interp

from networks import create_networks
from modules.graph_age_distribution import build as get_age_distribution
from modules.graph_age_params import cache as get_age_params
# from modules.graph_teachers_distribution import cache as get_teachers_distribution
from modules.graph_teachers_distribution import build as get_teachers_distribution
from modules.graph_household_sizes import cache as get_household_sizes
from modules.graph_classify_nodes import build as classify_nodes
from modules.graph_degree_distribution import build as get_degree_distribution


def create_graph_matrix(args):
    # Households
    household_sizes = get_household_sizes(args)
    ages = get_age_distribution(args)
    teachers = get_teachers_distribution(args)
    nodes = classify_nodes(args, ages, teachers)
    degrees = get_degree_distribution(args, ages, teachers, nodes)

    df_run_params = get_age_params(args)
    pop = args.population

    # Mask efficiencies in inhalation and exhalation taken from https://tinyurl.com/covid-estimator
    mask_inhalation = {"cloth": 0.5, "surgical": 0.3, "N95": 0.9}
    mask_exhalation = {"cloth": 0.5, "surgical": 0.65, "N95": 0.9}
    inhalation_mask = mask_inhalation[args.masks_type]
    exhalation_mask = mask_exhalation[args.masks_type]

    # Degree dist. mean and std div obtained by Prem et al data, scaled by 1/2.5 in order to ensure that community+friends+school = community data in Prem et al
    mean, std = args.community_mean, args.community_std
    p = 1 - (std ** 2 / mean)
    n_binom = mean / p
    community_degree = np2.random.binomial(n_binom, p, size=pop)

    # Split the age group of old population according to the population seen in the data
    prob = []
    for i in range(0, len(ages["community"][0])):
        prob.append(ages["community"][0][i] / sum(ages["community"][0]))

    age_group_community = np2.random.choice(np2.arange(0, len(ages["community"][0]), 1), size=pop, p=prob, replace=True)

    age_tracker_all = np2.zeros(int(pop*2))
    community_indx = np2.arange(0, int(pop), 1)
    for i in range((pop)):
        age_tracker_all[community_indx[i]] = age_group_community[i]

    matrix_household = create_networks.create_fully_connected(
        household_sizes, age_tracker_all, np2.arange(0, int(pop), 1), df_run_params, args.delta_t
    )

    age_tracker = degrees["age_tracker"]

    # Preschool
    matrix_preschool = create_networks.create_external_corr_schools(
        nodes["preschool"][1],
        degrees["preschool"][0],
        degrees["preschool"][2],
        args.preschool_r,
        nodes["preschool"][0],
        degrees["preschool"][1],
        age_tracker,
        df_run_params,
        args.delta_t,
        args.length_room_preschool,
        args.width_room_preschool,
        args.height_room_preschool,
        args.ventilation_out,
        inhalation_mask,
        exhalation_mask,
        args.fraction_people_masks,
        args.duration_event,
    )

    # Primary
    matrix_primary = create_networks.create_external_corr_schools(
        nodes["primary"][1],
        degrees["primary"][0],
        degrees["primary"][2],
        args.primary_r,
        nodes["primary"][0],
        degrees["primary"][1],
        age_tracker,
        df_run_params,
        args.delta_t,
        args.length_room_primary,
        args.width_room_primary,
        args.height_room_primary,
        args.ventilation_out,
        inhalation_mask,
        exhalation_mask,
        args.fraction_people_masks,
        args.duration_event,
    )

    # Highschool
    matrix_highschool = create_networks.create_external_corr_schools(
        nodes["highschool"][1],
        degrees["highschool"][0],
        degrees["highschool"][2],
        args.highschool_r,
        nodes["highschool"][0],
        degrees["highschool"][1],
        age_tracker,
        df_run_params,
        args.delta_t,
        args.length_room_highschool,
        args.width_room_highschool,
        args.height_room_highschool,
        args.ventilation_out,
        inhalation_mask,
        exhalation_mask,
        args.fraction_people_masks,
        args.duration_event,
    )

    # Work
    matrix_work = create_networks.create_external_corr(
        nodes["work"][1],
        degrees["work"][0],
        degrees["work"][2],
        args.work_r,
        nodes["work"][0],
        degrees["work"][1],
        age_tracker,
        df_run_params,
        args.delta_t,
    )

    # Community
    matrix_community = create_networks.create_external_corr(
        pop,
        community_degree,
        args.community_n,
        args.community_r,
        np2.arange(0, pop, 1),
        age_group_community,
        age_tracker,
        df_run_params,
        args.delta_t,
    )

    # Saves graphs
    return nodes, ages["total_pop"], [
        matrix_household,
        matrix_preschool,
        matrix_primary,
        matrix_highschool,
        matrix_work,
        matrix_community,
    ]
