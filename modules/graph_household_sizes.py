import jax.numpy as np
import numpy as np2
import pandas as pd


def build(args):
    # Get household size distribution from 2018 census data
    census_data_BOG = pd.read_csv(args.houses_data_path)
    one_house = np2.sum(census_data_BOG["HA_TOT_PER"] == 1.0)
    two_house = np2.sum(census_data_BOG["HA_TOT_PER"] == 2.0)
    three_house = np2.sum(census_data_BOG["HA_TOT_PER"] == 3.0)
    four_house = np2.sum(census_data_BOG["HA_TOT_PER"] == 4.0)
    five_house = np2.sum(census_data_BOG["HA_TOT_PER"] == 5.0)
    six_house = np2.sum(census_data_BOG["HA_TOT_PER"] == 6.0)
    seven_house = np2.sum(census_data_BOG["HA_TOT_PER"] == 7.0)
    total_house = one_house + two_house + three_house + four_house + five_house + six_house + seven_house

    return np2.array([one_house, two_house, three_house, four_house, five_house, six_house, seven_house]) / total_house


def cache(args):
    pop = args.population

    ########################### Static size distribution ################################################
    sizes_dist = {
        "bogota": [
            0.2185412568365075,
            0.2328740032970561,
            0.23781638487965906,
            0.19472046680806057,
            0.0792078098115842,
            0.02714306645433561,
            0.00969701191279695,
        ]
    }

    house_size_dist = sizes_dist[args.city]

    # House-hold sizes
    household_sizes = []

    household_sizes.extend(
        np2.random.choice(np.arange(1, 8, 1), p=house_size_dist, size=int(pop / 3))
    )  # This division is just to make the code faster
    pop_house = sum(household_sizes)

    while pop_house <= pop:
        size = np2.random.choice(np.arange(1, 8, 1), p=house_size_dist, size=1)
        household_sizes.extend(size)
        pop_house += size[0]

    household_sizes[-1] -= pop_house - pop

    # # Mean of household degree dist
    # mean_household = sum((np2.asarray(household_sizes)-1)*np2.asarray(household_sizes))/pop

    # # Keeping track of the household indx for each individual
    # house_indices = np2.repeat(np2.arange(0,len(household_sizes),1), household_sizes)

    # # Keeping track of the household size for each individual
    # track_house_size = np2.repeat(household_sizes, household_sizes)

    return household_sizes
