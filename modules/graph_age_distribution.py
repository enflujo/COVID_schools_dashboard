import jax.numpy as np
import pandas as pd


def build(args):
    """
    Get age distribution only for layers not involved in school
    """

    # Get age distribution
    ages_data_BOG = pd.read_csv(args.ages_data_path, encoding="unicode_escape", delimiter=";")
    # total_pop_BOG = int(ages_data_BOG["Total.3"][17].replace(".", ""))

    # Ages 0-4 (0)
    very_young_ = [int(ages_data_BOG["Total.3"][0].replace(".", ""))]

    # Ages 20-24 (4)
    university_ = [int(ages_data_BOG["Total.3"][4].replace(".", ""))]

    # Ages 25-64 (5,6,7,8,9,10,11,12)
    work_ = [int(ages_data_BOG["Total.3"][i].replace(".", "")) for i in range(5, 12 + 1)]

    # Ages 65+ (13,14,15,16)
    elderly_ = [int(ages_data_BOG["Total.3"][i].replace(".", "")) for i in range(13, 16 + 1)]

    # Community ages
    community_ = very_young_ + university_ + work_ + elderly_

    return {
        "very_young": [very_young_, sum(very_young_) / sum(community_)],
        "university": [university_, sum(university_) / sum(community_)],
        "work": [work_, sum(work_) / sum(community_)],
        "elderly": [elderly_, sum(elderly_) / sum(community_)],
        "adults": np.arange(4, 16 + 1, 1).tolist(),
        "community": [community_, sum(community_) / sum(community_)],
        "total_pop": sum(community_),
    }


def cache(args):

    ########################### Static ages distribution ################################################
    ages = {
        "bogota": {
            "very_young": [[493287], 0.07931229141386849],
            "university": [[711590], 0.11441175917304668],
            "work": [[749246, 673163, 613704, 539925, 477123, 468097, 435209, 352709], 0.692843360286503],
            "elderly": [[260464, 183141, 119787, 142108], 0.11343258912658193],
            "adults": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            "community": [
                [
                    493287,
                    711590,
                    749246,
                    673163,
                    613704,
                    539925,
                    477123,
                    468097,
                    435209,
                    352709,
                    260464,
                    183141,
                    119787,
                    142108,
                ],
                1.0,
            ],
            "total_pop": 6219553,
        }
    }

    return ages[args.city]
