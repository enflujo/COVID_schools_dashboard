import jax.numpy as np
import pandas as pd


def build(args):
    # Get age distribution
    ages_data_BOG = pd.read_csv(args.ages_data_path, encoding="unicode_escape", delimiter=";")
    total_pop_BOG = int(ages_data_BOG["Total.3"][17].replace(".", ""))

    # Ages 0-4 (0)
    very_young_ = [int(ages_data_BOG["Total.3"][0].replace(".", ""))]

    # Ages 5-9 (1)
    preschool_ = [int(ages_data_BOG["Total.3"][1].replace(".", ""))]

    # Ages 10-14 (2)
    primary_ = [int(ages_data_BOG["Total.3"][2].replace(".", ""))]

    # Ages 15-19 (3)
    highschool_ = [int(ages_data_BOG["Total.3"][3].replace(".", ""))]

    # Ages 20-24 (4)
    university_ = [int(ages_data_BOG["Total.3"][4].replace(".", ""))]

    # Ages 25-64 (5,6,7,8,9,10,11,12)
    work_ = [int(ages_data_BOG["Total.3"][i].replace(".", "")) for i in range(5, 12 + 1)]

    # Ages 65+ (13,14,15,16)
    elderly_ = [int(ages_data_BOG["Total.3"][i].replace(".", "")) for i in range(13, 16 + 1)]

    # Community ages
    community_ = very_young_ + preschool_ + primary_ + highschool_ + university_ + work_ + elderly_

    return {
        "very_young": [very_young_, sum(very_young_) / total_pop_BOG],
        "preschool": [preschool_, sum(preschool_) / total_pop_BOG],
        "primary": [primary_, sum(primary_) / total_pop_BOG],
        "highschool": [highschool_, sum(highschool_) / total_pop_BOG],
        "university": [university_, sum(university_) / total_pop_BOG],
        "work": [work_, sum(work_) / total_pop_BOG],
        "elderly": [elderly_, sum(elderly_) / total_pop_BOG],
        "adults": np.arange(4, 16 + 1, 1),
        "community": [community_, sum(community_) / total_pop_BOG],
        "total_pop": total_pop_BOG,
    }


def cache(args):

    ########################### Static ages distribution ################################################
    ages = {
        "bogota": {
            "very_young": [[493287], 0.06369962118839792],
            "preschool": [[482823], 0.06234837366694409],
            "primary": [[490700], 0.06336555416450639],
            "highschool": [[550879], 0.07113664787566559],
            "university": [[711590], 0.09188973851216853],
            "work": [[749246, 673163, 613704, 539925, 477123, 468097, 435209, 352709], 0.556456745939252],
            "elderly": [[260464, 183141, 119787, 142108], 0.09110331865306552],
            "adults": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            "community": [
                [
                    493287,
                    482823,
                    490700,
                    550879,
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
            "total_pop": 7743955,
        }
    }

    return ages[args.city]
