import pandas as pd


def build(args):
    # Get number of teachers per education level
    n_teachers_preschool = args.n_teachers_preschool
    n_teachers_primary = args.n_teachers_primary
    n_teachers_highschool = args.n_teachers_highschool

    total_teachers_system = n_teachers_preschool + n_teachers_primary + n_teachers_highschool

    teachers_preschool_ = [int(n_teachers_preschool)]
    teachers_primary_ = [int(n_teachers_primary)]
    teachers_highschool_ = [int(n_teachers_highschool)]

    return {
        "preschool": [teachers_preschool_, sum(teachers_preschool_) / total_teachers_system],
        "primary": [teachers_primary_, sum(teachers_primary_) / total_teachers_system],
        "highschool": [teachers_highschool_, sum(teachers_highschool_) / total_teachers_system],
        "total": total_teachers_system,
    }


def cache(args):
    ########################### Static teachers distribution ################################################
    teachers = {
        "bogota": {
            "preschool": [[5], 0.2777777777777778],
            "primary": [[6], 0.3333333333333333],
            "highschool": [[7], 0.3888888888888889],
            "total": 18,
        }
    }

    return teachers[args.city]
