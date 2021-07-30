import numpy as np2


def build(args, ages, teachers):
    pop = args.population
    pop_city = ages["total_pop"]
    preschool_pop_ = ages["preschool"][0] + teachers["preschool"][0]
    preschool_pop = sum(preschool_pop_)

    primary_pop_ = ages["primary"][0] + teachers["primary"][0]
    primary_pop = sum(primary_pop_)

    highschool_pop_ = ages["highschool"][0] + teachers["highschool"][0]
    highschool_pop = sum(highschool_pop_)

    work_pop_no_teachers = sum(ages["work"][0]) - teachers["total"]

    # Frac of population that is school going, working, preschool or elderly
    dist_of_pop = [
        preschool_pop / pop_city,
        primary_pop / pop_city,
        highschool_pop / pop_city,
        work_pop_no_teachers / pop_city,
        ages["very_young"][1] + ages["university"][1] + ages["elderly"][1],
    ]

    dist_of_pop[-1] += 1 - sum(dist_of_pop)

    # Classifying each person
    classify_pop = np2.random.choice(["preschool", "primary", "highschool", "work", "other"], size=pop, p=dist_of_pop)
    state, counts = np2.unique(classify_pop, return_counts=True)

    # Number of individuals in each group
    state, counts = np2.unique(classify_pop, return_counts=True)
    dict_of_counts = dict(zip(state, counts))

    # TODO: Revisar primero si el key existe
    return {
        "preschool": [np2.where(classify_pop == "preschool")[0], dict_of_counts["preschool"]],
        "primary": [np2.where(classify_pop == "primary")[0], dict_of_counts["primary"]],
        "highschool": [np2.where(classify_pop == "highschool")[0], dict_of_counts["highschool"]],
        "work": [np2.where(classify_pop == "work")[0], dict_of_counts["work"]],
        "other": [np2.where(classify_pop == "other")[0], dict_of_counts["other"]],
    }
