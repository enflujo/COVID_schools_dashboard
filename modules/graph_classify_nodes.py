import numpy as np2


def build(args, ages, teachers):
    '''
    The creation of external and school layers will have n_pop in each,
    This means that the total number of nodes in the system is n_pop*2
    '''
    pop = args.population
    pop_city = ages["total_pop"]

    preschool_pop = args.n_school_going_preschool + args.n_teachers_preschool
    primary_pop = args.n_school_going_primary + args.n_teachers_primary
    highschool_pop = args.n_school_going_highschool + args.n_teachers_highschool
    teachers_pop = args.n_teachers_preschool + args.n_teachers_primary + args.n_teachers_highschool
    work_pop_no_teachers = sum(ages["work"][0]) - teachers_pop
    pop_input = preschool_pop + primary_pop + highschool_pop + teachers_pop

    # Frac of population that is work going, of community (other)
    dist_of_pop = [
        work_pop_no_teachers / pop_city,
        ages["very_young"][1] + ages["university"][1] + ages["elderly"][1],
    ]
    dist_of_pop[-1] += 1 - sum(dist_of_pop)

    # Classifying each person in external layers
    classify_pop_ext = np2.random.choice(["work", "other"], size=pop, p=dist_of_pop)
    # state_ext, counts_ext = np2.unique(classify_pop_ext, return_counts=True)

    # Frac of population that is school going
    school_pop = [int(preschool_pop), int(primary_pop), int(highschool_pop)]
    dist_of_pop = []
    for i, spop in enumerate(school_pop):
        if spop > 0:
            dist_of_pop.append(spop/pop_input)

    # Frac of population that is school going
    dist_of_pop[-1] += 1 - sum(dist_of_pop)

    # Classifying each person in school layers
    levels_exists = np2.array(school_pop) > 0
    levels_def = np2.array(["preschool", "primary", "highschool"])
    levels_sys = list(levels_def[levels_exists])
    classify_pop_schl = np2.random.choice(levels_sys, size=pop, p=dist_of_pop)
    # state_schl, counts_schl = np2.unique(classify_pop_schl, return_counts=True)

    # Concatenate nodes
    classify_pop = np2.concatenate((classify_pop_ext, classify_pop_schl))

    # Number of individuals in each group
    state, counts = np2.unique(classify_pop, return_counts=True)

    dict_of_counts = dict(zip(state, counts))

    # TODO: Revisar primero si el key existe
    res = {"work": [np2.where(classify_pop == "work")[0], dict_of_counts["work"]],
          "other": [np2.where(classify_pop == "other")[0], dict_of_counts["other"]]}
    for level in levels_sys:
        res[level] = [np2.where(classify_pop == level)[0], dict_of_counts[level]]
        
    return res


# def build(args, ages, teachers):
#     pop = args.population
#     pop_city = ages["total_pop"]
#     preschool_pop_ = ages["preschool"][0] + teachers["preschool"][0]
#     preschool_pop = sum(preschool_pop_)

#     primary_pop_ = ages["primary"][0] + teachers["primary"][0]
#     primary_pop = sum(primary_pop_)

#     highschool_pop_ = ages["highschool"][0] + teachers["highschool"][0]
#     highschool_pop = sum(highschool_pop_)

#     work_pop_no_teachers = sum(ages["work"][0]) - teachers["total"]

#     # Frac of population that is school going, working, preschool or elderly
#     dist_of_pop = [
#         preschool_pop / pop_city,
#         primary_pop / pop_city,
#         highschool_pop / pop_city,
#         work_pop_no_teachers / pop_city,
#         ages["very_young"][1] + ages["university"][1] + ages["elderly"][1],
#     ]

#     dist_of_pop[-1] += 1 - sum(dist_of_pop)

#     # Classifying each person
#     classify_pop = np2.random.choice(["preschool", "primary", "highschool", "work", "other"], size=pop, p=dist_of_pop)
#     state, counts = np2.unique(classify_pop, return_counts=True)

#     # Number of individuals in each group
#     state, counts = np2.unique(classify_pop, return_counts=True)
#     dict_of_counts = dict(zip(state, counts))

#     # TODO: Revisar primero si el key existe
#     return {
#         "preschool": [np2.where(classify_pop == "preschool")[0], dict_of_counts["preschool"]],
#         "primary": [np2.where(classify_pop == "primary")[0], dict_of_counts["primary"]],
#         "highschool": [np2.where(classify_pop == "highschool")[0], dict_of_counts["highschool"]],
#         "work": [np2.where(classify_pop == "work")[0], dict_of_counts["work"]],
#         "other": [np2.where(classify_pop == "other")[0], dict_of_counts["other"]],
#     }
