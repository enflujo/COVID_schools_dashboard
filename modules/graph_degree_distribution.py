import jax.numpy as np
import numpy as np2


def build(args, ages, teachers, nodes):
    age_tracker = np2.zeros(args.population)
    # Preschool -------------------------------------------------------
    mean, std = args.preschool_mean, args.preschool_std
    p = 1 - (std ** 2 / mean)
    n_binom = mean / p

    preschool_going = nodes["preschool"][1]
    preschool_degree = np2.random.binomial(n_binom, p, size=preschool_going)
    n_preschool = preschool_going / args.preschool_size

    preschool_clroom = np2.random.choice(np.arange(0, n_preschool + 1, 1), size=preschool_going)

    # Assign ages to the preschool going population acc. to their proportion from the census data
    prob = []
    preschool_pop_ = ages["preschool"][0] + teachers["preschool"][0]
    preschool_pop = sum(preschool_pop_)

    for i in range(0, len(preschool_pop_)):
        prob.append(preschool_pop_[i] / preschool_pop)
    age_group_preschool = np2.random.choice(np.array([1, 7]), size=preschool_going, p=prob, replace=True)

    for i in range(preschool_going):
        age_tracker[nodes["preschool"][0][i]] = age_group_preschool[i]

    # Primary ---------------------------------------------------------
    mean, std = args.primary_mean, args.primary_std
    p = 1 - (std ** 2 / mean)
    n_binom = mean / p

    primary_going = nodes["primary"][1]
    primary_degree = np2.random.binomial(n_binom, p, size=primary_going)
    n_primary = primary_going / args.primary_size

    primary_clroom = np2.random.choice(np.arange(0, n_primary + 1, 1), size=primary_going)

    # Assign ages to the primary going population acc. to their proportion from the census data
    prob = []
    primary_pop_ = ages["primary"][0] + teachers["primary"][0]
    primary_pop = sum(primary_pop_)

    for i in range(0, len(primary_pop_)):
        prob.append(primary_pop_[i] / primary_pop)
    age_group_primary = np2.random.choice(np.array([2, 7]), size=primary_going, p=prob, replace=True)

    for i in range(primary_going):
        age_tracker[nodes["primary"][0][i]] = age_group_primary[i]

    # Highschool -------------------------------------------------------
    mean, std = args.highschool_mean, args.highschool_std
    p = 1 - (std ** 2 / mean)
    n_binom = mean / p

    highschool_going = nodes["highschool"][1]
    highschool_degree = np2.random.binomial(n_binom, p, size=highschool_going)
    n_highschool = highschool_going / args.highschool_size

    highschool_clroom = np2.random.choice(np.arange(0, n_highschool + 1, 1), size=highschool_going)

    # Assign ages to the highschool going population acc. to their proportion from the census data
    prob = []
    highschool_pop_ = ages["highschool"][0] + teachers["highschool"][0]
    highschool_pop = sum(highschool_pop_)

    for i in range(0, len(highschool_pop_)):
        prob.append(highschool_pop_[i] / highschool_pop)
    age_group_highschool = np2.random.choice(np.array([3, 7]), size=highschool_going, p=prob, replace=True)

    for i in range(highschool_going):
        age_tracker[nodes["highschool"][0][i]] = age_group_highschool[i]

    # Work -----------------------------------------------------------
    # Degree dist., the mean and std div have been taken from the Potter et al data. The factor of 1/3 is used to correspond to daily values and is chosen to match with the work contact survey data
    mean, std = args.work_mean, args.work_std
    p = 1 - (std ** 2 / mean)
    n_binom = mean / p

    working_going = nodes["work"][1]
    work_degree = np2.random.binomial(n_binom, p, size=working_going)

    # Assuming that on average the size of a work place is ~ 10 people and the correlation is
    # chosen such that the clustering coeff is high as the network in Potter et al had a pretty high value
    work_place_size = args.work_size
    n_work = working_going / work_place_size

    # Assign each working individual a 'work place'
    job_place = np2.random.choice(np.arange(0, n_work + 1, 1), size=working_going)

    # Split the age group of working population according to the populapreschool_tion seen in the data
    p = []
    work_pop_ = ages["university"][0] + ages["work"][0]
    work_pop = sum(work_pop_)

    for i in range(0, len(work_pop_)):
        p.append(work_pop_[i] / work_pop)
    age_group_work = np2.random.choice(np.arange(4, 12 + 1, 1), size=working_going, p=p, replace=True)

    for i in range(working_going):
        age_tracker[nodes["work"][0][i]] = age_group_work[i]

    return {
        "preschool": [preschool_degree, preschool_clroom, n_preschool],
        "primary": [primary_degree, primary_clroom, n_primary],
        "highschool": [highschool_degree, highschool_clroom, n_highschool],
        "work": [work_degree, job_place, n_work],
        "age_tracker": age_tracker,
    }
