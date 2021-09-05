import pandas as pd


def run(number_trials, hvec, soln_ind, number_nodes, nodes):

    df_soln_ind_list = []

    # iter (iteración) | tvec (vector tiempo) | persona1 | persona2 | ....
    #   0              |         4            |   0-6    |   0-6    | ....
    #   0              |         8            |   0-6    |   0-6    | ....

    for i in range(number_trials):
        inds_indx = [str(n) for n in range(0, number_nodes)]
        cols = ["iter", "tvec"]
        cols.extend(inds_indx)
        df_results_soln_ind_i = pd.DataFrame(columns=cols)
        df_results_soln_ind_i["iter"] = [i] * len(hvec)
        df_results_soln_ind_i["tvec"] = list(hvec)

        for ind in inds_indx:
            df_results_soln_ind_i[ind] = list(soln_ind[i, :, int(ind)])

        df_soln_ind_list.append(df_results_soln_ind_i)

    df_results_soln_ind = pd.concat(df_soln_ind_list)

    # df_results_soln_ind.to_csv("res.csv", index=False)

    preschool_nodes = [str(n) for n in nodes["preschool"][0]]
    primary_nodes = [str(n) for n in nodes["primary"][0]]
    highschool_nodes = [str(n) for n in nodes["highschool"][0]]
    work_nodes = [str(n) for n in nodes["work"][0]]
    other_nodes = [str(n) for n in nodes["other"][0]]

    df_soln_ind = df_results_soln_ind.copy()

    pre_inf_list = []
    pri_inf_list = []
    high_inf_list = []
    work_inf_list = []
    comm_inf_list = []

    for i in range(number_trials):
        df_soln_ind_i = df_soln_ind[df_soln_ind["iter"] == i]

        pre_inf = df_soln_ind_i[preschool_nodes] == 2
        pre_inf = pre_inf.any()
        pre_inf = pre_inf[pre_inf == True].shape[0]  # n infected in sim
        pre_inf_list.append(pre_inf / nodes["preschool"][1])

        pri_inf = df_soln_ind_i[primary_nodes] == 2
        pri_inf = pri_inf.any()
        pri_inf = pri_inf[pri_inf == True].shape[0]  # n infected in sim
        pri_inf_list.append(pri_inf / nodes["primary"][1])

        high_inf = df_soln_ind_i[highschool_nodes] == 2
        high_inf = high_inf.any()
        high_inf = high_inf[high_inf == True].shape[0]  # n infected in sim
        high_inf_list.append(high_inf / nodes["highschool"][1])

        work_inf = df_soln_ind_i[work_nodes] == 2
        work_inf = work_inf.any()
        work_inf = work_inf[work_inf == True].shape[0]  # n infected in sim
        work_inf_list.append(work_inf / nodes["work"][1])

        comm_inf = df_soln_ind_i[other_nodes] == 2
        comm_inf = comm_inf.any()
        comm_inf = comm_inf[comm_inf == True].shape[0]  # n infected in sim
        comm_inf_list.append(comm_inf / nodes["other"][1])

    return {
        "preschool": sum(pre_inf_list) / len(pre_inf_list),
        "primary": sum(pri_inf_list) / len(pri_inf_list),
        "highschool": sum(high_inf_list) / len(high_inf_list),
        "work": sum(work_inf_list) / len(work_inf_list),
        "comm": sum(comm_inf_list) / len(comm_inf_list),
    }
