import pandas as pd
import os, json


def save(args, hvec, soln_ind, number_nodes, nodes):

    df_soln_ind_list = []
    for i in range(args.number_trials):
        inds_indx = [str(n) for n in range(0,number_nodes)]
        cols = ['iter','tvec']
        cols.extend(inds_indx)
        df_results_soln_ind_i = pd.DataFrame(columns=cols)
        df_results_soln_ind_i['iter']  = [i] * len(hvec)
        df_results_soln_ind_i['tvec']  = list(hvec)
        for ind in inds_indx:
            df_results_soln_ind_i[ind] = list(soln_ind[i,:,int(ind)])
        df_soln_ind_list.append(df_results_soln_ind_i)
    df_results_soln_ind = pd.concat(df_soln_ind_list)

    preschool_nodes = [str(n) for n in nodes['preschool'][0]]
    primary_nodes = [str(n) for n in nodes['primary'][0]]
    highschool_nodes = [str(n) for n in nodes['highschool'][0]]
    work_nodes = [str(n) for n in nodes['work'][0]]
    other_nodes = [str(n) for n in nodes['other'][0]]

    df_soln_ind = df_results_soln_ind.copy()
    # df_soln_ind_mode = df_soln_ind.groupby(by='iter', axis=0).agg(lambda x:x.value_counts().index[0])
    
    pre_inf_list = []
    pri_inf_list = []
    high_inf_list = []
    work_inf_list = []
    comm_inf_list = []
    for i in range(args.number_trials):
        df_soln_ind_i = df_soln_ind[df_soln_ind['iter']==i]

        pre_inf = df_soln_ind_i[preschool_nodes] == 2
        pre_inf = pre_inf.any()
        pre_inf = pre_inf[pre_inf == True].shape[0] # n infected in sim
        pre_inf_list.append(pre_inf)

        pri_inf = df_soln_ind_i[primary_nodes] == 2
        pri_inf = pri_inf.any()
        pri_inf = pri_inf[pri_inf == True].shape[0] # n infected in sim
        pri_inf_list.append(pri_inf)

        high_inf = df_soln_ind_i[highschool_nodes] == 2
        high_inf = high_inf.any()
        high_inf = high_inf[high_inf == True].shape[0] # n infected in sim
        high_inf_list.append(high_inf)

        work_inf = df_soln_ind_i[work_nodes] == 2
        work_inf = work_inf.any()
        work_inf = work_inf[work_inf == True].shape[0] # n infected in sim
        work_inf_list.append(work_inf)

        comm_inf = df_soln_ind_i[other_nodes] == 2
        comm_inf = comm_inf.any()
        comm_inf = comm_inf[comm_inf == True].shape[0] # n infected in sim
        comm_inf_list.append(comm_inf)

    out = {'preschool': sum(pre_inf_list)/len(pre_inf_list),
           'primary': sum(pri_inf_list)/len(pri_inf_list),
           'highschool': sum(high_inf_list)/len(high_inf_list),
           'work': sum(work_inf_list)/len(work_inf_list),
           'comm': sum(comm_inf_list)/len(comm_inf_list)
           }

    out_file = open('output.json', 'w')
    json.dump(out, out_file)
    out_file.close()



# def save(args, tvec, hvec, soln, soln_cum, history, soln_ind, number_nodes, nodes):
#     df_soln_list = []
#     for i in range(args.number_trials):
#         df_results_soln_i = pd.DataFrame(columns=["iter", "tvec", "S", "E", "I1", "I2", "I3", "D", "R"])
#         df_results_soln_i["iter"] = [i] * len(tvec)
#         df_results_soln_i["tvec"] = list(tvec)
#         df_results_soln_i["S"] = list(soln[i, :, 0])
#         df_results_soln_i["E"] = list(soln[i, :, 1])
#         df_results_soln_i["I1"] = list(soln[i, :, 2])
#         df_results_soln_i["I2"] = list(soln[i, :, 3])
#         df_results_soln_i["I3"] = list(soln[i, :, 4])
#         df_results_soln_i["D"] = list(soln[i, :, 5])
#         df_results_soln_i["R"] = list(soln[i, :, 6])
#         df_soln_list.append(df_results_soln_i)
#     df_results_soln = pd.concat(df_soln_list)

#     df_soln_cum_list = []
#     for i in range(args.number_trials):
#         df_results_soln_cum_i = pd.DataFrame(columns=["iter", "tvec", "S", "E", "I1", "I2", "I3", "D", "R"])
#         df_results_soln_cum_i["iter"] = [i] * len(tvec)
#         df_results_soln_cum_i["tvec"] = list(tvec)
#         df_results_soln_cum_i["S"] = list(soln_cum[i, :, 0])
#         df_results_soln_cum_i["E"] = list(soln_cum[i, :, 1])
#         df_results_soln_cum_i["I1"] = list(soln_cum[i, :, 2])
#         df_results_soln_cum_i["I2"] = list(soln_cum[i, :, 3])
#         df_results_soln_cum_i["I3"] = list(soln_cum[i, :, 4])
#         df_results_soln_cum_i["D"] = list(soln_cum[i, :, 5])
#         df_results_soln_cum_i["R"] = list(soln_cum[i, :, 6])
#         df_soln_cum_list.append(df_results_soln_cum_i)
#     df_results_soln_cum = pd.concat(df_soln_cum_list)

#     df_soln_ind_list = []
#     for i in range(args.number_trials):
#         inds_indx = [str(n) for n in range(0,number_nodes)]
#         cols = ['iter','tvec']
#         cols.extend(inds_indx)
#         df_results_soln_ind_i = pd.DataFrame(columns=cols)
#         df_results_soln_ind_i['iter']  = [i] * len(hvec)
#         df_results_soln_ind_i['tvec']  = list(hvec)
#         for ind in inds_indx:
#             df_results_soln_ind_i[ind] = list(soln_ind[i,:,int(ind)])
#         df_soln_ind_list.append(df_results_soln_ind_i)
#     df_results_soln_ind = pd.concat(df_soln_ind_list)

#     preschool_nodes = nodes['preschool']
#     primary_nodes = nodes['primary']
#     highscool_nodes = nodes['highscool']
#     work_nodes = nodes['work']
#     other_nodes = nodes['other']

#     df_soln_ind = df_results_soln_ind.copy()
#     df_soln_ind_mode = df_soln_ind.groupby(by='iter', axis=0).agg(lambda x:x.value_counts().index[0])
    
    

    # df_results_history = pd.DataFrame(columns=["tvec", "S", "E", "I1", "I2", "I3", "D", "R"])
    # df_results_history["tvec"] = list(tvec)
    # df_results_history["S"] = list(history[:, 0])
    # df_results_history["E"] = list(history[:, 1])
    # df_results_history["I1"] = list(history[:, 2])
    # df_results_history["I2"] = list(history[:, 3])
    # df_results_history["I3"] = list(history[:, 4])
    # df_results_history["D"] = list(history[:, 5])
    # df_results_history["R"] = list(history[:, 6])

    # df_results_com_history = pd.DataFrame(columns=["tvec", "S", "E", "I1", "I2", "I3", "D", "R"])
    # df_results_com_history["tvec"] = list(tvec)
    # df_results_com_history["S"] = list(cumulative_history[:, 0])
    # df_results_com_history["E"] = list(cumulative_history[:, 1])
    # df_results_com_history["I1"] = list(cumulative_history[:, 2])
    # df_results_com_history["I2"] = list(cumulative_history[:, 3])
    # df_results_com_history["I3"] = list(cumulative_history[:, 4])
    # df_results_com_history["D"] = list(cumulative_history[:, 5])
    # df_results_com_history["R"] = list(cumulative_history[:, 6])

    # intervention_save = None

    # if args.intervention_type == "no_intervention":
    #     intervention_save = "no_intervention"

    # elif args.intervention_type == "intervention":
    #     intervention_save = "intervention"

    # elif args.intervention_type == "school_alternancy":
    #     intervention_save = "school_alternancy"

    # else:
    #     print("No valid intervention type")
    # results_path = args.results_path

    # if not os.path.isdir(os.path.join(results_path, intervention_save, str(number_nodes))):
    #     os.makedirs(os.path.join(results_path, intervention_save, str(number_nodes)))

    # path_save = os.path.join(results_path, intervention_save, str(number_nodes))

    # df_results_soln.to_csv(
    #     path_save
    #     + "/{}_inter_{}_schoolcap_{}_mask_{}_peopleMasked_{}_ventilation_{}_ID_{}_soln.csv".format(
    #         str(number_nodes),
    #         str(args.intervention),
    #         str(args.school_occupation),
    #         args.masks_type,
    #         str(args.fraction_people_masks),
    #         str(args.ventilation_out),
    #         args.res_id,
    #     ),
    #     index=False,
    # )
    # df_results_soln_cum.to_csv(
    #     path_save
    #     + "/{}_inter_{}_schoolcap_{}_mask_{}_peopleMasked_{}_ventilation_{}_ID_{}_soln_cum.csv".format(
    #         str(number_nodes),
    #         str(args.intervention),
    #         str(args.school_occupation),
    #         args.masks_type,
    #         str(args.fraction_people_masks),
    #         str(args.ventilation_out),
    #         args.res_id,
    #     ),
    #     index=False,
    # )
    # df_results_history.to_csv(
    #     path_save
    #     + "/{}_inter_{}_schoolcap_{}_mask_{}_peopleMasked_{}_ventilation_{}_ID_{}_history.csv".format(
    #         str(number_nodes),
    #         str(args.intervention),
    #         str(args.school_occupation),
    #         args.masks_type,
    #         str(args.fraction_people_masks),
    #         str(args.ventilation_out),
    #         args.res_id,
    #     ),
    #     index=False,
    # )
    # df_results_com_history.to_csv(
    #     path_save
    #     + "/{}_inter_{}_schoolcap_{}_mask_{}_peopleMasked_{}_ventilation_{}_ID_{}_com_history.csv".format(
    #         str(number_nodes),
    #         str(args.intervention),
    #         str(args.school_occupation),
    #         args.masks_type,
    #         str(args.fraction_people_masks),
    #         str(args.ventilation_out),
    #         args.res_id,
    #     ),
    #     index=False,
    # )

    # with open(
    #     "{}/{}_inter_{}_schoolcap_{}_mask_{}_peopleMasked_{}_ventilation_{}_ID_{}_com_history.json".format(
    #         path_save,
    #         str(number_nodes),
    #         str(args.intervention),
    #         str(args.school_occupation),
    #         args.masks_type,
    #         str(args.fraction_people_masks),
    #         str(args.ventilation_out),
    #         args.res_id,
    #     ),
    #     "w",
    # ) as outfile:
    #     json.dump(df_results_com_history.to_dict(), outfile, indent=2)
