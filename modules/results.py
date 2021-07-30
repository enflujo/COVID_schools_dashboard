import pandas as pd
import os, json


def save(args, tvec, soln, soln_cum, history, cumulative_history, number_nodes):
    df_soln_list = []
    for i in range(args.number_trials):
        df_results_soln_i = pd.DataFrame(columns=["iter", "tvec", "S", "E", "I1", "I2", "I3", "D", "R"])
        df_results_soln_i["iter"] = [i] * len(tvec)
        df_results_soln_i["tvec"] = list(tvec)
        df_results_soln_i["S"] = list(soln[i, :, 0])
        df_results_soln_i["E"] = list(soln[i, :, 1])
        df_results_soln_i["I1"] = list(soln[i, :, 2])
        df_results_soln_i["I2"] = list(soln[i, :, 3])
        df_results_soln_i["I3"] = list(soln[i, :, 4])
        df_results_soln_i["D"] = list(soln[i, :, 5])
        df_results_soln_i["R"] = list(soln[i, :, 6])
        df_soln_list.append(df_results_soln_i)
    df_results_soln = pd.concat(df_soln_list)

    df_soln_cum_list = []
    for i in range(args.number_trials):
        df_results_soln_cum_i = pd.DataFrame(columns=["iter", "tvec", "S", "E", "I1", "I2", "I3", "D", "R"])
        df_results_soln_cum_i["iter"] = [i] * len(tvec)
        df_results_soln_cum_i["tvec"] = list(tvec)
        df_results_soln_cum_i["S"] = list(soln_cum[i, :, 0])
        df_results_soln_cum_i["E"] = list(soln_cum[i, :, 1])
        df_results_soln_cum_i["I1"] = list(soln_cum[i, :, 2])
        df_results_soln_cum_i["I2"] = list(soln_cum[i, :, 3])
        df_results_soln_cum_i["I3"] = list(soln_cum[i, :, 4])
        df_results_soln_cum_i["D"] = list(soln_cum[i, :, 5])
        df_results_soln_cum_i["R"] = list(soln_cum[i, :, 6])
        df_soln_cum_list.append(df_results_soln_cum_i)
    df_results_soln_cum = pd.concat(df_soln_cum_list)

    df_results_history = pd.DataFrame(columns=["tvec", "S", "E", "I1", "I2", "I3", "D", "R"])
    df_results_history["tvec"] = list(tvec)
    df_results_history["S"] = list(history[:, 0])
    df_results_history["E"] = list(history[:, 1])
    df_results_history["I1"] = list(history[:, 2])
    df_results_history["I2"] = list(history[:, 3])
    df_results_history["I3"] = list(history[:, 4])
    df_results_history["D"] = list(history[:, 5])
    df_results_history["R"] = list(history[:, 6])

    df_results_com_history = pd.DataFrame(columns=["tvec", "S", "E", "I1", "I2", "I3", "D", "R"])
    df_results_com_history["tvec"] = list(tvec)
    df_results_com_history["S"] = list(cumulative_history[:, 0])
    df_results_com_history["E"] = list(cumulative_history[:, 1])
    df_results_com_history["I1"] = list(cumulative_history[:, 2])
    df_results_com_history["I2"] = list(cumulative_history[:, 3])
    df_results_com_history["I3"] = list(cumulative_history[:, 4])
    df_results_com_history["D"] = list(cumulative_history[:, 5])
    df_results_com_history["R"] = list(cumulative_history[:, 6])

    intervention_save = None

    if args.intervention_type == "no_intervention":
        intervention_save = "no_intervention"

    elif args.intervention_type == "intervention":
        intervention_save = "intervention"

    elif args.intervention_type == "school_alternancy":
        intervention_save = "school_alternancy"

    else:
        print("No valid intervention type")
    results_path = args.results_path

    if not os.path.isdir(os.path.join(results_path, intervention_save, str(number_nodes))):
        os.makedirs(os.path.join(results_path, intervention_save, str(number_nodes)))

    path_save = os.path.join(results_path, intervention_save, str(number_nodes))

    df_results_soln.to_csv(
        path_save
        + "/{}_inter_{}_schoolcap_{}_mask_{}_peopleMasked_{}_ventilation_{}_ID_{}_soln.csv".format(
            str(number_nodes),
            str(args.intervention),
            str(args.school_occupation),
            args.masks_type,
            str(args.fraction_people_masks),
            str(args.ventilation_out),
            args.res_id,
        ),
        index=False,
    )
    df_results_soln_cum.to_csv(
        path_save
        + "/{}_inter_{}_schoolcap_{}_mask_{}_peopleMasked_{}_ventilation_{}_ID_{}_soln_cum.csv".format(
            str(number_nodes),
            str(args.intervention),
            str(args.school_occupation),
            args.masks_type,
            str(args.fraction_people_masks),
            str(args.ventilation_out),
            args.res_id,
        ),
        index=False,
    )
    df_results_history.to_csv(
        path_save
        + "/{}_inter_{}_schoolcap_{}_mask_{}_peopleMasked_{}_ventilation_{}_ID_{}_history.csv".format(
            str(number_nodes),
            str(args.intervention),
            str(args.school_occupation),
            args.masks_type,
            str(args.fraction_people_masks),
            str(args.ventilation_out),
            args.res_id,
        ),
        index=False,
    )
    df_results_com_history.to_csv(
        path_save
        + "/{}_inter_{}_schoolcap_{}_mask_{}_peopleMasked_{}_ventilation_{}_ID_{}_com_history.csv".format(
            str(number_nodes),
            str(args.intervention),
            str(args.school_occupation),
            args.masks_type,
            str(args.fraction_people_masks),
            str(args.ventilation_out),
            args.res_id,
        ),
        index=False,
    )

    with open(
        "{}/{}_inter_{}_schoolcap_{}_mask_{}_peopleMasked_{}_ventilation_{}_ID_{}_com_history.json".format(
            path_save,
            str(number_nodes),
            str(args.intervention),
            str(args.school_occupation),
            args.masks_type,
            str(args.fraction_people_masks),
            str(args.ventilation_out),
            args.res_id,
        ),
        "w",
    ) as outfile:
        json.dump(df_results_com_history.to_dict(), outfile, indent=2)
