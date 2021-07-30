import pandas as pd
import numpy as np2


def build(args):
    # Get medians
    def get_medians(df_p, last):
        df_res = df_p.iloc[-last:].groupby(["param"]).median().reset_index()["median"][0]
        return df_res

    def medians_params(df_list, age_group, last):
        params_def = ["age", "beta", "IFR", "RecPeriod", "alpha", "sigma"]
        params_val = [
            age_group,
            get_medians(df_list[0], last),
            get_medians(df_list[1], last),
            get_medians(df_list[2], last),
            get_medians(df_list[3], last),
            get_medians(df_list[4], last),
        ]
        res = dict(zip(params_def, params_val))
        return res

    params_data_BOG = pd.read_csv(args.params_data_path, encoding="unicode_escape", delimiter=",")

    # Ages 0-19
    young_ages_params = pd.DataFrame(params_data_BOG[params_data_BOG["age_group"] == "0-19"])
    young_ages_beta = pd.DataFrame(young_ages_params[young_ages_params["param"] == "contact_rate"])
    young_ages_IFR = pd.DataFrame(young_ages_params[young_ages_params["param"] == "IFR"])
    young_ages_RecPeriod = pd.DataFrame(young_ages_params[young_ages_params["param"] == "recovery_period"])
    young_ages_alpha = pd.DataFrame(young_ages_params[young_ages_params["param"] == "report_rate"])
    young_ages_sigma = pd.DataFrame(young_ages_params[young_ages_params["param"] == "relative_asymp_transmission"])
    young_params = [young_ages_beta, young_ages_IFR, young_ages_RecPeriod, young_ages_alpha, young_ages_sigma]

    # Ages 20-39
    youngAdults_ages_params = pd.DataFrame(params_data_BOG[params_data_BOG["age_group"] == "20-39"])
    youngAdults_ages_beta = pd.DataFrame(youngAdults_ages_params[youngAdults_ages_params["param"] == "contact_rate"])
    youngAdults_ages_IFR = pd.DataFrame(youngAdults_ages_params[youngAdults_ages_params["param"] == "IFR"])
    youngAdults_ages_RecPeriod = pd.DataFrame(
        youngAdults_ages_params[youngAdults_ages_params["param"] == "recovery_period"]
    )
    youngAdults_ages_alpha = pd.DataFrame(youngAdults_ages_params[youngAdults_ages_params["param"] == "report_rate"])
    youngAdults_ages_sigma = pd.DataFrame(
        youngAdults_ages_params[youngAdults_ages_params["param"] == "relative_asymp_transmission"]
    )
    youngAdults_params = [
        youngAdults_ages_beta,
        youngAdults_ages_IFR,
        youngAdults_ages_RecPeriod,
        youngAdults_ages_alpha,
        youngAdults_ages_sigma,
    ]

    # Ages 40-49
    adults_ages_params = pd.DataFrame(params_data_BOG[params_data_BOG["age_group"] == "40-49"])
    adults_ages_beta = pd.DataFrame(adults_ages_params[adults_ages_params["param"] == "contact_rate"])
    adults_ages_IFR = pd.DataFrame(adults_ages_params[adults_ages_params["param"] == "IFR"])
    adults_ages_RecPeriod = pd.DataFrame(adults_ages_params[adults_ages_params["param"] == "recovery_period"])
    adults_ages_alpha = pd.DataFrame(adults_ages_params[adults_ages_params["param"] == "report_rate"])
    adults_ages_sigma = pd.DataFrame(adults_ages_params[adults_ages_params["param"] == "relative_asymp_transmission"])
    adults_params = [adults_ages_beta, adults_ages_IFR, adults_ages_RecPeriod, adults_ages_alpha, adults_ages_sigma]

    # Ages 50-59
    seniorAdults_ages_params = pd.DataFrame(params_data_BOG[params_data_BOG["age_group"] == "50-59"])
    seniorAdults_ages_beta = pd.DataFrame(seniorAdults_ages_params[seniorAdults_ages_params["param"] == "contact_rate"])
    seniorAdults_ages_IFR = pd.DataFrame(seniorAdults_ages_params[seniorAdults_ages_params["param"] == "IFR"])
    seniorAdults_ages_RecPeriod = pd.DataFrame(
        seniorAdults_ages_params[seniorAdults_ages_params["param"] == "recovery_period"]
    )
    seniorAdults_ages_alpha = pd.DataFrame(seniorAdults_ages_params[seniorAdults_ages_params["param"] == "report_rate"])
    seniorAdults_ages_sigma = pd.DataFrame(
        seniorAdults_ages_params[seniorAdults_ages_params["param"] == "relative_asymp_transmission"]
    )
    seniorAdults_params = [
        seniorAdults_ages_beta,
        seniorAdults_ages_IFR,
        seniorAdults_ages_RecPeriod,
        seniorAdults_ages_alpha,
        seniorAdults_ages_sigma,
    ]

    # Ages 60-69
    senior_ages_params = pd.DataFrame(params_data_BOG[params_data_BOG["age_group"] == "60-69"])
    senior_ages_beta = pd.DataFrame(senior_ages_params[senior_ages_params["param"] == "contact_rate"])
    senior_ages_IFR = pd.DataFrame(senior_ages_params[senior_ages_params["param"] == "IFR"])
    senior_ages_RecPeriod = pd.DataFrame(senior_ages_params[senior_ages_params["param"] == "recovery_period"])
    senior_ages_alpha = pd.DataFrame(senior_ages_params[senior_ages_params["param"] == "report_rate"])
    senior_ages_sigma = pd.DataFrame(senior_ages_params[senior_ages_params["param"] == "relative_asymp_transmission"])
    senior_params = [senior_ages_beta, senior_ages_IFR, senior_ages_RecPeriod, senior_ages_alpha, senior_ages_sigma]

    # Ages 70+
    elderly_ages_params = pd.DataFrame(params_data_BOG[params_data_BOG["age_group"] == "70-90+"])
    elderly_ages_beta = pd.DataFrame(elderly_ages_params[elderly_ages_params["param"] == "contact_rate"])
    elderly_ages_IFR = pd.DataFrame(elderly_ages_params[elderly_ages_params["param"] == "IFR"])
    elderly_ages_RecPeriod = pd.DataFrame(elderly_ages_params[elderly_ages_params["param"] == "recovery_period"])
    elderly_ages_alpha = pd.DataFrame(elderly_ages_params[elderly_ages_params["param"] == "report_rate"])
    elderly_ages_sigma = pd.DataFrame(
        elderly_ages_params[elderly_ages_params["param"] == "relative_asymp_transmission"]
    )
    elderly_params = [
        elderly_ages_beta,
        elderly_ages_IFR,
        elderly_ages_RecPeriod,
        elderly_ages_alpha,
        elderly_ages_sigma,
    ]

    young_params_medians = medians_params(young_params, "0-19", last=15)  # Schools
    youngAdults_params_medians = medians_params(youngAdults_params, "20-39", last=15)  # Adults
    adults_params_medians = medians_params(adults_params, "40-49", last=15)  # Adults
    seniorAdults_params_medians = medians_params(seniorAdults_params, "50-59", last=15)  # Adults
    senior_params_medians = medians_params(senior_params, "60-69", last=15)  # Elders
    elderly_params_medians = medians_params(elderly_params, "70-90+", last=15)  # Elders

    # Simplify, get medians of values
    params_desc = ["age", "beta", "IFR", "RecPeriod", "alpha", "sigma"]

    main_adults_params_values = [
        "20-59",
        np2.median(
            [youngAdults_params_medians["beta"], adults_params_medians["beta"], seniorAdults_params_medians["beta"]]
        ),
        np2.median(
            [youngAdults_params_medians["IFR"], adults_params_medians["IFR"], seniorAdults_params_medians["IFR"]]
        ),
        np2.median(
            [
                youngAdults_params_medians["RecPeriod"],
                adults_params_medians["RecPeriod"],
                seniorAdults_params_medians["RecPeriod"],
            ]
        ),
        np2.median(
            [youngAdults_params_medians["alpha"], adults_params_medians["alpha"], seniorAdults_params_medians["alpha"]]
        ),
        np2.median(
            [youngAdults_params_medians["sigma"], adults_params_medians["sigma"], seniorAdults_params_medians["sigma"]]
        ),
    ]
    main_adults_params_medians = dict(zip(params_desc, main_adults_params_values))

    main_elders_params_values = [
        "60-90+",
        np2.median([senior_params_medians["beta"], elderly_params_medians["beta"]]),
        np2.median([senior_params_medians["IFR"], elderly_params_medians["IFR"]]),
        np2.median([senior_params_medians["RecPeriod"], elderly_params_medians["RecPeriod"]]),
        np2.median([senior_params_medians["alpha"], elderly_params_medians["alpha"]]),
        np2.median([senior_params_medians["sigma"], elderly_params_medians["sigma"]]),
    ]
    main_elders_params_medians = dict(zip(params_desc, main_elders_params_values))

    # Define parameters per layers
    def calculate_R0(IFR, alpha, beta, RecPeriod, sigma):
        return (1 - IFR) * (alpha * beta * RecPeriod + (1 - alpha) * beta * sigma * RecPeriod)

    def model_params(params_dict, layer):
        layer_params = {
            "layer": layer,
            "RecPeriod": params_dict["RecPeriod"],
            "R0": calculate_R0(
                params_dict["IFR"],
                params_dict["alpha"],
                params_dict["beta"],
                params_dict["RecPeriod"],
                params_dict["sigma"],
            ),
        }
        return layer_params

    school_params = model_params(young_params_medians, "schools")
    adults_params = model_params(main_adults_params_medians, "adults")
    elders_params = model_params(main_elders_params_medians, "elders")

    params_def = ["layer", "RecPeriod", "R0"]
    run_params = [
        [school_params["layer"], adults_params["layer"], elders_params["layer"]],
        [school_params["RecPeriod"], adults_params["RecPeriod"], elders_params["RecPeriod"]],
        [school_params["R0"], adults_params["R0"], elders_params["R0"]],
    ]
    run_params = dict(zip(params_def, run_params))

    return pd.DataFrame.from_dict(run_params)


def cache(args):
    ########################### Static params ################################################
    params = {
        "bogota": {
            "layer": ["schools", "adults", "elders"],
            "RecPeriod": [3.447429, 3.199665, 3.587770],
            "R0": [2.341840, 2.409857, 2.404539],
        }
    }

    return pd.DataFrame(params[args.city])
