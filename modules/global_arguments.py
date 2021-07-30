import argparse
import os
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Simulating interventions")

    parser.add_argument("--res_id", default="ND", type=str, help="Result ID for simulation save")

    parser.add_argument("--population", default=500, type=int, help="Speficy the number of individials")
    parser.add_argument("--intervention", default=0.6, type=float, help="Intervention efficiancy")
    parser.add_argument(
        "--intervention_type",
        default="intervention",
        type=str,
        help="Define the type of intervention [no_intervention,internvention,school_alternancy]",
    )
    parser.add_argument(
        "--work_occupation", default=0.6, type=float, help="Percentage of occupation at workplaces over intervention"
    )
    parser.add_argument(
        "--school_occupation", default=0.35, type=float, help="Percentage of occupation at classrooms over intervention"
    )
    parser.add_argument("--school_openings", default=20, type=int, help="Day of the simulation where schools are open")

    parser.add_argument(
        "--ventilation_out",
        default=3,
        type=float,
        help="Ventilation values (h-1) that define how much ventilated is a classroom [2-15]",
    )
    parser.add_argument(
        "--fraction_people_masks", default=1.0, type=float, help="Fraction value of people wearing masks"
    )
    parser.add_argument(
        "--masks_type",
        default="N95",
        type=str,
        help="Type of masks that individuals are using. Options are: cloth, surgical, N95",
    )
    parser.add_argument(
        "--duration_event", default=6, type=float, help="Duration of event (i.e. classes/lectures) in hours over a day"
    )

    parser.add_argument("--height_room", default=3.1, type=float, help="Schools height of classroom")
    parser.add_argument("--preschool_length_room", default=7.0, type=float, help="Preschool length of classroom")
    parser.add_argument("--preschool_width_room", default=7.0, type=float, help="Preschool length of classroom")
    parser.add_argument("--primary_length_room", default=10.0, type=float, help="primary length of classroom")
    parser.add_argument("--primary_width_room", default=10.0, type=float, help="primary length of classroom")
    parser.add_argument("--highschool_length_room", default=10.0, type=float, help="highschool length of classroom")
    parser.add_argument("--highschool_width_room", default=10.0, type=float, help="highschool length of classroom")

    parser.add_argument("--Tmax", default=200, type=int, help="Length of simulation (days)")
    parser.add_argument("--delta_t", default=0.08, type=float, help="Time steps")
    parser.add_argument("--number_trials", default=10, type=int, help="Number of iterations per step")

    parser.add_argument("--preschool_mean", default=9.4, type=float, help="preschool degree distribution (mean)")
    parser.add_argument(
        "--preschool_std", default=1.8, type=float, help="preschool degree distribution (standard deviation)"
    )
    parser.add_argument("--preschool_size", default=15, type=float, help="Number of students per classroom")
    parser.add_argument("--preschool_r", default=1, type=float, help="Correlation in preschool layer")

    parser.add_argument("--primary_mean", default=9.4, type=float, help="primary degree distribution (mean)")
    parser.add_argument(
        "--primary_std", default=1.8, type=float, help="primary degree distribution (standard deviation)"
    )
    parser.add_argument("--primary_size", default=35, type=float, help="Number of students per classroom")
    parser.add_argument("--primary_r", default=1, type=float, help="Correlation in primary layer")

    parser.add_argument("--highschool_mean", default=9.4, type=float, help="highschool degree distribution (mean)")
    parser.add_argument(
        "--highschool_std", default=1.8, type=float, help="highschool degree distribution (standard deviation)"
    )
    parser.add_argument("--highschool_size", default=35, type=float, help="Number of students per classroom")
    parser.add_argument("--highschool_r", default=1, type=float, help="Correlation in highschool layer")

    parser.add_argument("--work_mean", default=14.4 / 3, type=float, help="Work degree distribution (mean)")
    parser.add_argument("--work_std", default=6.2 / 3, type=float, help="Work degree distribution (standard deviation)")
    parser.add_argument("--work_size", default=10, type=float, help="Approximation of a work place size")
    parser.add_argument("--work_r", default=1, type=float, help="Correlation in work layer")

    parser.add_argument("--community_mean", default=4.3 / 2, type=float, help="Community degree distribution (mean)")
    parser.add_argument(
        "--community_std", default=1.9 / 2, type=float, help="Community degree distribution (standard deviation)"
    )
    parser.add_argument("--community_n", default=1, type=float, help="Number of community")
    parser.add_argument("--community_r", default=0, type=float, help="Correlation in community layer")

    ################### new args #########################
    parser.add_argument("--city", default="bogota", type=str, help="City")
    # Paths
    base = os.getcwd()
    config_data = pd.read_csv("config.csv", sep=",", header=None, index_col=0)

    parser.add_argument(
        "--figures_path", default=base + config_data.loc["figures_dir"][1], help="Directory to save figures"
    )
    parser.add_argument(
        "--results_path", default=base + config_data.loc["results_dir"][1], help="Directory to save results .csv files"
    )
    parser.add_argument(
        "--params_data_path",
        default=base + config_data.loc["bogota_params_ages_data"][1],
        help="Path to ages parameters data",
    )
    parser.add_argument(
        "--ages_data_path",
        default=base + config_data.loc["bogota_age_data_dir"][1],
        help="Path to ages distgribution data",
    )
    parser.add_argument(
        "--houses_data_path",
        default=base + config_data.loc["bogota_houses_data_dir"][1],
        help="Path to city census data",
    )
    parser.add_argument(
        "--teachers_data_path",
        default=base + config_data.loc["bogota_teachers_data_dir"][1],
        help="Path to teachers level distribution data",
    )

    return parser.parse_args()
