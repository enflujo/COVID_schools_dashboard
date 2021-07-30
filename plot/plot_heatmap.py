import sys

sys.path.append("../")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

### Config folders

config_data = pd.read_csv("config.csv", sep=",", header=None, index_col=0)
figures_path = config_data.loc["figures_dir"][1]
results_path = config_data.loc["results_dir"][1]
results_test_path = config_data.loc["results_test_dir"][1]
ages_data_path = config_data.loc["bogota_age_data_dir"][1]
houses_data_path = config_data.loc["bogota_houses_data_dir"][1]

### Arguments

import argparse

parser = argparse.ArgumentParser(description="Dynamics visualization.")

parser.add_argument("--population", default=5000, type=int, help="Speficy the number of individials")
parser.add_argument("--type_sim", default="intervention", type=str, help="Speficy the type of simulation to plot")
args = parser.parse_args()

number_nodes = args.population
pop = number_nodes

### Read functions


def load_results(type_res, n, work_occ, comm_occ, path=results_path):
    read_path = os.path.join(
        path, "{}_schoolcap_0.35_work_occ_{}_comm_occ_{}_{}.csv".format(str(n), str(work_occ), str(comm_occ), type_res)
    )
    read_file = pd.read_csv(read_path)
    return read_file


### Plot functions


def return_pivoted_df(df_to_pivot, var="I"):
    df_heat_map = df_to_pivot.copy()
    df_heat_map = df_heat_map.pivot("work_occ", "comm_occ", var)
    df_heat_map = df_heat_map.iloc[::-1]
    return df_heat_map


def create_heatmaps_total(df_response, figs_path):
    heatmap_epid = return_pivoted_df(df_response, "E")

    ## Labels
    title_epid_hm = r"Total Infections (%), schools 35%"
    xlabel = r"Work Occupancy, %"
    ylabel = r"Community Occupancy, %"

    ## Epidemic heatmap
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    sns.heatmap(ax=ax, data=heatmap_epid, cmap="gist_heat_r", cbar=True)  # , vmin=0.0, vmax=100.0)
    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize=16)
    ax.set_title(title_epid_hm, fontsize=21)
    ax.set_xlabel(xlabel, fontsize=22)
    ax.set_ylabel(ylabel, fontsize=21)

    xticks = heatmap_epid.columns
    keptxticksidx = np.linspace(0, len(xticks), 6)
    # xtickslabels = list(xticks[ np.maximum(keptxticksidx.astype(int)-1,0) ])
    xtickslabels = ["{}".format(int(l * 100)) for l in xticks]
    # ax.set_xticks(keptxticksidx)
    ax.set_xticklabels(xtickslabels, fontsize=20, rotation=0)

    yticks = heatmap_epid.index
    keptyticksidx = np.linspace(0, len(yticks), 6)
    # ytickslabels = list(yticks[ np.maximum(keptyticksidx.astype(int)-1,0) ])
    ytickslabels = ["{}".format(int(l * 100)) for l in yticks]
    # ax.set_yticks(keptyticksidx)
    ax.set_yticklabels(ytickslabels, fontsize=20)

    plt.tight_layout()
    # plt.show()

    # Save heatmap
    if not os.path.isdir(figs_path):
        os.makedirs(figs_path)

    plt.savefig(
        os.path.join(figs_path, "totalInfections_occupations_heatmap_n_{}.png".format(5000)),
        dpi=400,
        transparent=False,
        bbox_inches="tight",
        pad_inches=0.1,
    )


def create_heatmaps_peaks(df_response, figs_path):
    heatmap_epid = return_pivoted_df(df_response, "peak_E")

    ## Labels
    title_epid_hm = r"Peak Infections, schools 35%"
    xlabel = r"Work Occupancy, %"
    ylabel = r"Community Occupancy, %"

    ## Epidemic heatmap
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    sns.heatmap(ax=ax, data=heatmap_epid, cmap="gist_heat_r", cbar=True)  # , vmin=0.0, vmax=1.0)
    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize=16)
    ax.set_title(title_epid_hm, fontsize=21)
    ax.set_xlabel(xlabel, fontsize=22)
    ax.set_ylabel(ylabel, fontsize=21)

    xticks = heatmap_epid.columns
    keptxticksidx = np.linspace(0, len(xticks), 6)
    # xtickslabels = list(xticks[ np.maximum(keptxticksidx.astype(int)-1,0) ])
    xtickslabels = ["{}".format(int(l * 100)) for l in xticks]
    # ax.set_xticks(keptxticksidx)
    ax.set_xticklabels(xtickslabels, fontsize=20, rotation=0)

    yticks = heatmap_epid.index
    keptyticksidx = np.linspace(0, len(yticks), 6)
    # ytickslabels = list(yticks[ np.maximum(keptyticksidx.astype(int)-1,0) ])
    ytickslabels = ["{}".format(int(l * 100)) for l in yticks]
    # ax.set_yticks(keptyticksidx)
    ax.set_yticklabels(ytickslabels, fontsize=20)

    plt.tight_layout()
    # plt.show()

    # Save heatmap
    if not os.path.isdir(figs_path):
        os.makedirs(figs_path)

    plt.savefig(
        os.path.join(figs_path, "peakInfections_occupations_heatmap_n_{}.png".format(5000)),
        dpi=400,
        transparent=False,
        bbox_inches="tight",
        pad_inches=0.1,
    )


def create_heatmaps_deaths(df_response, figs_path):
    heatmap_epid = return_pivoted_df(df_response, "D")

    ## Labels
    title_epid_hm = r"Total Deaths (%), schools 35%"
    xlabel = r"Work Occupancy, %"
    ylabel = r"Community Occupancy, %"

    ## Epidemic heatmap
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    sns.heatmap(ax=ax, data=heatmap_epid, cmap="bone_r", cbar=True)  # , vmin=0.0, vmax=1.0)
    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize=16)
    ax.set_title(title_epid_hm, fontsize=21)
    ax.set_xlabel(xlabel, fontsize=22)
    ax.set_ylabel(ylabel, fontsize=21)

    xticks = heatmap_epid.columns
    keptxticksidx = np.linspace(0, len(xticks), 6)
    # xtickslabels = list(xticks[ np.maximum(keptxticksidx.astype(int)-1,0) ])
    xtickslabels = ["{}".format(int(l * 100)) for l in xticks]
    # ax.set_xticks(keptxticksidx)
    ax.set_xticklabels(xtickslabels, fontsize=20, rotation=0)

    yticks = heatmap_epid.index
    keptyticksidx = np.linspace(0, len(yticks), 6)
    # ytickslabels = list(yticks[ np.maximum(keptyticksidx.astype(int)-1,0) ])
    ytickslabels = ["{}".format(int(l * 100)) for l in yticks]
    # ax.set_yticks(keptyticksidx)
    ax.set_yticklabels(ytickslabels, fontsize=20)

    plt.tight_layout()
    # plt.show()

    # Save heatmap
    if not os.path.isdir(figs_path):
        os.makedirs(figs_path)

    plt.savefig(
        os.path.join(figs_path, "totalDeaths_occupations_heatmap_n_{}.png".format(5000)),
        dpi=400,
        transparent=False,
        bbox_inches="tight",
        pad_inches=0.1,
    )


### Read file
results_path = os.path.join(results_path, "occupations", str(pop))
work_occup = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
comm_occup = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

df_res = []
for i in work_occup:
    for j in comm_occup:
        res_read = load_results("soln_cum", pop, i, j, results_path)
        res_last = res_read.groupby("tvec").median().iloc[-1]
        res = pd.DataFrame()
        res["work_occ"] = [i]
        res["comm_occ"] = [j]
        res["E"] = [res_last["E"] * 100]
        df_res.append(res)
df_res = pd.concat(df_res)

df_res_peaks = []
for i in work_occup:
    for j in comm_occup:
        res_read = load_results("soln", pop, i, j, results_path)
        # res_smooth = model.smooth_timecourse(res_read)
        res_med = res_read.groupby("tvec").median()
        res_med = res_med.reset_index()
        peak_E = max(res_med["E"]) * 100
        res = pd.DataFrame()
        res["work_occ"] = [i]
        res["comm_occ"] = [j]
        res["peak_E"] = [peak_E]
        df_res_peaks.append(res)
df_res_peaks = pd.concat(df_res_peaks)

df_res_D = []
for i in work_occup:
    for j in comm_occup:
        res_read = load_results("soln_cum", pop, i, j, results_path)
        res_last = res_read.groupby("tvec").median().iloc[-1]
        res = pd.DataFrame()
        res["work_occ"] = [i]
        res["comm_occ"] = [j]
        res["D"] = [res_last["D"] * 100]
        df_res_D.append(res)
df_res_D = pd.concat(df_res_D)


### Save
save_path = os.path.join(figures_path, "heatmaps")
create_heatmaps_total(df_res, save_path)
create_heatmaps_deaths(df_res_D, save_path)
create_heatmaps_peaks(df_res_peaks, save_path)
