import json
from modules.graph_age_distribution import build as age_distribution_data
from modules.graph_age_params import build as age_aparams_data
from modules.graph_household_sizes import build as household_sizes_distribution_data
from modules.graph_teachers_distribution import build as teachers_distribution_data
from utils.NumpyEncoder import NumpyEncoder


def run(args):
    cache_data_path = "cached_data"
    age_params = age_aparams_data(args)
    age_distribution = age_distribution_data(args)
    teachers_distribution = teachers_distribution_data(args)

    with open("{}/age_distribution_{}.json".format(cache_data_path, str(args.city)), "w") as outfile:
        json.dump(age_distribution, outfile)

    with open("{}/age_params_{}.json".format(cache_data_path, str(args.city)), "w") as outfile:
        json.dump(age_params, outfile)

    household_sizes_distribution = json.dumps(household_sizes_distribution_data(args), cls=NumpyEncoder)

    with open("{}/household_sizes_distribution_{}.json".format(cache_data_path, str(args.city)), "w") as outfile:
        json.dump(household_sizes_distribution, outfile)

    with open("{}/teachers_distribution_{}.json".format(cache_data_path, str(args.city)), "w") as outfile:
        json.dump(teachers_distribution, outfile)
