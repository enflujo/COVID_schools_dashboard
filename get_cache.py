import json
import numpy as np
from modules.global_arguments import parse_args
from modules.graph_age_distribution import build as age_distribution_data
from modules.graph_age_params import build as age_aparams_data
from modules.graph_household_sizes import build as household_sizes_distribution_data
from modules.graph_teachers_distribution import build as teachers_distribution_data


args = parse_args()
age_params = age_aparams_data(args)
age_distribution = age_distribution_data(args)
teachers_distribution = teachers_distribution_data(args)

with open("{}/age_distribution_{}.json".format(args.results_path, str(args.city)), "w") as outfile:
    json.dump(age_distribution, outfile)

with open("{}/age_params_{}.json".format(args.results_path, str(args.city)), "w") as outfile:
    json.dump(age_params, outfile)


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


household_sizes_distribution = json.dumps(household_sizes_distribution_data(args), cls=NumpyEncoder)

with open("{}/household_sizes_distribution_{}.json".format(args.results_path, str(args.city)), "w") as outfile:
    json.dump(household_sizes_distribution, outfile)

with open("{}/teachers_distribution_{}.json".format(args.results_path, str(args.city)), "w") as outfile:
    json.dump(teachers_distribution, outfile)
