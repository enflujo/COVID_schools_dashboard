import json
import numpy as np
import jax.numpy as jxnp


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, obj):
        if isinstance(obj, np.integer) or isinstance(obj, jxnp.integer):
            return int(obj)
        elif isinstance(obj, np.floating) or isinstance(obj, jxnp.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray) or isinstance(obj, jxnp.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
