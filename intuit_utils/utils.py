import pickle
import os
import pandas as pd

from pathlib import Path


def load_intuit_data(path):

    #Make it work via relative path
    #path = Path(os.path.dirname(os.path.abspath(__file__))) / path
    path = Path(path)

    simulation_path = path/"simulation_for_ucp.pickle"
    provider_id_path = path/"provider_ids.pickle"

    with open(provider_id_path, "rb") as f:
        provider_ids = pickle.load(f)

    with open(simulation_path, "rb") as f:
        simulation = pickle.load(f)
        simulation = simulation.A
        simulation = pd.DataFrame(simulation, index = provider_ids.index)

    return simulation, provider_ids