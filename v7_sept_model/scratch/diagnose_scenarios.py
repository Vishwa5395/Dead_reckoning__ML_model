import pickle
import numpy as np

with open("v7_sept_model/data/test_scenarios_v7.pkl", "rb") as f:
    scens = pickle.load(f)

for name, journeys in scens.items():
    total_outages = 0
    total_len = 0
    for j in journeys:
        total_len += len(j["x_gps"])
        total_outages += len(range(11, len(j["x_gps"]) - 10, 10))
    print(f"Scenario: {name:15s} | Journeys: {len(journeys)} | Outages: {total_outages:3d} | Total timesteps: {total_len:6d}")
