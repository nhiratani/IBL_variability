from pathlib import Path
import pickle
import pandas as pd

PKL = Path(
    "prior_localization_single_session_output/"
    "dfbe628d-365b-461c-a07f-8b9911ba83aa/"
    "whole_session_summary_SINGLE_SESSION_ACAd.pkl"
)

with open(PKL, "rb") as f:
    rows = pickle.load(f)

df = pd.DataFrame(rows)
print("Loaded:", PKL)
print(df[["eid","roi","group","n_trials_group_used","r2_real","r2_fake_mean","r2_corr","status"]].to_string(index=False))