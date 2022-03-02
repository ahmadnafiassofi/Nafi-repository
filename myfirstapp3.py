import streamlit as st

import numpy as np
import pandas as pd

st.header("My first Streamlit App")


map_data = diabetes

st.map(map_data)