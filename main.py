import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
from numpy import log, sqrt, exp 
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Option Pricing using Black-Scholes",
    page_icon=" ",
    layout="wide",
    initial_sidebar_state="expanded")


# Custom CSS to inject into Streamlit
st.markdown("""
<style>
/* Adjust the size and alignment of the CALL and PUT value containers */
.metric-container {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 6px; /* Adjust the padding to control height */
    width: auto; /* Auto width for responsiveness, or set a fixed width if necessary */
    margin: 0 auto; /* Center the container */
}

/* Custom classes for CALL and PUT values */
.metric-call {
    background-color: #0aab35; 
    color: white; 
    margin-right: 10px; /* Spacing between the call and put */
    border-radius: 5px; /* corners */
}

.metric-put {
    background-color: #ff4f2b; 
    color: white; 
    border-radius: 5px; /* corners */
}

/* Style for the value text */
.metric-value {
    font-size: 1.5rem;
    font-weight: bold;
    margin: 0; /* Remove default margins */
}

/* Style for the label text */
.metric-label {
    font-size: 1rem; /* Adjust font size */
    margin-bottom: 4px; /* Spacing between label and value */
}

</style>
""", unsafe_allow_html=True)

# (Include the BlackScholes class definition here)

class BlackScholes:
    def __init__(
        self,
        time_to_maturity: float,
        strike: float,
        current_price: float,
        volatility: float,
        interest_rate: float,
    ):
        self.time_to_maturity = time_to_maturity
        self.strike = strike
        self.current_price = current_price
        self.volatility = volatility
        self.interest_rate = interest_rate

# for the heatmap
    def calculate_prices(
        self,
    ):
        time_to_maturity = self.time_to_maturity
        strike = self.strike
        current_price = self.current_price
        volatility = self.volatility
        interest_rate = self.interest_rate

        d1 = (
            log(current_price / strike) +
            (interest_rate + 0.5 * volatility ** 2) * time_to_maturity
            ) / (
                volatility * sqrt(time_to_maturity)
            )
        d2 = d1 - volatility * sqrt(time_to_maturity)

        call_price = current_price * norm.cdf(d1) - (
            strike * exp(-(interest_rate * time_to_maturity)) * norm.cdf(d2)
        )
        put_price = (
            strike * exp(-(interest_rate * time_to_maturity)) * norm.cdf(-d2)
        ) - current_price * norm.cdf(-d1)

        self.call_price = call_price
        self.put_price = put_price

        # GREEKS
        # Delta
        self.call_delta = norm.cdf(d1)
        self.put_delta = 1 - norm.cdf(d1)

        # Gamma
        self.call_gamma = norm.pdf(d1) / (
            strike * volatility * sqrt(time_to_maturity)
        )
        self.put_gamma = self.call_gamma

        return call_price, put_price


# Sidebar for User Inputs
with st.sidebar:
    st.title("Options Parameters")

    strike = st.number_input("Strike Price", value=100.0, step=0.5)
    current_price = st.number_input("Current Share Price", value=100.0, step=0.5)
    time_to_maturity = st.number_input("Time to Maturity in Weeks", value=52.0, step=1.0) * 1.0 / 52.0
    interest_rate = st.number_input("Risk-Free Interest Rate", value=0.039, step=0.005, format="%.3f")
    volatility = st.number_input("Volatility (σ)", value=0.1, step=0.01)

    st.markdown("---")
    st.title("Heatmap bounds")
    share_min = st.number_input('Min Share Price', min_value=0.01, value=current_price*0.8, step=0.01)
    share_max = st.number_input('Max Share Price', min_value=0.01, value=current_price*1.2, step=0.01)
    vol_min = st.slider('Min Volatility for Heatmap', min_value=0.01, max_value=1.0, value=volatility*0.5, step=0.01)
    vol_max = st.slider('Max Volatility for Heatmap', min_value=0.01, max_value=1.0, value=volatility*1.5, step=0.01)
    
    share_range = np.linspace(share_min, share_max, 10)
    vol_range = np.linspace(vol_min, vol_max, 10)

def plot_heatmap(bs_model, share_range, vol_range, strike):
    call_prices = np.zeros((len(vol_range), len(share_range)))
    put_prices = np.zeros((len(vol_range), len(share_range)))
    
    #looping through for the call and put heatmaps
    for i, vol in enumerate(vol_range):
        for j, share in enumerate(share_range):
            bs_temp = BlackScholes(
                time_to_maturity=bs_model.time_to_maturity,
                strike=strike,
                current_price=share,
                volatility=vol,
                interest_rate=bs_model.interest_rate
            )
            bs_temp.calculate_prices()
            call_prices[i, j] = bs_temp.call_price
            put_prices[i, j] = bs_temp.put_price
    
    # Plots
    fig_call, ax_call = plt.subplots(figsize=(10, 8))
    sns.heatmap(call_prices, xticklabels=np.round(share_range, 2), yticklabels=np.round(vol_range, 2), annot=True, fmt=".2f", cmap="crest", ax=ax_call)
    ax_call.set_title('Calls')
    ax_call.set_xlabel('Share Price')
    ax_call.set_ylabel('Volatility')
    
    # Plotting Put Price Heatmap
    fig_put, ax_put = plt.subplots(figsize=(10, 8))
    sns.heatmap(put_prices, xticklabels=np.round(share_range, 2), yticklabels=np.round(vol_range, 2), annot=True, fmt=".2f", cmap="flare", ax=ax_put)
    ax_put.set_title('Puts')
    ax_put.set_xlabel('Share Price')
    ax_put.set_ylabel('Volatility')
    
    return fig_call, fig_put


# Main Page for Output Display
st.title("Black-Scholes Pricing Model")
st.write("""See how option contracts vary in prices based off of the parameters of the Black-Scholes equation.
         The Black-Sholes equation is a partial differential equation that, when solved,
         gives an explicit price for an option contract in terms of the 
         strike price, current asset price, the time to maturity,
         the risk free interest rate, and the volitity of the asset (sigma) These values
         can be adjusted on the menu on the left side.""")


# Calculate Call and Put values
bs_model = BlackScholes(time_to_maturity, strike, current_price, volatility, interest_rate)
call_price, put_price = bs_model.calculate_prices()

# Display Call and Put Values in colored tables
col1, col2 = st.columns([1,1], gap="small")

with col1:
    # Using the custom class for CALL value
    st.markdown(f"""
        <div class="metric-container metric-call">
            <div>
                <div class="metric-value">Call - ${call_price:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    # Using the custom class for PUT value
    st.markdown(f"""
        <div class="metric-container metric-put">
            <div>
                <div class="metric-value">Put - ${put_price:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("")

# Interactive Sliders and Heatmaps for Call and Put Options
col1, col2 = st.columns([1,1], gap="small")

with col1:
    st.subheader("Call Price Heatmap")
    heatmap_fig_call, _ = plot_heatmap(bs_model, share_range, vol_range, strike)
    st.pyplot(heatmap_fig_call)

with col2:
    st.subheader("Put Price Heatmap")
    _, heatmap_fig_put = plot_heatmap(bs_model, share_range, vol_range, strike)
    st.pyplot(heatmap_fig_put)
