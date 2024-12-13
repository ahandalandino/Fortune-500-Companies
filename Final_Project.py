import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pydeck as pdk
import numpy as np

# Set custom colors palette for website
st.set_page_config(page_title="Fortune 500 Corporate Headquarters Explorer", page_icon="🏢", layout="wide")

# Apply custom CSS to the Streamlit app with the color scale and layout
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@400;700&family=Roboto:wght@400;700&display=swap');
        body {
            background-color: #F1F3F6;
            color: #333333;
            font-family: 'Roboto', sans-serif;
        }
        .sidebar .sidebar-content {
            background-color: #F1B3A1;
        }
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Quicksand', sans-serif;
            color: #B4C9DD;
        }
        .stSubheader[data-baseweb="true"]:nth-child(1) {
            background-color: #EDE1DB;
            padding: 10px;
            color: #333333;
        }
        .stButton>button:hover {
            background-color: #CCDBE9;
            color: #333333;
        }
        .stBarChart {
            color: #F1B3A1;
        }
        .stHeader {
            background-color: #DDE4ED;
        }
        .stCard {
            background-color: #CCDBE9;
            border-radius: 10px;
        }
    </style>
    """, unsafe_allow_html=True
)

# Documentation String
"""
Name:       Ana Handal Andino
CS230:      Section 2
Data:       Fortune 500 Corporate Headquarters
URL:        http://localhost:8501/

Description:
This program visualizes the Fortune 500 corporate headquarters data, exploring the distribution of revenue, employees, and geographic locations of the top companies.
"""

# Load the data with error handling and return multiple values
@st.cache_data
def load_data(file):
    """
    Load data from an uploaded file.

    Args:
        file (file-like object): The file to load.

    Returns:
        tuple: (data, error_message) where data is the loaded DataFrame and error_message is a string if an error occurred.
    """
    try:
        data = pd.read_excel(file)
        return data, None
    except Exception as e:
        return None, str(e)


# Extra function with a default argument
def filter_data(df, min_revenue=1000, max_revenue=50000, state=None):
    """
    Filter the dataset by revenue range and state.

    Args:
        df (pd.DataFrame): The DataFrame to filter.
        min_revenue (int): Minimum revenue (default is 1000).
        max_revenue (int): Maximum revenue (default is 50000).
        state (str): Filter for state (default is None).

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    filtered_data = df[(df['REVENUES'] >= min_revenue) & (df['REVENUES'] <= max_revenue)]

    if state:
        filtered_data = filtered_data[filtered_data['STATE'] == state]

    return filtered_data


# Main function
def main():
    st.title("Fortune 500 Corporate Headquarters Explorer")

    # Sidebar Section 1: Filters
    with st.sidebar:
        st.header("Filters & Controls")
        uploaded_file = st.file_uploader("Upload Fortune 500 Data", type=["xlsx"],
                                         help="Upload the Excel file with Fortune 500 data.")

        if uploaded_file is not None:
            df, error = load_data(uploaded_file)
        else:
            st.error("Please upload a file to continue.")
            return

        if error:
            st.error(f"Error loading data: {error}")
            return

        # Dynamic filters
        states = df['STATE'].unique() if 'STATE' in df.columns else []

        revenue_filter = st.slider("Revenue Range (in millions)", int(df['REVENUES'].min()),
                                   int(df['REVENUES'].max()), (1000, 50000),
                                   help="Select the revenue range to filter the data.")
        employees_filter = st.slider("Employees Range", int(df['EMPLOYEES'].min()), int(df['EMPLOYEES'].max()),
                                     (1000, 50000), help="Select the range of employees to filter by.")
        state_filter = st.selectbox("Select State", options=['All'] + list(states),
                                     help="Choose the state for filtering.")

        # Filter data based on user inputs using the extra function filter_data
        filtered_data = filter_data(df, revenue_filter[0], revenue_filter[1],
                                    state_filter if state_filter != 'All' else None)

    # Visualize Revenue Distribution
    st.subheader("Revenue Distribution")
    plot_revenue_distribution(filtered_data)

    # Visualize Top Companies by Revenue as a Bar Chart
    st.subheader("Top Companies by Revenue")
    plot_top_companies_by_revenue(filtered_data)

    # Display Data Insights
    st.subheader("Key Insights")
    display_data_insights(filtered_data)

    # Interactive Data Table
    st.subheader("Filtered Data")
    st.dataframe(filtered_data)  # Enable sorting and filtering in the data table

    # Map Showing Locations of Corporate Headquarters
    st.subheader("Corporate Headquarters Location Map")
    show_location_map(filtered_data)

    # Add a download button with icon
    st.subheader("Download Filtered Data")
    add_download_button(filtered_data)

    # Create and display pivot table by state
    st.subheader("Revenue Summary by State")
    pivot_table_by_state(filtered_data)

    # Pie Chart for Revenue Distribution by State
    st.subheader("Revenue Distribution by State (Top 10)")
    plot_revenue_pie_chart(filtered_data)


# Visualization of revenue distribution
def plot_revenue_distribution(filtered_data):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.hist(filtered_data['REVENUES'], bins=30, color='#F1B3A1', edgecolor='black', alpha=0.7)
    ax.set_title('Revenue Distribution of Fortune 500 Companies (in millions)', fontsize=18, color='#333333')
    ax.set_xlabel('Revenue ($ millions)', fontsize=14, color='#333333')
    ax.set_ylabel('Number of Companies', fontsize=14, color='#333333')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_facecolor('#EFEFE9')


    ax.set_xticklabels([f"${x:,.0f}" for x in ax.get_xticks()])

    st.pyplot(fig)


# Bar chart for Top Companies by Revenue
def plot_top_companies_by_revenue(filtered_data):
    top_companies = filtered_data[['NAME', 'REVENUES']].sort_values(by='REVENUES', ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(top_companies['NAME'], top_companies['REVENUES'], color='#F1B3A1', edgecolor='black')
    ax.set_title('Top 10 Fortune 500 Companies by Revenue', fontsize=18, color='#333333')
    ax.set_xlabel('Revenue ($ millions)', fontsize=14, color='#333333')
    ax.set_ylabel('Company', fontsize=14, color='#333333')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_facecolor('#EFEFE9')


    ax.set_xticklabels([f"${x:,.0f}" for x in ax.get_xticks()])

    st.pyplot(fig)


# Display Data Insights
def display_data_insights(filtered_data):
    top_5_companies = filtered_data[['NAME', 'REVENUES']].sort_values(by='REVENUES', ascending=False).head(5)


    top_5_companies['REVENUES'] = top_5_companies['REVENUES'].apply(lambda x: f"${x:,.0f}")

    total_employees = filtered_data['EMPLOYEES'].sum()
    average_revenue = filtered_data['REVENUES'].mean()

    st.write("Top 5 Companies by Revenue:")
    st.write(top_5_companies)
    st.write(f"Total Employees across selected companies: {total_employees:,}")
    st.write(f"Average Revenue across selected companies: ${average_revenue:,.2f} million")


# Show a map with the locations of corporate headquarters
def show_location_map(filtered_data):
    filtered_data = filtered_data.dropna(subset=['LATITUDE', 'LONGITUDE'])

    filtered_data = filtered_data[(filtered_data['LATITUDE'] >= -90) & (filtered_data['LATITUDE'] <= 90)]
    filtered_data = filtered_data[(filtered_data['LONGITUDE'] >= -180) & (filtered_data['LONGITUDE'] <= 180)]

    if filtered_data.empty:
        st.write("No valid location data available to display on the map.")
        return

    layer = pdk.Layer(
        "ScatterplotLayer",
        filtered_data,
        pickable=True,
        get_position=["LONGITUDE", "LATITUDE"],
        get_radius=50000,
        get_fill_color=[255, 182, 193],  # Light Pink color
        get_line_color=[0, 0, 0],
        radius_scale=2,
        tooltip={"text": "{NAME} - ${REVENUES} million"}
    )

    view_state = pdk.ViewState(
        latitude=filtered_data['LATITUDE'].mean(),
        longitude=filtered_data['LONGITUDE'].mean(),
        zoom=3,
        pitch=0,
    )

    deck = pdk.Deck(layers=[layer], initial_view_state=view_state)
    st.pydeck_chart(deck)


# Add download button for filtered data
def add_download_button(filtered_data):
    csv = filtered_data.to_csv(index=False)
    st.download_button(label="Download Filtered Data", data=csv, file_name="filtered_data.csv", mime="text/csv")


# Create a pivot table to show revenue summary by state
def pivot_table_by_state(filtered_data):
    # Let's assume  a STATE column to group by
    if 'STATE' in filtered_data.columns:
        pivot = filtered_data.pivot_table(values='REVENUES', index='STATE', aggfunc='sum')


        pivot['REVENUES'] = pivot['REVENUES'].apply(lambda x: f"${x:,.0f}")

        st.write(pivot)
    else:
        st.write("No STATE column found for pivot table.")


# Pie Chart for Revenue Distribution by State
def plot_revenue_pie_chart(filtered_data):
    if 'STATE' in filtered_data.columns:
        # Group by STATE and calculate total REVENUES for each state
        state_revenue = filtered_data.groupby('STATE')['REVENUES'].sum().reset_index()

        # Sort the states by revenue for better visual distribution
        state_revenue = state_revenue.sort_values(by='REVENUES', ascending=False)

        # Filter to show only the top 10 states
        top_states = state_revenue.head(10)

        # Create the pie chart
        fig, ax = plt.subplots(figsize=(12, 8))

        # Exploding the top 5
        explode = [0.1 if i < 5 else 0 for i in range(len(top_states))]

        wedges, texts, autotexts = ax.pie(top_states['REVENUES'],
                                          labels=top_states['STATE'],
                                          autopct='%1.1f%%',
                                          startangle=90,
                                          explode=explode,  # Exploding top 5 states
                                          colors=plt.cm.Paired.colors,
                                          textprops={'fontsize': 8},
                                          wedgeprops={'linewidth': 0.5, 'edgecolor': 'black'})

        # Add annotations for hover-like effects
        for i, wedge in enumerate(wedges):
            angle = (wedge.theta2 + wedge.theta1) / 2
            x = wedge.r * 0.8 * np.cos(np.deg2rad(angle))
            y = wedge.r * 0.8 * np.sin(np.deg2rad(angle))
            ax.annotate(f"${top_states['REVENUES'].iloc[i]:,.0f}M", xy=(x, y), ha='center', fontsize=8, color='black')

        ax.set_title('Revenue Distribution by State (Top 10)', fontsize=14)

        st.pyplot(fig)
    else:
        st.write("No STATE column found for pie chart.")


# Run the app
if __name__ == "__main__":
    main()
