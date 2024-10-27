import streamlit as st
import getpass
import os
import pandas as pd
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

# Check if the Mistral API key is set in the environment variables
if "MISTRAL_API_KEY" not in os.environ:
    os.environ["MISTRAL_API_KEY"] = 'LdpjtEOZsI0U8zaFnwkRZW0zz8ArZMl0'

# Initialize the LLM
llm = ChatMistralAI(
    model="mistral-large-latest",
    temperature=0,
    max_retries=2,
)

# Load the dataset
@st.cache_resource
def load_data():
    return pd.read_csv("data_retailers.csv")

data = load_data()

# Streamlit application layout
st.title("Retailer Fraud Detection")
st.write("Enter the name of the retailer to check their legitimacy.")

# Input field for retailer name
retailer_name = st.text_input("Retailer Name", placeholder="Enter retailer name")

# Function to get retailer details using LLM
def get_retailer_details(retailer_name):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "human",
            """You are provided with a dataset containing details of retailers selling their products on e-commerce platforms.
            Your job is to provide the details of the retailer about whom it is asked and determine whether the given retailer is legitimate or verified or not.
            If the data does not have any information, simply answer as 'No information available'. Do not give further explanation or description.
            Data:{data}
            Retailer:{retailer}
            Present the answer in bullet point format."""
            )
        ]
    )
    chain = prompt | llm
    run = chain.invoke(
        {
            "data": data,
            "retailer": retailer_name
        }
    )
    return run.content

# Button to trigger LLM analysis
if st.button("Search"):
    if retailer_name:
        # Call the function to get retailer details
        with st.spinner('Analyzing retailer details...'):
            try:
                result = get_retailer_details(retailer_name)

                # Display the retailer details in two columns
                st.subheader("Retailer Details")
                details = result.split("\n")
                col1, col2 = st.columns(2)

                # Show alternating details in two columns
                for idx, detail in enumerate(details):
                    if idx % 2 == 0:
                        with col1:
                            st.markdown(f'<div class="detail-box">{detail}</div>', unsafe_allow_html=True)
                    else:
                        with col2:
                            st.markdown(f'<div class="detail-box">{detail}</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter a retailer name to search.")

# Add custom CSS for styling
st.markdown("""
    <style>
        body {
            background-image: url('https://images.unsplash.com/photo-1581091223571-9bdb6c819168?auto=format&fit=crop&w=2100&q=80');
            background-size: cover;
            background-position: center;
        }
        .detail-box {
            padding: 10px;
            margin: 5px;
            background-color: rgba(0, 0, 0, 0.6);
            color: #fff;
            border-radius: 8px;
            box-shadow: 2px 2px 12px rgba(0, 0, 0, 0.3);
            transition: transform 0.3s, box-shadow 0.3s;
        }
        .detail-box:hover {
            transform: scale(1.02);
            box-shadow: 2px 2px 12px rgba(255, 255, 255, 0.6);
        }
        h1, h2, h3, h4 {
            color: white;
        }
        .stTextInput input {
            background-color: rgba(0, 0, 0, 0.5);
            color: white;
        }
    </style>
""", unsafe_allow_html=True)