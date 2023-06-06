import streamlit as st
import requests

st.title('Server Connection Test')

# Simple GET request
response = requests.get('https://mobilex.kr/ai/dev/team8/process')

if response.status_code == 200:
    st.success('Connection with the server is successful.')
else:
    st.error('Could not connect to the server.')
