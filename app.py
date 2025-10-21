import streamlit as st

st.set_page_config(
    page_title="AWS Cost Optimization Dashboard",
    page_icon="💸",
    layout="wide"
)

st.title("💡 AWS Cost Optimization Assistant")
st.sidebar.success("👈 Choose a page above to get started")

st.markdown("""
Welcome to the **AWS Cost Optimization Dashboard** 🚀  

Use:
- 💬 **Chatbot** to ask questions or get LLM insights.  
- 💰 **Cost Analyzer** to inspect and manage AWS predicted costs.
""")
