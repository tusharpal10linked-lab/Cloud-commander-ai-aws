import os
import io
import json
import tempfile
import streamlit as st
import pandas as pd
import boto3
from dotenv import load_dotenv
from botocore.exceptions import ClientError

# ============ LOAD CONFIG ============
load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "eu-north-1")  # for S3, Lambda
BEDROCK_REGION = os.getenv("BEDROCK_REGION", "us-east-1")
BEDROCK_URL = os.getenv("BEDROCK_URL")  # e.g. "https://bedrock-runtime.us-east-1.amazonaws.com"
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")
S3_BUCKET = os.getenv("S3_BUCKET", "aws-ai-hackathon-09")
COST_CSV_KEY = os.getenv("COST_CSV_KEY", "synthetic_aws_billing_data.csv")
PREDICTIONS_CSV_KEY = os.getenv("PREDICTIONS_CSV_KEY", "flagged_high_costs.csv")
LAMBDA_FUNCTION_NAME = os.getenv("LAMBDA_FUNCTION_NAME", "test-aws-ai-hackathon")

# ============ INITIALIZE CLIENTS ============
s3 = boto3.client("s3", region_name=AWS_REGION)
lambda_client = boto3.client("lambda", region_name=AWS_REGION)

bedrock_kwargs = {
    "region_name": BEDROCK_REGION
}
if BEDROCK_URL:
    bedrock_kwargs["endpoint_url"] = BEDROCK_URL
bedrock_client = boto3.client("bedrock-runtime", **bedrock_kwargs)

# ============ UTILITIES ============
def load_csv_from_s3(bucket: str, key: str) -> pd.DataFrame:
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))

def save_df_to_s3(df: pd.DataFrame, bucket: str, key: str):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        df.to_csv(tmp.name, index=False)
        tmp_path = tmp.name
    s3.upload_file(tmp_path, bucket, key)

def trigger_lambda_for_service(account: str, service: str) -> dict:
    payload = {"account": account, "service": service}
    try:
        resp = lambda_client.invoke(
            FunctionName=LAMBDA_FUNCTION_NAME,
            InvocationType="RequestResponse",
            Payload=json.dumps(payload),
        )
        body = resp.get("Payload")
        result = body.read().decode("utf-8") if body else "{}"
        return {"status": resp.get("StatusCode"), "response": result}
    except ClientError as e:
        return {"error": str(e)}

def df_summary_tables(df: pd.DataFrame, top_n: int = 10):
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), "No flagged resources."
    preview_df = df.head(top_n).copy()
    acct_col = next((c for c in df.columns if c.lower() in ["account", "accountid", "account_id"]), None)
    service_col = next((c for c in df.columns if c.lower() in ["service", "servicename", "service_name"]), None)
    cost_col = next((c for c in df.columns if "predict" in c.lower() or "cost" in c.lower()), None)

    acct_agg_df = pd.DataFrame()
    serv_agg_df = pd.DataFrame()
    if acct_col and cost_col:
        acct_agg_df = (
            df.groupby(acct_col)[cost_col]
            .sum()
            .reset_index()
            .sort_values(cost_col, ascending=False)
        )
    if service_col and cost_col:
        serv_agg_df = (
            df.groupby(service_col)[cost_col]
            .sum()
            .reset_index()
            .sort_values(cost_col, ascending=False)
        )
    text_summary = "Preview + aggregates available as tables."
    return preview_df, acct_agg_df, serv_agg_df, text_summary

# ============ BEDROCK LLM CHAT FUNCTION ============
def call_bedrock(messages, temperature=0.3, max_tokens=800):
    # Convert OpenAI style messages to Claude-style
    prompt = ""
    for m in messages:
        if m["role"] == "system":
            prompt += f"System: {m['content']}\n"
        elif m["role"] == "user":
            prompt += f"Human: {m['content']}\n"
        elif m["role"] == "assistant":
            prompt += f"Assistant: {m['content']}\n"
    prompt += "Assistant:"

    # Invoke Bedrock model
    body = {
        "prompt": prompt,
        "max_tokens_to_sample": max_tokens,
        "temperature": temperature,
        "stop_sequences": ["\nHuman:"]
    }
    response = bedrock_client.invoke_model(
        modelId=BEDROCK_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body)
    )
    result = json.loads(response['body'].read())
    # 'completion' for Claude, 'generation' for others — check model docs
    answer = result.get("completion") or result.get("generation") or ""
    return answer.strip()

# ============ STREAMLIT SETUP ============
st.set_page_config(page_title="AWS Cost Optimizer", layout="wide")
st.sidebar.title("💡 Navigation")
page = st.sidebar.radio("Go to:", ["💬 Chat Assistant", "📊 Cost Analyzer"])

# ============ LOAD INITIAL DATA ============
if "preds_df" not in st.session_state:
    try:
        st.session_state.preds_df = load_csv_from_s3(S3_BUCKET, PREDICTIONS_CSV_KEY)
    except Exception as e:
        st.session_state.preds_df = pd.DataFrame()
        st.error(f"Failed to load data from S3: {e}")

df = st.session_state.preds_df

# ============ PAGE 1: CHAT ASSISTANT ============
if page == "💬 Chat Assistant":
    st.title("💬 Chat with Cost Optimizer Assistant")
    preview_df, acct_agg_df, serv_agg_df, context_text = df_summary_tables(df)
    st.markdown("**Current data context (preview):**")
    st.dataframe(preview_df)

    if not acct_agg_df.empty:
        st.markdown("**Total predicted cost by account:**")
        st.table(acct_agg_df.head(10))
    if not serv_agg_df.empty:
        st.markdown("**Total predicted cost by service:**")
        st.table(serv_agg_df.head(10))

    st.code(context_text)
    if "chat_history" not in st.session_state:
        preview_text = preview_df.to_string(index=False) if 'preview_df' in locals() else "No preview available."
        st.session_state.chat_history = [
            {
                "role": "system",
                "content": (
                    "You are an AWS cost optimization assistant. "
                    "When the user wants to take action (like stopping a service), "
                    "reply ONLY in JSON like: "
                    '{"action":"stop","account":"<account>","service":"<service>"} '
                    "Otherwise, reply normally."
                ),
            },
            {
                "role": "system",
                "content": "DATA_PREVIEW:\n" + preview_text
            }
        ]
    user_input = st.text_input("Ask a question or request (e.g., 'Which EC2 should I stop?')")

    if st.button("Send") and user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        messages = st.session_state.chat_history + [
            {"role": "user", "content": "CONTEXT:\n" + context_text}
        ]
        try:
            reply = call_bedrock(messages)
        except Exception as e:
            reply = f"Error contacting Bedrock: {e}"

        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.markdown("**Assistant:**")
        st.write(reply)

        # Try to parse action JSON
        try:
            parsed = json.loads(reply)
            if parsed.get("action") == "stop":
                account = parsed.get("account")
                service = parsed.get("service")
                if account and service:
                    st.warning(f"⚙️ Triggering Lambda for {service} (Account: {account})...")
                    result = trigger_lambda_for_service(account, service)
                    st.success(f"✅ Lambda Invoked: {result}")
                    acct_col = next((c for c in df.columns if c.lower() in ["account", "accountid", "account_id"]), None)
                    serv_col = next((c for c in df.columns if c.lower() in ["service", "servicename", "service_name"]), None)
                    if acct_col and serv_col:
                        df = df[~((df[acct_col].astype(str) == str(account)) & (df[serv_col].astype(str) == str(service)))]
                        st.session_state.preds_df = df.reset_index(drop=True)
                        save_df_to_s3(st.session_state.preds_df, S3_BUCKET, PREDICTIONS_CSV_KEY)
                        st.success("Table updated and saved to S3.")
        except Exception:
            pass
    st.subheader("📊 Current flagged predictions")
    st.dataframe(df)

# ============ PAGE 2: COST ANALYZER ============
elif page == "📊 Cost Analyzer":
    st.title("📊 Cost Analyzer & Manual Actions")
    if df.empty:
        st.info("No data loaded.")
    else:
        acct_col = next((c for c in df.columns if c.lower() in ["account", "accountid", "account_id"]), None)
        serv_col = next((c for c in df.columns if c.lower() in ["service", "servicename", "service_name"]), None)
        cost_col = next((c for c in df.columns if "predict" in c.lower() or "cost" in c.lower()), None)

        col_main, col_right = st.columns([3, 1])
        with col_right:
            st.markdown("**Summary (small)**")
            if serv_col and cost_col:
                serv_agg = (
                    df.groupby(serv_col)[cost_col]
                    .sum()
                    .reset_index()
                    .sort_values(cost_col, ascending=False)
                    .head(10)
                )
                st.table(serv_agg)
            elif acct_col and cost_col:
                acct_agg = (
                    df.groupby(acct_col)[cost_col]
                    .sum()
                    .reset_index()
                    .sort_values(cost_col, ascending=False)
                    .head(10)
                )
                st.table(acct_agg)
            else:
                st.table(df.head(5))

        with col_main:
            st.markdown("**Full table**")
            st.dataframe(df)
            selected = st.multiselect("Select rows to trigger Lambda:", df.index.tolist())
            if st.button("Trigger Lambda for selected"):
                for idx in selected:
                    row = df.loc[idx]
                    account = str(row[acct_col]) if acct_col else ""
                    service = str(row[serv_col]) if serv_col else ""
                    if not account or not service:
                        st.warning(f"Skipping row {idx} — missing account/service info.")
                        continue
                    result = trigger_lambda_for_service(account, service)
                    st.success(f"Triggered for {service} (Account {account}): {result}")
                df = df.drop(index=selected).reset_index(drop=True)
                st.session_state.preds_df = df
                save_df_to_s3(df, S3_BUCKET, PREDICTIONS_CSV_KEY)
                st.success("✅ Updated S3 and removed triggered entries.")
                st.rerun()
