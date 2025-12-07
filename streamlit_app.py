# streamlit_app.py
"""
Streamlit app for Task 2 — Two-Dashboard AI Feedback System
User Dashboard (public): submit rating + review -> AI reply -> saved
Admin Dashboard (internal): live list of submissions, AI summaries, recommended actions + analytics

Storage: CSV file in app directory ('submissions.csv')
LLM: Perplexity sonar-pro via REST (requests). Set API key in STREAMLIT secrets or env var PERPLEXITY_API_KEY.
"""

import streamlit as st
import pandas as pd
import requests
import json
import os
import re
from datetime import datetime
from io import StringIO
import unicodedata

# ---------- CONFIG ----------
API_URL = "https://api.perplexity.ai/chat/completions"
MODEL = "sonar-pro"
STORAGE_FILE = "submissions.csv"
# ----------------------------

st.set_page_config(layout="wide", page_title="AI Feedback System")


def clean_text(s: str):
    if not isinstance(s, str):
        return s
    # Normalize unicode to avoid weird characters like â€™
    s = unicodedata.normalize("NFKD", s)
    # Replace smart quotes with normal quotes
    replacements = {
    "\u2019": "'",   # right apostrophe
    "\u2018": "'",   # left apostrophe
    "\uFF07": "'",   # full-width apostrophe
    "\u201C": '"',   # left double quote
    "\u201D": '"',   # right double quote
    "\u2032": "'",   # prime (sometimes used)
    "\u2033": '"'
    }
    for bad, good in replacements.items():
        s = s.replace(bad, good)
    return s


# ---------- Helpers: LLM call & JSON extractor ----------
def call_perplexity(prompt, api_key, max_tokens=500, temperature=0.0):
    """Call Perplexity sonar-pro; returns raw text or None."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        # "temperature": temperature # if supported by endpoint; optional
    }
    try:
        r = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        if r.status_code == 200:
            data = r.json()
            return data["choices"][0]["message"]["content"]
        else:
            st.error(f"LLM API error {r.status_code}: {r.text[:300]}")
            return None
    except Exception as e:
        st.error(f"LLM call exception: {e}")
        return None

def extract_json_fragment(text):
    """Return parsed JSON from within a text blob, or None."""
    if text is None: 
        return None
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except:
        return None

# ---------- Storage helpers ----------
def ensure_storage():
    try:
        df = pd.read_csv(STORAGE_FILE)
    except FileNotFoundError:
        df = pd.DataFrame(columns=[
            "id","timestamp","rating","review","ai_reply","ai_summary","ai_actions"
        ])
        df.to_csv(STORAGE_FILE, index=False)
    return

def load_submissions():
    try:
        df = pd.read_csv(STORAGE_FILE)
    except FileNotFoundError:
        df = pd.DataFrame(columns=[
            "id","timestamp","rating","review","ai_reply","ai_summary","ai_actions"
        ])
    return df

def append_submission(row: dict):
    df = load_submissions()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(STORAGE_FILE, index=False, encoding="utf-8-sig")

def update_submission(index, updates: dict):
    df = load_submissions()
    for k,v in updates.items():
        df.at[index, k] = v
    df.to_csv(STORAGE_FILE, index=False, encoding="utf-8-sig")

import os

def get_api_key():
    # First: environment variable
    key = os.getenv("PERPLEXITY_API_KEY")
    if key:
        return key
    
    # Second: Streamlit secrets (only available on deployment)
    try:
        return st.secrets["PERPLEXITY_API_KEY"]
    except:
        return None

# ---------- Prompts ----------
PROMPT_USER_REPLY = """You are a helpful customer-response assistant.
Given the user's rating and review, respond politely and helpfully in a single short paragraph.

Rules:
- Output exactly one short reply (no JSON).
- Keep it <= 2 sentences.
- Tone: friendly and constructive.
- If rating <= 2, apologize and offer to escalate; if rating >= 4, thank and suggest next positive step.

Input:
Rating: {rating}
Review: "{review}"
"""

PROMPT_ADMIN_SUMMARY = """You are a concise summariser for admins.
Given a user's review, provide a one-line summary and a single short recommended action for internal teams.

Return JSON ONLY in this format:
{{ "summary": "<one-line summary>", "recommended_action": "<one short action>" }}

Review:
"{review}"
Rating: {rating}
"""

# ---------- UI: Tabs for User and Admin ----------
ensure_storage()
tab1, tab2 = st.tabs(["User Dashboard", "Admin Dashboard"])

# ---------- USER DASHBOARD ----------
with tab1:
    st.header("User Dashboard — Submit a Review")
    st.markdown("Select rating, write a short review, submit — an AI reply will be generated and saved.")
    col1, col2 = st.columns([3,1])

    with col1:
        rating = st.selectbox("Star rating", [5,4,3,2,1], index=0)
        review_text = st.text_area("Write your review", height=140, placeholder="Write a short review...")
        submit_btn = st.button("Submit review")

    with col2:
        st.markdown("**Quick tips**")
        st.write("- Be concise (1–3 sentences).")
        st.write("- Mention what you liked or disliked.")
        st.write("- The AI reply will be short and friendly.")

    if submit_btn:
        if not review_text.strip():
            st.warning("Please write a review before submitting.")
        else:
            api_key = get_api_key()
            if not api_key:
                st.error("Perplexity API key required. Add to Streamlit secrets or paste it in the sidebar input.")
            else:
                st.info("Generating AI reply...")
                prompt = PROMPT_USER_REPLY.format(rating=rating, review=review_text.replace('"','\\"'))
                ai_reply = clean_text(call_perplexity(prompt, api_key) or "")
                # Build row and save
                now = datetime.utcnow().isoformat()
                df = load_submissions()
                next_id = int(df["id"].max())+1 if (not df.empty and pd.notnull(df["id"].max())) else 1
                row = {
                    "id": next_id,
                    "timestamp": now,
                    "rating": rating,
                    "review": clean_text(review_text),
                    "ai_reply": ai_reply,
                    "ai_summary": "",
                    "ai_actions": ""
                }
                append_submission(row)
                st.success("Saved! AI reply below:")
                st.write(ai_reply)


# ---------- ADMIN DASHBOARD ----------
with tab2:
    st.header("Admin Dashboard — Submissions & Analytics")

    import os
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if api_key is None:
        try:
            api_key = st.secrets["PERPLEXITY_API_KEY"]
        except:
            api_key = None
    refresh_btn = st.button("Refresh submissions")
    df = load_submissions()

    if df.empty:
        st.info("No submissions yet.")
    else:
        # Ensure timestamp & date columns
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['date'] = df['timestamp'].dt.date.astype(str)

        st.subheader("Overall Analytics")

        # -------- FILTERS --------
        rating_filter = st.multiselect(
            "Filter by Rating:",
            options=[1,2,3,4,5],
            default=[1,2,3,4,5]
        )

        filtered_df = df[df['rating'].isin(rating_filter)].copy()

        # -------- BASIC STATS --------
        colA, colB, colC = st.columns(3)
        with colA:
            st.metric("Total Reviews", len(filtered_df))
        with colB:
            st.metric("Average Rating", round(filtered_df['rating'].mean(), 2))
        with colC:
            filtered_df = filtered_df.copy()
            filtered_df['review_length'] = filtered_df['review'].apply(lambda x: len(str(x).split()))
            st.metric("Avg Review Length (words)", round(filtered_df['review_length'].mean(), 2))

        # -------- PIE CHART --------
        st.subheader("Rating Distribution")
        import plotly.express as px
        rating_counts = filtered_df['rating'].value_counts().sort_index()
        fig_pie = px.pie(
            names=rating_counts.index,
            values=rating_counts.values,
            title="Distribution of Ratings",
            color=rating_counts.index,
            color_discrete_sequence=px.colors.sequential.Blues
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        # -------- BAR CHART --------
        st.subheader("Average Rating Over Time")
        daily_avg = filtered_df.groupby('date')['rating'].mean().reset_index()
        fig_bar = px.bar(
            daily_avg,
            x='date',
            y='rating',
            title="Daily Average Rating",
            labels={"rating": "Average Rating", "date": "Date"},
            color='rating',
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")

        # -------- BULK SUMMARY BUTTON --------
        st.subheader("AI Tools")
        col_bulk1, col_bulk2 = st.columns(2)

        with col_bulk1:
            if st.button("Generate Summaries for ALL Reviews"):
                if not api_key:
                    st.error("Perplexity API key not found!")
                else:
                    with st.spinner("Generating summaries for all rows..."):
                        for idx, row in df.iterrows():
                            prompt = PROMPT_ADMIN_SUMMARY.format(
                                review=row['review'].replace('"','\\"'),
                                rating=int(row['rating'])
                            )
                            out = call_perplexity(prompt, api_key)
                            parsed = extract_json_fragment(out)
                            if parsed:
                                update_submission(idx, {
                                    "ai_summary": clean_text(parsed.get("summary", "")),
                                    "ai_actions": clean_text(parsed.get("recommended_action", ""))
                                })
                    st.success("All summaries generated!")

        with col_bulk2:
            if st.button("Generate Summaries for Filtered Reviews Only"):
                if not api_key:
                    st.error("API key not found!")
                else:
                    with st.spinner("Generating summaries for filtered rows..."):
                        for idx, row in filtered_df.iterrows():
                            real_idx = df[df['id'] == row['id']].index[0]  
                            prompt = PROMPT_ADMIN_SUMMARY.format(
                                review=row['review'].replace('"','\\"'),
                                rating=int(row['rating'])
                            )
                            out = call_perplexity(prompt, api_key)
                            parsed = extract_json_fragment(out)
                            if parsed:
                                update_submission(real_idx, {
                                    "ai_summary": clean_text(parsed.get("summary", "")),
                                    "ai_actions": clean_text(parsed.get("recommended_action", ""))
                                })
                    st.success("Filtered summaries generated!")

        st.markdown("---")
        st.subheader("Raw Submissions (Filtered)")

        # -------- DISPLAY LIST --------
        for idx, row in filtered_df.iterrows():
            st.markdown(f"### Review ID {int(row['id'])} — ⭐ {int(row['rating'])}")
            st.write(f"**Date:** {row['date']}")
            st.write(f"**Review:** {row['review']}")
            st.write(f"**AI Reply:** {row['ai_reply']}")

            cols = st.columns([1,1,1,4])

            with cols[0]:
                if st.button(f"Summarize #{int(row['id'])}", key=f"summ_{idx}"):
                    prompt = PROMPT_ADMIN_SUMMARY.format(
                        review=row['review'].replace('"','\\"'),
                        rating=int(row['rating'])
                    )
                    out = call_perplexity(prompt, api_key)
                    parsed = extract_json_fragment(out)
                    if parsed:
                        update_submission(df[df['id']==row['id']].index[0], {
                            "ai_summary": clean_text(parsed.get("summary", "")),
                            "ai_actions": clean_text(parsed.get("recommended_action", ""))
                        })
                        st.success("Summary updated!")

            with cols[1]:
                if st.button(f"Regenerate Reply #{int(row['id'])}", key=f"reply_{idx}"):
                    prompt = PROMPT_USER_REPLY.format(
                        rating=int(row['rating']),
                        review=row['review'].replace('"','\\"')
                    )
                    out = call_perplexity(prompt, api_key)
                    update_submission(df[df['id']==row['id']].index[0], {"ai_reply": clean_text(out)})
                    st.success("Reply regenerated!")

            with cols[2]:
                if st.button(f"Delete #{int(row['id'])}", key=f"del_{idx}"):
                    df2 = df.drop(df[df['id']==row['id']].index[0]).reset_index(drop=True)
                    df2.to_csv(STORAGE_FILE, index=False)
                    st.experimental_rerun()

            with cols[3]:
                st.write(f"**Summary:** {row.get('ai_summary','')}")
                st.write(f"**Recommended Action:** {row.get('ai_actions','')}")

            st.markdown("---")

        # -------- DOWNLOAD CSV --------
        csv = filtered_df.to_csv(index=False)
        st.download_button("Download Filtered CSV", data=csv, file_name="filtered_reviews.csv", mime="text/csv")
