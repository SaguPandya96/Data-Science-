"""Streamlit demo UI for the Amazon recommender API.

Run the API first:  uvicorn src.api.main:app --reload
Then:               streamlit run src/app/streamlit_app.py
"""

import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Amazon Recommender & Review Intelligence", layout="wide")
st.title("Amazon Recommender & Review Intelligence")
st.caption(
    "Hybrid collaborative-filtering + content-based recommender on the "
    "Amazon Reviews 2023 (All_Beauty) dataset"
)


def api_get(path: str, **params):
    try:
        response = requests.get(f"{API_URL}{path}", params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.error(f"API request failed: {exc}")
        return None


def api_post(path: str, payload: dict):
    try:
        response = requests.post(f"{API_URL}{path}", json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.error(f"API request failed: {exc}")
        return None


recommend_tab, similar_tab, sentiment_tab, rating_tab = st.tabs(
    ["Recommendations", "Similar products", "Review sentiment", "Rating inference"]
)

with recommend_tab:
    st.subheader("Personalized recommendations")
    st.write(
        "Enter a user ID from the dataset. "
        "Unknown IDs demonstrate the cold-start fallback path."
    )
    user_id = st.text_input("User ID", key="rec_user")
    k = st.slider("Number of recommendations", 1, 20, 10, key="rec_k")
    if st.button("Get recommendations", key="rec_btn") and user_id:
        data = api_get(f"/recommend/{user_id}", k=k)
        if data:
            st.info(f"Strategy: **{data['strategy']}** | Cold start: **{data['is_cold_start']}**")
            st.dataframe(data["recommendations"], use_container_width=True)

with similar_tab:
    st.subheader("Customers who liked this also liked...")
    st.write(
        "Content-based similarity over product titles - this is what handles "
        "brand-new items with no interaction history."
    )
    asin = st.text_input("Product ID (parent_asin)", key="sim_asin")
    sim_k = st.slider("Number of similar items", 1, 20, 10, key="sim_k")
    if st.button("Find similar products", key="sim_btn") and asin:
        data = api_get(f"/similar/{asin}", k=sim_k)
        if data:
            st.write(f"**Seed product:** {data['product_title']}")
            st.dataframe(data["similar_items"], use_container_width=True)

with sentiment_tab:
    st.subheader("Review sentiment analysis")
    text = st.text_area(
        "Review text",
        value="The color is gorgeous and it lasted all week without chipping.",
        key="sent_text",
    )
    if st.button("Analyze sentiment", key="sent_btn") and text:
        data = api_post("/sentiment", {"text": text})
        if data:
            st.metric("Predicted sentiment", data["label"], delta=f"score {data['score']:+.3f}")
            st.bar_chart(data["probabilities"])

with rating_tab:
    st.subheader("Rating inference with SHAP explanation")
    st.write(
        "Infers the star rating that goes with a piece of written feedback, and shows "
        "which features drove it. Note this is inference, not prediction: the ablation "
        "in `notebooks/02` shows that without the review text the model barely beats "
        "predicting the mean, so it cannot tell you how someone will rate a product "
        "they have not reviewed yet."
    )
    col1, col2 = st.columns(2)
    with col1:
        pr_user = st.text_input("User ID", key="pr_user")
        pr_verified = st.checkbox("Verified purchase", value=True, key="pr_verified")
    with col2:
        pr_asin = st.text_input("Product ID (parent_asin)", key="pr_asin")
    pr_text = st.text_area("Draft review text (optional)", key="pr_text")

    if st.button("Predict rating", key="pr_btn") and pr_user and pr_asin:
        data = api_post("/predict_rating", {
            "user_id": pr_user,
            "parent_asin": pr_asin,
            "review_text": pr_text,
            "verified_purchase": pr_verified,
        })
        if data:
            st.metric("Predicted rating", f"{data['predicted_rating']:.2f} / 5")
            st.write("**Why this prediction (SHAP contributions):**")
            st.dataframe(data["explanation"], use_container_width=True)
