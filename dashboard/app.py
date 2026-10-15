
import streamlit as st
import pandas as pd

from src.priority.priority_engine import assign_priority


st.set_page_config(
    page_title="Customer Review Sentiment Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("Customer Review Sentiment Dashboard")
st.caption("Monitor feedback and prioritize customer complaints.")

# Demo data until the Kafka and ML pipeline is integrated.
reviews = [
    {
        "review_id": "R001",
        "product": "Wireless Headphones",
        "review": "Battery stopped working after two days.",
        "sentiment": "negative",
        "confidence": 0.96,
        "keywords": "battery, stopped working",
        "priority": "High",
        "rating": 1,
    },
    {
        "review_id": "R002",
        "product": "Laptop Stand",
        "review": "Very sturdy and easy to use.",
        "sentiment": "positive",
        "confidence": 0.94,
        "keywords": "",
        "priority": "Low",
        "rating": 5,
    },
    {
        "review_id": "R003",
        "product": "Phone Charger",
        "review": "The charger overheats during use.",
        "sentiment": "negative",
        "confidence": 0.91,
        "keywords": "overheats, safety",
        "priority": "Critical",
        "rating": 1,
    },
    {
        "review_id": "R004",
        "product": "Laptop Bag",
        "review": "Good quality and fast delivery.",
        "sentiment": "positive",
        "confidence": 0.89,
        "keywords": "",
        "priority": "Low",
        "rating": 4,
    },
    {
        "review_id": "R005",
        "product": "Bluetooth Speaker",
        "review": "Sound keeps cutting out.",
        "sentiment": "negative",
        "confidence": 0.87,
        "keywords": "sound cutting out",
        "priority": "Medium",
        "rating": 2,
    },
]

df = pd.DataFrame(reviews)

df["priority"] = df.apply(
    lambda row: assign_priority(
        sentiment=row["sentiment"],
        rating=int(row["rating"]),
        keywords=row["keywords"],
    ),
    axis=1,
)


st.sidebar.header("Review Filters")

sentiment_filter = st.sidebar.selectbox(
    "Sentiment", ["All", "negative", "positive"]
)
priority_filter = st.sidebar.selectbox(
    "Priority", ["All", "Critical", "High", "Medium", "Low"]
)

filtered = df.copy()

if sentiment_filter != "All":
    filtered = filtered[filtered["sentiment"] == sentiment_filter]

if priority_filter != "All":
    filtered = filtered[filtered["priority"] == priority_filter]

total = len(filtered)
negative = int((filtered["sentiment"] == "negative").sum())
positive = int((filtered["sentiment"] == "positive").sum())
urgent = int(filtered["priority"].isin(["Critical", "High"]).sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Reviews", total)
c2.metric("Negative Reviews", negative)
c3.metric("Positive Reviews", positive)
c4.metric("Critical / High", urgent)

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Sentiment Distribution")
    if not filtered.empty:
        st.bar_chart(filtered["sentiment"].value_counts())
    else:
        st.info("No reviews match these filters.")

with right:
    st.subheader("Priority Distribution")
    if not filtered.empty:
        st.bar_chart(filtered["priority"].value_counts())
    else:
        st.info("No reviews match these filters.")

st.divider()
st.subheader("🚨 Prioritized Complaint Queue")

priority_order = {
    "Critical": 0,
    "High": 1,
    "Medium": 2,
    "Low": 3,
}

complaints = filtered[filtered["sentiment"] == "negative"].copy()
complaints["priority_rank"] = complaints["priority"].map(priority_order)
complaints = complaints.sort_values(
    by=["priority_rank", "confidence"],
    ascending=[True, False],
)

if complaints.empty:
    st.success("No negative reviews match the selected filters.")
else:
    for _, review in complaints.iterrows():
        title = (
            f"{review['priority']} | {review['product']} "
            f"| {review['review_id']}"
        )
        with st.expander(title):
            st.write("**Review:**", review["review"])
            st.write("**Rating:**", f"{review['rating']}/5")
            st.write("**Model confidence:**", f"{review['confidence']:.0%}")
            st.write("**Complaint keywords:**", review["keywords"])

st.divider()
st.subheader("All Reviews")

st.dataframe(
    filtered.drop(columns=["priority_rank"], errors="ignore"),
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "Prototype using sample data. Live model and streaming integration "
    "will be connected after the team pipeline is ready."
)
