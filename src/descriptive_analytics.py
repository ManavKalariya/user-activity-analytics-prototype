import pandas as pd
import numpy as np
import os

# Use compact sessionized file
SESSIONS_PATH = "pre-processed/session_sequences_embedded.parquet"

def load_sessions():
    """Load compact sessionized data with embedded sequences."""
    return pd.read_parquet(SESSIONS_PATH)

def generate_session_summaries(sessions):
    """
    Generate per-session descriptive analytics from embedded sequences.
    """
    summaries = []

    for _, row in sessions.iterrows():
        user = row["user_id"]
        session = row["user_session"]

        start_time = row["start_time"]
        end_time = row["end_time"]
        duration = (end_time - start_time).total_seconds() / 60.0

        categories = list(set([c for c in row["category_code"] if c is not None]))
        brands = list(set([b for b in row["brand"] if b is not None]))
        purchased_products = [p for e, p in zip(row["event_type"], row["product_id"]) if e == "purchase"]
        total_spend = sum([pr for e, pr in zip(row["event_type"], row["price"]) if e == "purchase"])

        # Most frequent action in this session
        event_counts = pd.Series(row["event_type"]).value_counts()
        frequent_action = event_counts.idxmax() if not event_counts.empty else "none"

        # Session timeline: events per minute
        # Not available in this structure, so skip or set to empty
        timeline = {}

        # Unique products interacted
        unique_products = len(set(row["product_id"]))

        # Purchase conversion rate for session
        num_events = len(row["event_type"])
        conversion_rate = len(purchased_products) / num_events if num_events > 0 else 0

        # Most/least expensive product IDs
        purchase_indices = [i for i, e in enumerate(row["event_type"]) if e == "purchase"]
        if purchase_indices:
            purchase_prices = [row["price"][i] for i in purchase_indices]
            purchase_products = [row["product_id"][i] for i in purchase_indices]
            most_expensive_idx = np.argmax(purchase_prices)
            least_expensive_idx = np.argmin(purchase_prices)
            most_expensive_product = purchase_products[most_expensive_idx]
            least_expensive_product = purchase_products[least_expensive_idx]
            most_expensive = purchase_prices[most_expensive_idx]
            least_expensive = purchase_prices[least_expensive_idx]
            avg_purchase_price = np.mean(purchase_prices)
        else:
            most_expensive_product = None
            least_expensive_product = None
            most_expensive = 0
            least_expensive = 0
            avg_purchase_price = 0

        # Session inactivity (max gap between events)
        if num_events > 1 and "start_time_seq" in row:
            time_diffs = np.diff([t for t in row["start_time_seq"]])
            max_inactivity = max(time_diffs) / 60.0 if len(time_diffs) > 0 else 0
        else:
            max_inactivity = 0

        summary_text = (
            f"User {user}, Session {session}: "
            f"duration {duration:.1f} minutes, browsed {len(categories)} categories "
            f"({', '.join(categories[:3])}...), interacted with {len(brands)} brands, "
            f"most frequent action was '{frequent_action}'. "
            f"Purchased {len(purchased_products)} items, spending ${total_spend:.2f}. "
            f"Most expensive purchase: ${most_expensive:.2f} (product {most_expensive_product}), "
            f"least expensive: ${least_expensive:.2f} (product {least_expensive_product}), "
            f"average purchase price: ${avg_purchase_price:.2f}. "
            f"Unique products interacted: {unique_products}. "
            f"Session purchase conversion rate: {conversion_rate:.2f}. "
            f"Max inactivity gap: {max_inactivity:.1f} min."
        )

        summaries.append({
            "user_id": user,
            "session_id": session,
            "duration_min": duration,
            "categories": categories,
            "brands": brands,
            "total_spend": total_spend,
            "frequent_action": frequent_action,
            "unique_products": unique_products,
            "conversion_rate": conversion_rate,
            "most_expensive_product": most_expensive_product,
            "least_expensive_product": least_expensive_product,
            "most_expensive_purchase": most_expensive,
            "least_expensive_purchase": least_expensive,
            "avg_purchase_price": avg_purchase_price,
            "max_inactivity_min": max_inactivity,
            "summary_text": summary_text
        })

    return pd.DataFrame(summaries)

def generate_user_summaries(session_summaries):
    """
    Aggregate per-session summaries to user-level summaries.
    """
    summaries = []

    grouped = session_summaries.groupby("user_id")
    for user, group in grouped:
        num_sessions = group["session_id"].nunique()
        total_duration = group["duration_min"].sum()
        total_spend = group["total_spend"].sum()
        categories = list(set([c for sublist in group["categories"] for c in sublist]))
        brands = list(set([b for sublist in group["brands"] for b in sublist]))
        frequent_action = group["frequent_action"].value_counts().idxmax()
        unique_products = sum(group["unique_products"])
        avg_conversion_rate = group["conversion_rate"].mean()
        most_expensive_product = group["most_expensive_product"].mode().iloc[0] if not group["most_expensive_product"].isnull().all() else None
        least_expensive_product = group["least_expensive_product"].mode().iloc[0] if not group["least_expensive_product"].isnull().all() else None
        avg_purchase_price = group["avg_purchase_price"].mean()
        max_inactivity = group["max_inactivity_min"].max()

        summary_text = (
            f"User {user}: {num_sessions} sessions, total duration {total_duration:.1f} minutes. "
            f"Browsed {len(categories)} unique categories ({', '.join(categories[:5])}...), "
            f"interacted with {len(brands)} brands. "
            f"Most frequent action: '{frequent_action}'. "
            f"Spent ${total_spend:.2f} across sessions. "
            f"Unique products interacted: {unique_products}. "
            f"Average session purchase conversion rate: {avg_conversion_rate:.2f}. "
            f"Most expensive product: {most_expensive_product}, "
            f"Least expensive product: {least_expensive_product}. "
            f"Average purchase price: ${avg_purchase_price:.2f}. "
            f"Max inactivity gap in any session: {max_inactivity:.1f} min."
        )

        summaries.append({
            "user_id": user,
            "num_sessions": num_sessions,
            "total_duration_min": total_duration,
            "categories": categories,
            "brands": brands,
            "total_spend": total_spend,
            "frequent_action": frequent_action,
            "unique_products": unique_products,
            "avg_conversion_rate": avg_conversion_rate,
            "most_expensive_product": most_expensive_product,
            "least_expensive_product": least_expensive_product,
            "avg_purchase_price": avg_purchase_price,
            "max_inactivity_min": max_inactivity,
            "summary_text": summary_text
        })

    return pd.DataFrame(summaries)

if __name__ == "__main__":
    print("Loading sessionized data...")
    df = load_sessions()

    print("Generating session-level summaries...")
    session_summaries_df = generate_session_summaries(df)

    print("Generating user-level summaries...")
    user_summaries_df = generate_user_summaries(session_summaries_df)

    # Save outputs
    os.makedirs("analytics", exist_ok=True)
    session_out = "analytics/session_summaries.csv"
    user_out = "analytics/user_summaries.csv"

    session_summaries_df.to_csv(session_out, index=False)
    user_summaries_df.to_csv(user_out, index=False)

    print(f"✅ Session summaries saved at {session_out}")
    print(f"✅ User summaries saved at {user_out}")

    print("\nSample Session Summary:")
    print(session_summaries_df.head(3)["summary_text"].to_list())

    print("\nSample User Summary:")
    print(user_summaries_df.head(3)["summary_text"].to_list())