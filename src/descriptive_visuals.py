import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

SESSION_SUMMARY_PATH = "analytics/session_summaries.csv"
USER_SUMMARY_PATH = "analytics/user_summaries.csv"

def plot_session_summary(session_df):
    """Visualize session-level statistics."""
    os.makedirs("analytics/plots", exist_ok=True)

    # 1. Distribution of session durations
    plt.figure(figsize=(8,5))
    sns.histplot(session_df["duration_min"], bins=30, kde=True)
    plt.title("Distribution of Session Durations (minutes)")
    plt.xlabel("Duration (minutes)")
    plt.ylabel("Count")
    plt.savefig("analytics/plots/session_duration_dist.png")
    plt.close()

    # 2. Spending per session
    plt.figure(figsize=(8,5))
    sns.histplot(session_df["total_spend"], bins=30, kde=True, color="green")
    plt.title("Distribution of Spending per Session")
    plt.xlabel("Spending ($)")
    plt.ylabel("Count")
    plt.savefig("analytics/plots/session_spend_dist.png")
    plt.close()

    # 3. Purchase conversion rate per session
    plt.figure(figsize=(8,5))
    sns.histplot(session_df["conversion_rate"], bins=30, kde=True, color="orange")
    plt.title("Session Purchase Conversion Rate")
    plt.xlabel("Purchases / Events")
    plt.ylabel("Session Count")
    plt.savefig("analytics/plots/session_conversion_rate.png")
    plt.close()

    # 4. Unique products interacted per session
    plt.figure(figsize=(8,5))
    sns.histplot(session_df["unique_products"], bins=30, kde=True, color="blue")
    plt.title("Unique Products Interacted per Session")
    plt.xlabel("Unique Products")
    plt.ylabel("Session Count")
    plt.savefig("analytics/plots/session_unique_products.png")
    plt.close()

    # 5. Most frequent actions per session
    plt.figure(figsize=(8,5))
    sns.countplot(y=session_df["frequent_action"], order=session_df["frequent_action"].value_counts().index)
    plt.title("Most Frequent Actions per Session")
    plt.xlabel("Count")
    plt.ylabel("Action")
    plt.savefig("analytics/plots/session_frequent_actions.png")
    plt.close()

    # 6. Most expensive purchase per session
    plt.figure(figsize=(8,5))
    sns.histplot(session_df["most_expensive_purchase"], bins=30, kde=True, color="red")
    plt.title("Most Expensive Purchase per Session")
    plt.xlabel("Price ($)")
    plt.ylabel("Session Count")
    plt.savefig("analytics/plots/session_most_expensive_purchase.png")
    plt.close()

    # 7. Least expensive purchase per session
    plt.figure(figsize=(8,5))
    sns.histplot(session_df["least_expensive_purchase"], bins=30, kde=True, color="purple")
    plt.title("Least Expensive Purchase per Session")
    plt.xlabel("Price ($)")
    plt.ylabel("Session Count")
    plt.savefig("analytics/plots/session_least_expensive_purchase.png")
    plt.close()

    # 8. Average purchase price per session
    plt.figure(figsize=(8,5))
    sns.histplot(session_df["avg_purchase_price"], bins=30, kde=True, color="brown")
    plt.title("Average Purchase Price per Session")
    plt.xlabel("Average Price ($)")
    plt.ylabel("Session Count")
    plt.savefig("analytics/plots/session_avg_purchase_price.png")
    plt.close()

    # 9. Max inactivity gap per session
    plt.figure(figsize=(8,5))
    sns.histplot(session_df["max_inactivity_min"], bins=30, kde=True, color="grey")
    plt.title("Max Inactivity Gap per Session (minutes)")
    plt.xlabel("Max Inactivity (min)")
    plt.ylabel("Session Count")
    plt.savefig("analytics/plots/session_max_inactivity.png")
    plt.close()

def plot_user_summary(user_df):
    """Visualize user-level aggregated statistics."""
    os.makedirs("analytics/plots", exist_ok=True)

    # 1. Number of sessions per user
    plt.figure(figsize=(8,5))
    sns.histplot(user_df["num_sessions"], bins=30, kde=False)
    plt.title("Distribution of Number of Sessions per User")
    plt.xlabel("Number of Sessions")
    plt.ylabel("User Count")
    plt.savefig("analytics/plots/user_sessions_dist.png")
    plt.close()

    # 2. Total spending per user
    plt.figure(figsize=(8,5))
    sns.histplot(user_df["total_spend"], bins=30, kde=True, color="purple")
    plt.title("Distribution of Total Spending per User")
    plt.xlabel("Spending ($)")
    plt.ylabel("User Count")
    plt.savefig("analytics/plots/user_spend_dist.png")
    plt.close()

    # 3. Total duration per user
    plt.figure(figsize=(8,5))
    sns.histplot(user_df["total_duration_min"], bins=30, kde=True, color="orange")
    plt.title("Total Session Duration per User")
    plt.xlabel("Total Duration (minutes)")
    plt.ylabel("User Count")
    plt.savefig("analytics/plots/user_total_duration.png")
    plt.close()

    # 4. Average session purchase conversion rate per user
    plt.figure(figsize=(8,5))
    sns.histplot(user_df["avg_conversion_rate"], bins=30, kde=True, color="green")
    plt.title("Average Session Purchase Conversion Rate per User")
    plt.xlabel("Avg Purchases / Events")
    plt.ylabel("User Count")
    plt.savefig("analytics/plots/user_avg_conversion_rate.png")
    plt.close()

    # 5. Unique products interacted per user
    plt.figure(figsize=(8,5))
    sns.histplot(user_df["unique_products"], bins=30, kde=True, color="blue")
    plt.title("Unique Products Interacted per User")
    plt.xlabel("Unique Products")
    plt.ylabel("User Count")
    plt.savefig("analytics/plots/user_unique_products.png")
    plt.close()

    # 6. Most frequent actions per user
    plt.figure(figsize=(8,5))
    sns.countplot(y=user_df["frequent_action"], order=user_df["frequent_action"].value_counts().index)
    plt.title("Most Frequent Actions per User")
    plt.xlabel("Count")
    plt.ylabel("Action")
    plt.savefig("analytics/plots/user_frequent_actions.png")
    plt.close()

    # 7. Most expensive purchase per user
    plt.figure(figsize=(8,5))
    sns.histplot(user_df["avg_purchase_price"], bins=30, kde=True, color="red")
    plt.title("Average Purchase Price per User")
    plt.xlabel("Average Price ($)")
    plt.ylabel("User Count")
    plt.savefig("analytics/plots/user_avg_purchase_price.png")
    plt.close()

    # 8. Max inactivity gap in any session per user
    plt.figure(figsize=(8,5))
    sns.histplot(user_df["max_inactivity_min"], bins=30, kde=True, color="grey")
    plt.title("Max Inactivity Gap in Any Session per User (minutes)")
    plt.xlabel("Max Inactivity (min)")
    plt.ylabel("User Count")
    plt.savefig("analytics/plots/user_max_inactivity.png")
    plt.close()

if __name__ == "__main__":
    print("Loading summaries...")
    session_df = pd.read_csv(SESSION_SUMMARY_PATH)
    user_df = pd.read_csv(USER_SUMMARY_PATH)

    print("Plotting session-level analytics...")
    plot_session_summary(session_df)

    print("Plotting user-level analytics...")
    plot_user_summary(user_df)

    print("✅ Plots saved in analytics/plots/")