
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Auto layout
plt.rcParams.update({"figure.autolayout": True})

DATA_CSV = "retail_sales.csv"
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)


# ----------------------------
# Generate Dataset
# ----------------------------
def generate_dataset_if_missing():

    if os.path.exists(DATA_CSV):
        return

    rng = np.random.default_rng(42)

    n = 10000

    start = np.datetime64("2023-01-01")

    dates = start + rng.integers(0, 365 * 2, size=n).astype("timedelta64[D]")

    categories = [
        "Electronics",
        "Clothing",
        "Home & Kitchen",
        "Beauty",
        "Sports"
    ]

    subcats = {
        "Electronics": ["Mobiles", "Laptops", "Audio", "Accessories"],
        "Clothing": ["Men", "Women", "Kids"],
        "Home & Kitchen": ["Furniture", "Appliances", "Decor"],
        "Beauty": ["Skincare", "Haircare", "Fragrance"],
        "Sports": ["Fitness", "Outdoor", "Team Sports"]
    }

    regions = ["North", "South", "East", "West", "Central"]

    data = []

    for i in range(1, n + 1):

        cat = rng.choice(categories)
        sub = rng.choice(subcats[cat])

        region = rng.choice(
            regions,
            p=[0.25, 0.20, 0.20, 0.20, 0.15]
        )

        unit_price = round(rng.uniform(5, 800), 2)

        qty = max(1, int(round(rng.normal(2.2, 1.1))))

        discount = round(rng.beta(2, 8) * 0.4, 2)

        revenue = unit_price * qty * (1 - discount)

        cost_ratio = rng.uniform(0.5, 0.85)

        profit = round(revenue * (1 - cost_ratio), 2)

        age = int(np.clip(rng.normal(34, 10), 18, 65))

        gender = rng.choice(
            ["Female", "Male", "Other"],
            p=[0.48, 0.48, 0.04]
        )

        customer_id = rng.integers(1000, 3000)

        # Save dates as DD-MM-YYYY
        date = pd.Timestamp(dates[i - 1]).strftime("%d-%m-%Y")

        data.append([
            i,
            date,
            cat,
            sub,
            region,
            unit_price,
            qty,
            discount,
            round(revenue, 2),
            profit,
            age,
            gender,
            customer_id
        ])

    df = pd.DataFrame(data, columns=[
        "order_id",
        "order_date",
        "category",
        "subcategory",
        "region",
        "unit_price",
        "quantity",
        "discount",
        "revenue",
        "profit",
        "customer_age",
        "gender",
        "customer_id"
    ])

    # Missing values
    idx = rng.choice(df.index, size=100, replace=False)
    df.loc[idx, "customer_age"] = np.nan

    # Outliers
    outliers = rng.choice(df.index, size=5, replace=False)
    df.loc[outliers, "revenue"] *= 10
    df.loc[outliers, "profit"] *= 10

    df.to_csv(DATA_CSV, index=False)

    print("Dataset created.")


# ----------------------------
# EDA
# ----------------------------
def eda():

    df = pd.read_csv(DATA_CSV)

    # Convert DD-MM-YYYY
    df["order_date"] = pd.to_datetime(
        df["order_date"],
        format="%d-%m-%Y"
    )

    df["customer_age"] = df["customer_age"].fillna(
        df["customer_age"].median()
    )

    # KPIs
    total_rev = df["revenue"].sum()
    total_profit = df["profit"].sum()
    aov = df["revenue"].mean()
    orders = len(df)
    customers = df["customer_id"].nunique()
    revenue_customer = total_rev / customers

    with open("eda_summary.txt", "w") as f:

        f.write("===== KPI SUMMARY =====\n\n")
        f.write(f"Orders : {orders}\n")
        f.write(f"Customers : {customers}\n")
        f.write(f"Revenue : {total_rev:.2f}\n")
        f.write(f"Profit : {total_profit:.2f}\n")
        f.write(f"Average Order Value : {aov:.2f}\n")
        f.write(f"Revenue per Customer : {revenue_customer:.2f}\n\n")

        f.write("===== DESCRIPTIVE STATISTICS =====\n\n")
        f.write(df.describe().to_string())

        f.write("\n\n===== MISSING VALUES =====\n\n")
        f.write(df.isnull().sum().to_string())

    # Set datetime index
    df = df.set_index("order_date")

    # Monthly Revenue
    plt.figure()
    df["revenue"].resample("ME").sum().plot()
    plt.title("Monthly Revenue Trend")
    plt.xlabel("Month")
    plt.ylabel("Revenue")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/monthly_revenue.png")
    plt.close()

    # Revenue by Category
    plt.figure()
    df.groupby("category")["revenue"].sum().sort_values().plot(kind="barh")
    plt.title("Revenue by Category")
    plt.xlabel("Revenue")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/revenue_by_category.png")
    plt.close()

    # Top Profit Subcategories
    plt.figure()
    df.groupby(
        ["category", "subcategory"]
    )["profit"].sum().sort_values(
        ascending=False
    ).head(10).plot(kind="bar")

    plt.title("Top 10 Subcategories by Profit")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/top_subcategories_profit.png")
    plt.close()

    # Revenue by Region
    plt.figure()
    df.groupby("region")["revenue"].sum().plot(kind="bar")
    plt.title("Revenue by Region")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/revenue_by_region.png")
    plt.close()

    # Age Distribution
    plt.figure()
    df["customer_age"].plot(kind="hist", bins=20)
    plt.title("Customer Age Distribution")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/age_distribution.png")
    plt.close()

    # Revenue Distribution
    plt.figure()
    df["revenue"].plot(kind="hist", bins=30)
    plt.title("Revenue Distribution")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/revenue_distribution.png")
    plt.close()

    # Profit Distribution
    plt.figure()
    df["profit"].plot(kind="hist", bins=30)
    plt.title("Profit Distribution")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/profit_distribution.png")
    plt.close()

    # Gender Distribution
    plt.figure()
    df["gender"].value_counts().plot(kind="bar")
    plt.title("Gender Distribution")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/gender_distribution.png")
    plt.close()

    print("\nEDA Completed Successfully!")
    print("Dataset :", DATA_CSV)
    print("Summary :", "eda_summary.txt")
    print("Figures :", FIG_DIR)


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":

    # Delete old CSV if you want to regenerate
    # os.remove(DATA_CSV)

    generate_dataset_if_missing()

    eda()
