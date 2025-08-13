import pandas as pd
from functools import reduce

def add_location_key(df, country_col="Country", city_col="City"):
    df[country_col] = df[country_col].fillna('').astype(str).str.strip()
    if city_col in df.columns:
        df[city_col] = df[city_col].fillna('').astype(str).str.strip()
        df['location_key'] = df.apply(
            lambda row: f"{row[country_col]}-{row[city_col]}" if row[city_col] else row[country_col],
            axis=1
        )
    else:
        df['location_key'] = df[country_col]
    return df

def group_topic(df, location_key_col="location_key", text_cols=[]):
    df = df.fillna("")
    df["combined"] = df[text_cols].apply(lambda x: " | ".join([str(i) for i in x if i != ""]), axis=1)
    grouped = df.groupby(location_key_col)["combined"].apply(list).reset_index()
    return grouped

# Load CSVs
accommodation = pd.read_csv(r"data\Accommodations.csv")
activity = pd.read_csv(r"data\Activity.csv")
dishes = pd.read_csv(r"data\Dishes.csv")
restaurants = pd.read_csv(r"data\Restaurants.csv")
scams = pd.read_csv(r"data\Scams.csv")
transport = pd.read_csv(r"data\Transport.csv")
visa = pd.read_csv(r"data\VISA.csv")

# Add location_key column
accommodation = add_location_key(accommodation)
activity = add_location_key(activity)
dishes = add_location_key(dishes)
restaurants = add_location_key(restaurants)
scams = add_location_key(scams)
transport = add_location_key(transport, city_col="From")
visa = add_location_key(visa, city_col=None)  # No city in visa data

# Group each topic
accommodation_g = group_topic(accommodation, text_cols=["Accommodation Name", "Accommodation Details", "Type", "Avg Night Price (USD)"])
accommodation_g = accommodation_g.rename(columns={"combined": "accommodation"})

activity_g = group_topic(activity, text_cols=["Activity", "Description", "Type of Traveler", "Duration", "Budget (USD)", "Tips and Recommendations"])
activity_g = activity_g.rename(columns={"combined": "activities"})

dishes_g = group_topic(dishes, text_cols=["Dish Name", "Dish Details", "Type", "Avg Price (USD)", "Best For"])
dishes_g = dishes_g.rename(columns={"combined": "dishes"})

restaurants_g = group_topic(restaurants, text_cols=["Restaurant Name", "Type of Cuisine", "Meals Served", "Recommended Dish", "Meal Description", "Avg Price per Person (USD)"])
restaurants_g = restaurants_g.rename(columns={"combined": "restaurants"})

scams_g = group_topic(scams, text_cols=["Scam Type", "Description", "Location", "Prevention Tips"])
scams_g = scams_g.rename(columns={"combined": "scams"})

transport_g = group_topic(transport, text_cols=["To", "Transport Mode", "Provider", "Schedule", "Duration in hours", "Price Range in USD"])
transport_g = transport_g.rename(columns={"combined": "transport"})

visa_g = visa.groupby("location_key")["Answer"].apply(list).reset_index().rename(columns={"Answer": "visa_info"})

# Merge all grouped DataFrames on location_key
dfs = [accommodation_g, activity_g, dishes_g, restaurants_g, scams_g, transport_g]
master_df = reduce(lambda left, right: pd.merge(left, right, on="location_key", how="outer"), dfs)

# Merge visa separately (also on location_key)
master_df = master_df.merge(visa_g, on="location_key", how="left")

# Save final merged dataset
master_df.to_csv("travel_master.csv", index=False)
print("✅ Final dataset saved to travel_master.csv")
