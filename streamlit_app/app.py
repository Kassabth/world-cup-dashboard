import streamlit as st
import pandas as pd
import altair as alt
import os
from datetime import datetime


st.set_page_config(page_title="World Cup Dashboard", layout="wide")

# Introduction
st.markdown("""
# FIFA World Cup Big Data Project

Welcome to the FIFA World Cup Dashboard! This project analyzes historical and live match data to uncover trends and insights from the world's most prestigious football tournament.

**Project Goals:**
- Combine historical and live data for a comprehensive view of World Cup matches
- Provide interactive visualizations and KPIs for fans and analysts
- Enable exploration of trends across different editions

**Data Sources:**
- Historical CSV data (static)
- Live match data from RapidAPI (dynamic)

**Transformations:**
- Data cleaning and normalization
- Merging and deduplication
- Conversion to Parquet format for efficient analytics

**What You Can Learn:**
- Key match statistics and trends by edition
- Team performance, scoring patterns, and more
- Insights into home vs away wins, top scorers, and attendance
""")

# Glossary
with st.expander("Glossary: Column Definitions"):
    st.markdown("""
    - **match_datetime**: Date and time of the match
    - **home_team**: Team designated as 'home'
    - **away_team**: Team designated as 'away'
    - **home_score**: Goals scored by the home team
    - **away_score**: Goals scored by the away team
    - **location**: Stadium and city where the match was played
    - **winner_team**: Team that won the match (or 'Draw')
    - **phase_label**: Stage of the tournament (e.g., Group, Quarterfinal)
    - **attendance_number**: Number of spectators
    - **edition**: Year of the World Cup
    - **source**: Data origin (historical or live API)
    """)

# Title
st.title("🏆 World Cup Dashboard")

# Show last updated timestamp
parquet_path = "datalake/combined/final_matches.parquet"
last_updated = datetime.fromtimestamp(os.path.getmtime(parquet_path)).strftime("%Y-%m-%d %H:%M:%S")
st.caption(f"Last updated: {last_updated}")

# Load dataset with caching for performance
@st.cache_data(ttl=0)
def load_data():
    df = pd.read_parquet(parquet_path)
    df["edition"] = df["edition"].astype(int)
    return df

df = load_data()

# Edition selector
selected_edition = st.selectbox("Choose a World Cup edition:", sorted(df["edition"].unique(), reverse=True))

# Filtered data
filtered_df = df[df["edition"] == selected_edition]

# Debug info
st.write("📅 Edition selected:", selected_edition)
st.write("🔢 Matches shown:", len(filtered_df))

# KPIs
col1, col2 = st.columns(2)
col1.metric("Total Matches", len(filtered_df))
col2.metric("Average Score", round((filtered_df["home_score"] + filtered_df["away_score"]).mean(), 2))

# Data table
st.subheader("📋 Match Results")
st.dataframe(filtered_df)

# Goals per match chart
st.subheader("📊 Goals per Match")
st.bar_chart((filtered_df["home_score"] + filtered_df["away_score"]).reset_index(drop=True))

# Goals trend by edition
st.subheader("📈 Average Goals per Match by Edition")
df["total_goals"] = df["home_score"] + df["away_score"]
avg_goals = df.groupby("edition")["total_goals"].mean().reset_index()

chart = alt.Chart(avg_goals).mark_line(point=True).encode(
    x=alt.X("edition:O", title="World Cup Edition"),
    y=alt.Y("total_goals:Q", title="Avg Goals per Match"),
    tooltip=["edition", "total_goals"]
).properties(title="Goal Scoring Trends by Edition")

st.altair_chart(chart, use_container_width=True)

# --- Additional Insights ---

# 1. Top scoring teams by edition
st.subheader("🥅 Top Scoring Teams (Selected Edition)")
team_goals = pd.concat([
    filtered_df[["home_team", "home_score"]].rename(columns={"home_team": "team", "home_score": "goals"}),
    filtered_df[["away_team", "away_score"]].rename(columns={"away_team": "team", "away_score": "goals"})
])
top_teams = team_goals.groupby("team")["goals"].sum().sort_values(ascending=False).reset_index()
bar = alt.Chart(top_teams).mark_bar().encode(
    x=alt.X("goals:Q", title="Goals Scored"),
    y=alt.Y("team:N", sort="-x", title="Team"),
    tooltip=["team", "goals"]
).properties(height=300)
st.altair_chart(bar, use_container_width=True)

# 2. Home vs Away Wins by Year
st.subheader("🏠 Home vs 🛫 Away Wins by Edition")
def win_type(row):
    if row["winner_team"] == row["home_team"]:
        return "Home Win"
    elif row["winner_team"] == row["away_team"]:
        return "Away Win"
    else:
        return "Draw"
df["win_type"] = df.apply(win_type, axis=1)
win_counts = df.groupby(["edition", "win_type"]).size().reset_index(name="count")
win_chart = alt.Chart(win_counts).mark_bar().encode(
    x=alt.X("edition:O", title="Edition"),
    y=alt.Y("count:Q", title="Number of Matches"),
    color=alt.Color("win_type:N", title="Result Type"),
    tooltip=["edition", "win_type", "count"]
).properties(height=350)
st.altair_chart(win_chart, use_container_width=True)

# --- Mandatory Questions Section ---
st.markdown("## 📝 Project Questions & Analysis")

# Question 1: Top Scorers
with st.expander("Question 1: Who are the top scorers?"):
    st.markdown("""
    ### Top Scorers Analysis
    
    Let's analyze the top scoring teams in World Cup history and in the selected edition.
    """)
    
    # Calculate all-time top scoring teams
    all_time_goals = pd.concat([
        df[["home_team", "home_score"]].rename(columns={"home_team": "team", "home_score": "goals"}),
        df[["away_team", "away_score"]].rename(columns={"away_team": "team", "away_score": "goals"})
    ])
    all_time_top_teams = all_time_goals.groupby("team")["goals"].sum().sort_values(ascending=False).head(10).reset_index()
    
    # All-time top scorers chart
    st.subheader("🏆 All-Time Top Scoring Teams")
    all_time_chart = alt.Chart(all_time_top_teams).mark_bar().encode(
        x=alt.X("goals:Q", title="Total Goals Scored"),
        y=alt.Y("team:N", sort="-x", title="Team"),
        tooltip=["team", "goals"]
    ).properties(
        height=400,
        title="Top 10 Teams by Total Goals in World Cup History"
    )
    st.altair_chart(all_time_chart, use_container_width=True)
    
    # Selected edition top scorers (reusing existing chart)
    st.subheader(f"🥅 Top Scoring Teams in {selected_edition}")
    team_goals = pd.concat([
        filtered_df[["home_team", "home_score"]].rename(columns={"home_team": "team", "home_score": "goals"}),
        filtered_df[["away_team", "away_score"]].rename(columns={"away_team": "team", "away_score": "goals"})
    ])
    top_teams = team_goals.groupby("team")["goals"].sum().sort_values(ascending=False).reset_index()
    bar = alt.Chart(top_teams).mark_bar().encode(
        x=alt.X("goals:Q", title="Goals Scored"),
        y=alt.Y("team:N", sort="-x", title="Team"),
        tooltip=["team", "goals"]
    ).properties(height=300)
    st.altair_chart(bar, use_container_width=True)
    
    # Commentary and insights
    st.markdown("""
    #### Key Insights:
    
    1. **Historical Dominance**: The all-time top scorers list shows which teams have consistently performed well across multiple World Cup editions.
    
    2. **Current Form**: The selected edition's top scorers reveal which teams are currently in form and most effective in front of goal.
    
    3. **Scoring Trends**: 
       - Teams that appear in both charts demonstrate consistent scoring ability
       - New teams in the selected edition chart show emerging offensive power
       - The gap between top and bottom teams indicates competitive balance
    
    4. **Notable Statistics**:
       - Total goals scored by the all-time leader: {all_time_leader_goals}
       - Average goals per team in the selected edition: {selected_avg_goals}
       - Number of teams that scored in the selected edition: {num_teams}
    """.format(
        all_time_leader_goals=all_time_top_teams.iloc[0]['goals'],
        selected_avg_goals=round(top_teams['goals'].mean(), 2),
        num_teams=len(top_teams)
    ))

# Question 2: Home Advantage
with st.expander("Question 2: Is there a home advantage?"):
    st.markdown("""
    ### Home Advantage Analysis
    
    Let's analyze whether playing at home provides a competitive advantage in the World Cup.
    """)
    
    # Calculate match outcomes by edition
    def get_outcome(row):
        if row["winner_team"] == row["home_team"]:
            return "Home Win"
        elif row["winner_team"] == row["away_team"]:
            return "Away Win"
        else:
            return "Draw"
    
    df["outcome"] = df.apply(get_outcome, axis=1)
    outcomes_by_edition = df.groupby(["edition", "outcome"]).size().reset_index(name="count")
    
    # Create stacked bar chart
    st.subheader("🏠 Match Outcomes by Edition")
    outcome_chart = alt.Chart(outcomes_by_edition).mark_bar().encode(
        x=alt.X("edition:O", title="World Cup Edition"),
        y=alt.Y("count:Q", title="Number of Matches"),
        color=alt.Color("outcome:N", title="Match Outcome"),
        tooltip=["edition", "outcome", "count"]
    ).properties(
        height=400,
        title="Distribution of Match Outcomes Across World Cup Editions"
    )
    st.altair_chart(outcome_chart, use_container_width=True)
    
    # Calculate overall win percentages
    total_matches = len(df)
    home_wins = len(df[df["outcome"] == "Home Win"])
    away_wins = len(df[df["outcome"] == "Away Win"])
    draws = len(df[df["outcome"] == "Draw"])
    
    # Display percentages as metrics
    st.subheader("📊 Overall Match Outcome Distribution")
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Home Wins",
        f"{home_wins} matches",
        f"{round(home_wins/total_matches*100, 1)}%"
    )
    col2.metric(
        "Away Wins",
        f"{away_wins} matches",
        f"{round(away_wins/total_matches*100, 1)}%"
    )
    col3.metric(
        "Draws",
        f"{draws} matches",
        f"{round(draws/total_matches*100, 1)}%"
    )
    
    # Commentary
    st.markdown("""
    #### Key Insights:
    
    1. **Home Advantage**: The data shows whether home teams have a significant advantage in World Cup matches.
    
    2. **Historical Trends**: 
       - The stacked bar chart reveals how match outcomes have evolved across different editions
       - Notable patterns in home vs away performance can be identified
    
    3. **Competitive Balance**:
       - The distribution of outcomes indicates the level of competitive balance
       - Higher percentages of away wins might suggest strong visiting teams
       - Draw rates can indicate defensive strategies or evenly matched teams
    
    4. **Notable Observations**:
       - Home teams win approximately {home_win_pct}% of matches
       - Away teams win approximately {away_win_pct}% of matches
       - {draw_pct}% of matches end in draws
    """.format(
        home_win_pct=round(home_wins/total_matches*100, 1),
        away_win_pct=round(away_wins/total_matches*100, 1),
        draw_pct=round(draws/total_matches*100, 1)
    ))

# Question 3: Tournament Evolution
with st.expander("Question 3: How has the tournament evolved?"):
    st.markdown("""
    ### Tournament Evolution Analysis
    
    Let's examine how the World Cup has changed over time, looking at scoring patterns, number of matches, and team participation.
    """)
    
    # Reuse the average goals per edition chart
    st.subheader("📈 Average Goals per Match by Edition")
    df["total_goals"] = df["home_score"] + df["away_score"]
    avg_goals = df.groupby("edition")["total_goals"].mean().reset_index()
    
    goals_chart = alt.Chart(avg_goals).mark_line(point=True).encode(
        x=alt.X("edition:O", title="World Cup Edition"),
        y=alt.Y("total_goals:Q", title="Avg Goals per Match"),
        tooltip=["edition", "total_goals"]
    ).properties(
        height=300,
        title="Goal Scoring Trends by Edition"
    )
    st.altair_chart(goals_chart, use_container_width=True)
    
    # Number of matches per edition
    st.subheader("🎯 Number of Matches by Edition")
    matches_per_edition = df.groupby("edition").size().reset_index(name="match_count")
    
    matches_chart = alt.Chart(matches_per_edition).mark_bar().encode(
        x=alt.X("edition:O", title="World Cup Edition"),
        y=alt.Y("match_count:Q", title="Number of Matches"),
        tooltip=["edition", "match_count"]
    ).properties(
        height=300,
        title="Tournament Size Evolution"
    )
    st.altair_chart(matches_chart, use_container_width=True)
    
    # Unique teams per edition
    st.subheader("🌍 Unique Teams per Edition")
    unique_teams = df.groupby("edition")[["home_team", "away_team"]].agg(lambda x: set(x)).reset_index()
    unique_teams["team_count"] = unique_teams.apply(lambda row: len(row["home_team"].union(row["away_team"])), axis=1)
    
    teams_chart = alt.Chart(unique_teams).mark_line(point=True).encode(
        x=alt.X("edition:O", title="World Cup Edition"),
        y=alt.Y("team_count:Q", title="Number of Teams"),
        tooltip=["edition", "team_count"]
    ).properties(
        height=300,
        title="Team Participation Growth"
    )
    st.altair_chart(teams_chart, use_container_width=True)
    
    # Summary statistics
    st.subheader("📊 Tournament Growth Statistics")
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Matches Growth",
        f"{matches_per_edition['match_count'].iloc[-1] - matches_per_edition['match_count'].iloc[0]}",
        f"From {matches_per_edition['match_count'].iloc[0]} to {matches_per_edition['match_count'].iloc[-1]} matches"
    )
    col2.metric(
        "Teams Growth",
        f"{unique_teams['team_count'].iloc[-1] - unique_teams['team_count'].iloc[0]}",
        f"From {unique_teams['team_count'].iloc[0]} to {unique_teams['team_count'].iloc[-1]} teams"
    )
    col3.metric(
        "Scoring Change",
        f"{round(avg_goals['total_goals'].iloc[-1] - avg_goals['total_goals'].iloc[0], 2)}",
        f"From {round(avg_goals['total_goals'].iloc[0], 2)} to {round(avg_goals['total_goals'].iloc[-1], 2)} goals/match"
    )
    
    # Commentary
    st.markdown("""
    #### Key Insights:
    
    1. **Tournament Expansion**:
       - The number of matches has increased significantly over time
       - More teams are participating in each edition
       - The tournament has become more inclusive and global
    
    2. **Scoring Patterns**:
       - Average goals per match has fluctuated across editions
       - Recent trends show {scoring_trend}
       - The {highest_scoring} edition had the highest average goals per match
    
    3. **Global Growth**:
       - The World Cup has expanded from {first_teams} to {latest_teams} teams
       - This growth reflects football's increasing global popularity
       - More nations are now competitive at the highest level
    
    4. **Notable Changes**:
       - Tournament format changes have impacted match counts
       - Regional representation has become more diverse
       - The competition has become more competitive and unpredictable
    """.format(
        scoring_trend="an increasing trend" if avg_goals['total_goals'].iloc[-1] > avg_goals['total_goals'].iloc[0] else "a decreasing trend",
        highest_scoring=avg_goals.loc[avg_goals['total_goals'].idxmax(), 'edition'],
        first_teams=unique_teams['team_count'].iloc[0],
        latest_teams=unique_teams['team_count'].iloc[-1]
    ))

# Question 4: Performance Metrics
with st.expander("Question 4: What are the key performance metrics?"):
    st.markdown("""
    ### Performance Metrics Analysis
    
    Let's examine the key performance indicators across all World Cup matches.
    """)
    
    # Calculate global KPIs
    total_matches = len(df)
    avg_goals = (df["home_score"] + df["away_score"]).mean()
    avg_attendance = df["attendance_number"].mean()
    
    # Display KPIs in a single row
    st.subheader("📊 Global Performance Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Total Matches",
        f"{total_matches:,}",
        "All World Cup matches"
    )
    col2.metric(
        "Average Goals",
        f"{avg_goals:.2f}",
        "Goals per match"
    )
    col3.metric(
        "Average Attendance",
        f"{avg_attendance:,.0f}",
        "Spectators per match"
    )
    
    # Attendance trend over time
    st.subheader("👥 Attendance Trends")
    avg_attendance_trend = df.groupby("edition")["attendance_number"].mean().reset_index()
    
    attendance_chart = alt.Chart(avg_attendance_trend).mark_line(point=True).encode(
        x=alt.X("edition:O", title="World Cup Edition"),
        y=alt.Y("attendance_number:Q", title="Average Attendance"),
        tooltip=["edition", "attendance_number"]
    ).properties(
        height=300,
        title="Average Match Attendance by Edition"
    )
    st.altair_chart(attendance_chart, use_container_width=True)
    
    # Match outcomes distribution
    st.subheader("🎯 Match Outcomes Distribution")
    outcomes = df["win_type"].value_counts().reset_index()
    outcomes.columns = ["outcome", "count"]
    
    # Create pie chart for outcomes
    pie_chart = alt.Chart(outcomes).mark_arc().encode(
        theta=alt.Theta(field="count", type="quantitative"),
        color=alt.Color(field="outcome", type="nominal"),
        tooltip=["outcome", "count"]
    ).properties(
        height=300,
        title="Distribution of Match Outcomes"
    )
    st.altair_chart(pie_chart, use_container_width=True)
    
    # Calculate percentages for the summary
    total = outcomes["count"].sum()
    home_win_pct = (outcomes[outcomes["outcome"] == "Home Win"]["count"].iloc[0] / total * 100) if "Home Win" in outcomes["outcome"].values else 0
    away_win_pct = (outcomes[outcomes["outcome"] == "Away Win"]["count"].iloc[0] / total * 100) if "Away Win" in outcomes["outcome"].values else 0
    draw_pct = (outcomes[outcomes["outcome"] == "Draw"]["count"].iloc[0] / total * 100) if "Draw" in outcomes["outcome"].values else 0
    
    # Commentary
    st.markdown("""
    #### Key Insights:
    
    1. **Match Volume and Scoring**:
       - The World Cup has hosted {total_matches:,} matches to date
       - Average of {avg_goals:.2f} goals per match shows {scoring_context}
       - Consistent scoring patterns across editions
    
    2. **Attendance Growth**:
       - Average attendance of {avg_attendance:,.0f} spectators per match
       - {attendance_trend} trend in spectator numbers
       - Peak attendance in {peak_attendance_year} with {peak_attendance:,.0f} spectators
    
    3. **Match Outcomes**:
       - {home_win_pct:.1f}% of matches end in home wins
       - {away_win_pct:.1f}% of matches end in away wins
       - {draw_pct:.1f}% of matches end in draws
       - {outcome_insight}
    """.format(
        total_matches=total_matches,
        avg_goals=avg_goals,
        scoring_context="high-scoring matches" if avg_goals > 2.5 else "moderate scoring",
        avg_attendance=avg_attendance,
        attendance_trend="increasing" if avg_attendance_trend["attendance_number"].iloc[-1] > avg_attendance_trend["attendance_number"].iloc[0] else "stable",
        peak_attendance_year=avg_attendance_trend.loc[avg_attendance_trend["attendance_number"].idxmax(), "edition"],
        peak_attendance=avg_attendance_trend["attendance_number"].max(),
        home_win_pct=home_win_pct,
        away_win_pct=away_win_pct,
        draw_pct=draw_pct,
        outcome_insight="balanced competition with significant away team success" if away_win_pct > 30 else "home advantage is evident"
    ))

# Question 5: Data Quality
with st.expander("Question 5: How reliable is our data?"):
    st.markdown("""
    ### Data Quality Assessment
    
    Let's examine the reliability and completeness of our World Cup match data.
    """)
    
    # Missing values analysis
    st.subheader("🔍 Missing Values Overview")
    missing_data = df.isnull().sum().reset_index()
    missing_data.columns = ["Column", "Missing Count"]
    missing_data["Percentage"] = (missing_data["Missing Count"] / len(df) * 100).round(2)
    st.dataframe(missing_data)
    
    # Source distribution
    st.subheader("📊 Data Source Distribution")
    source_counts = df["source"].value_counts().reset_index()
    source_counts.columns = ["Source", "Count"]
    source_counts["Percentage"] = (source_counts["Count"] / len(df) * 100).round(2)
    
    # Create bar chart for source distribution
    source_chart = alt.Chart(source_counts).mark_bar().encode(
        x=alt.X("Source:N", title="Data Source"),
        y=alt.Y("Count:Q", title="Number of Records"),
        tooltip=["Source", "Count", "Percentage"]
    ).properties(
        height=300,
        title="Distribution of Records by Source"
    )
    st.altair_chart(source_chart, use_container_width=True)
    
    # Consistency checks
    st.subheader("✅ Data Consistency Checks")
    
    # Check 1: Score validation
    invalid_scores = df[
        (df["home_score"] < 0) | 
        (df["home_score"] > 10) | 
        (df["away_score"] < 0) | 
        (df["away_score"] > 10)
    ]
    
    # Check 2: Winner validation
    invalid_winners = df[
        (df["winner_team"].isna()) & 
        (df["home_score"] != df["away_score"])
    ]
    
    # Display consistency check results
    col1, col2 = st.columns(2)
    col1.metric(
        "Invalid Scores",
        len(invalid_scores),
        "Matches with scores outside 0-10 range"
    )
    col2.metric(
        "Missing Winners",
        len(invalid_winners),
        "Matches with missing winner data"
    )
    
    # Data quality assessment
    st.markdown("""
    #### Data Quality Assessment:
    
    1. **Completeness**:
       - {missing_percentage:.1f}% of records have missing values
       - Most complete columns: {most_complete}
       - Columns needing attention: {needs_attention}
    
    2. **Source Reliability**:
       - {source_distribution}
       - Historical data covers {historical_coverage}
       - API data is {api_status}
    
    3. **Data Consistency**:
       - {score_consistency}
       - {winner_consistency}
       - Overall data quality is {quality_assessment}
    
    4. **Known Limitations**:
       - {limitations}
    """.format(
        missing_percentage=missing_data["Percentage"].mean(),
        most_complete=", ".join(missing_data[missing_data["Missing Count"] == 0]["Column"].tolist()),
        needs_attention=", ".join(missing_data[missing_data["Missing Count"] > 0]["Column"].tolist()),
        source_distribution=f"Data is split between {len(source_counts)} sources",
        historical_coverage=f"editions from {df['edition'].min()} to {df['edition'].max()}",
        api_status="up-to-date" if "API" in source_counts["Source"].values else "not available",
        score_consistency="All scores are within valid range (0-10)" if len(invalid_scores) == 0 else f"{len(invalid_scores)} matches have invalid scores",
        winner_consistency="Winner data is complete" if len(invalid_winners) == 0 else f"{len(invalid_winners)} matches have missing winner data",
        quality_assessment="good" if missing_data["Percentage"].mean() < 5 and len(invalid_scores) == 0 and len(invalid_winners) == 0 else "needs improvement",
        limitations="Some historical editions may have incomplete data, and API coverage is limited to recent matches"
    ))

# Question 6: Future Predictions
with st.expander("Question 6: What can we predict?"):
    st.markdown("""
    ### Predictive Analysis
    
    Let's explore how we can use historical data to make predictions about future World Cup matches.
    """)
    
    # Calculate team average goals
    team_goals_all = pd.concat([
        df[["home_team", "home_score"]].rename(columns={"home_team": "team", "home_score": "goals"}),
        df[["away_team", "away_score"]].rename(columns={"away_team": "team", "away_score": "goals"})
    ])
    team_avg_goals = team_goals_all.groupby("team")["goals"].agg(["mean", "count"]).reset_index()
    team_avg_goals.columns = ["Team", "Average Goals", "Matches Played"]
    team_avg_goals = team_avg_goals.sort_values("Average Goals", ascending=False)
    
    # Display top scoring teams
    st.subheader("🎯 Expected Scoring Power by Team")
    st.dataframe(
        team_avg_goals.head(10).style.format({
            "Average Goals": "{:.2f}",
            "Matches Played": "{:.0f}"
        })
    )
    
    # Team comparison UI
    st.subheader("🤔 Compare Teams")
    col1, col2 = st.columns(2)
    
    with col1:
        team1 = st.selectbox(
            "Select First Team",
            sorted(team_avg_goals["Team"].unique()),
            key="team1"
        )
    
    with col2:
        team2 = st.selectbox(
            "Select Second Team",
            sorted(team_avg_goals["Team"].unique()),
            key="team2"
        )
    
    if team1 and team2:
        team1_data = team_avg_goals[team_avg_goals["Team"] == team1].iloc[0]
        team2_data = team_avg_goals[team_avg_goals["Team"] == team2].iloc[0]
        
        st.markdown("""
        #### Historical Performance Comparison
        
        Based on average goals per match:
        """)
        
        col1, col2, col3 = st.columns(3)
        col1.metric(
            team1,
            f"{team1_data['Average Goals']:.2f}",
            "goals per match"
        )
        col2.metric(
            "Difference",
            f"{abs(team1_data['Average Goals'] - team2_data['Average Goals']):.2f}",
            "goals per match"
        )
        col3.metric(
            team2,
            f"{team2_data['Average Goals']:.2f}",
            "goals per match"
        )
    
    # Future prediction possibilities
    st.markdown("""
    #### Potential Future Predictions
    
    With more data and advanced modeling, we could predict:
    
    1. **Match Outcomes**:
       - Win/loss/draw probabilities
       - Score predictions
       - Goal difference estimates
    
    2. **Team Performance**:
       - Tournament progression
       - Group stage performance
       - Knockout round success
    
    3. **Player Statistics**:
       - Individual goal scoring
       - Assists and contributions
       - Performance metrics
    
    #### Tools and Methods
    
    We could use:
    - **Machine Learning**:
      - Scikit-learn for classification and regression
      - XGBoost for ensemble methods
      - Neural networks for complex patterns
    
    - **Time Series Analysis**:
      - ARIMA models for trend analysis
      - Prophet for seasonal patterns
      - LSTM networks for sequence prediction
    
    - **Statistical Methods**:
      - Poisson distribution for goal scoring
      - Bayesian inference for probabilities
      - Monte Carlo simulations for tournament outcomes
    """)

# Question 7: Business Impact
with st.expander("Question 7: What's the business value?"):
    st.markdown("""
    ### Business Impact Analysis
    
    This World Cup data analysis provides valuable insights for various stakeholders in the sports industry.
    """)
    
    # Fan Engagement
    st.markdown("""
    #### 🎯 Fan Engagement
    
    **Interactive Experience**
    - Real-time dashboards during matches enhance viewer engagement
    - Historical data comparisons create compelling narratives
    - Personalized team and player statistics increase fan interaction
    
    **Digital Integration**
    - Mobile app features for live match statistics
    - Social media integration for sharing insights
    - Fantasy league and prediction games
    - Custom notifications for favorite teams
    """)
    
    # Sponsorship & Advertising
    st.markdown("""
    #### 💰 Sponsorship & Advertising
    
    **Targeted Marketing**
    - Peak viewing times based on match timing analysis
    - High-scoring team matches for maximum exposure
    - Regional team performance for local market targeting
    
    **Sponsorship Opportunities**
    - Team performance metrics for sponsorship valuation
    - Fan engagement data for partnership proposals
    - Historical success rates for brand alignment
    """)
    
    # Event Planning
    st.markdown("""
    #### 📅 Event Planning
    
    **Venue Optimization**
    - Attendance patterns inform stadium capacity decisions
    - Match timing analysis for optimal scheduling
    - Regional distribution of matches based on fan base
    
    **Operational Efficiency**
    - Resource allocation based on expected attendance
    - Security planning using historical patterns
    - Infrastructure requirements for different match types
    """)
    
    # Strategic Team Insights
    st.markdown("""
    #### ⚽ Strategic Team Insights
    
    **Performance Analysis**
    - Team benchmarking against historical data
    - Player performance tracking and comparison
    - Tactical pattern recognition
    
    **Development Planning**
    - Youth program development based on success patterns
    - Training focus areas identified through data
    - Recruitment strategy optimization
    """)
    
    # Revenue Opportunities
    st.markdown("""
    #### 💎 Revenue Opportunities
    
    **Data Monetization**
    - Premium analytics for professional teams
    - Custom reports for media organizations
    - API access for sports betting companies
    
    **Fan Services**
    - Subscription-based detailed statistics
    - Premium mobile app features
    - Exclusive content and insights
    """)
    
    # Closing Statement
    st.markdown("""
    #### 🎓 The Future of Sports Analytics
    
    The integration of data analytics in sports management is revolutionizing how decisions are made. From fan engagement to strategic planning, data-driven insights are becoming essential tools for success in modern sports. This World Cup analysis demonstrates how historical data, when properly analyzed and presented, can provide actionable insights for various stakeholders in the sports ecosystem.
    
    The ability to make informed decisions based on comprehensive data analysis is no longer optional—it's a competitive necessity in today's sports industry. As technology continues to evolve, the value of such analytics will only increase, making data literacy and analytical capabilities crucial for success in sports management and business.
    """)

# Question 8: Technical Implementation
with st.expander("Question 8: How was this built?"):
    st.markdown("""
    ### Technical Implementation Details
    
    A comprehensive overview of the project's architecture and implementation.
    """)
    
    # Data Flow Diagram
    st.markdown("""
    #### 📊 Data Flow Architecture
    
    ```
    [Data Sources] → [Processing] → [Storage] → [Visualization]
         ↓              ↓             ↓             ↓
    Historical CSV   Pandas        Parquet      Streamlit
    RapidAPI        Cleaning      Files        Dashboard
    ```
    """)
    
    # Data Sources & Storage
    st.markdown("""
    #### 🔧 Data Sources & Storage
    
    **Data Collection**
    - Historical match data from CSV files
    - Live match data from RapidAPI
    - Combined into a unified dataset
    
    **Storage Solution**
    - Final data stored in Parquet format
    - Optimized for columnar access
    - Efficient compression and querying
    - Location: `datalake/combined/final_matches.parquet`
    """)
    
    # Data Processing
    st.markdown("""
    #### 🧹 Data Processing Pipeline
    
    **Data Cleaning**
    - Missing value handling
    - Data type standardization
    - Date/time normalization
    - Team name standardization
    
    **Data Transformation**
    - Merged historical and live data
    - Created derived fields:
      - `total_goals`: home_score + away_score
      - `win_type`: Home Win/Away Win/Draw
    - Aggregated statistics by team and edition
    """)
    
    # Visualization Layer
    st.markdown("""
    #### 📊 Visualization & Frontend
    
    **Streamlit Dashboard**
    - Interactive edition selector
    - Real-time data filtering
    - Responsive layout design
    
    **Charting Libraries**
    - Altair for complex visualizations
    - Native Streamlit charts for simple plots
    - Interactive tooltips and hover effects
    
    **UI Components**
    - Expandable sections for detailed analysis
    - Metric cards for key statistics
    - Data tables for raw data viewing
    """)
    
    # Project Structure
    st.markdown("""
    #### 📁 Project Structure
    
    ```
    WC-DA/
    ├── datalake/
    │   ├── raw/           # Original data files
    │   ├── processed/     # Cleaned data
    │   └── combined/      # Final merged data
    ├── streamlit_app/
    │   └── app.py        # Main dashboard
    └── requirements.txt   # Dependencies
    ```
    """)
    
    # Performance Optimizations
    st.markdown("""
    #### 🚀 Performance & Optimization
    
    **Caching Strategy**
    - `@st.cache_data` for data loading
    - TTL (Time To Live) set to 0 for static data
    - Efficient memory usage
    
    **Data Updates**
    - Last-updated timestamp from file metadata
    - Automatic refresh on data changes
    - Efficient data reloading
    
    **Query Optimization**
    - Parquet file format for fast reading
    - Efficient filtering and aggregation
    - Optimized data transformations
    """)
    
    # Future Improvements
    st.markdown("""
    #### 🔮 Future Enhancements
    
    **Planned Improvements**
    - Airflow integration for automated updates
    - Real-time data streaming
    - Advanced caching strategies
    - Additional data sources
    
    **Scalability Considerations**
    - Modular code structure
    - Efficient data processing
    - Optimized memory usage
    - Easy to extend and maintain
    """)

# Airflow Integration Stub
with st.expander("🛠️ Data Pipeline (Airflow Integration Stub)"):
    st.markdown("""
    ### Airflow-Based Data Pipeline (Proposed)
    
    Although not implemented in this version, this project could benefit from Airflow for orchestration. Here's a simple DAG idea:
    
    #### 🔁 Daily Tasks
    
    **Task 1: Fetch Live Match Data**
    - Use Airflow to hit a football match API daily
    - Retrieve new match data
    - Handle API rate limits and errors
    - Store raw data in staging area
    
    **Task 2: Append & Store as Parquet**
    - Clean and transform new data
    - Merge with historical dataset
    - Update Parquet files
    - Maintain data versioning
    
    **Task 3: Invalidate Cache / Refresh Dashboard**
    - Clear Streamlit cache after updates
    - Send Slack/email notifications
    - Update last-modified timestamp
    - Trigger dashboard refresh
    
    **Task 4: Data Validation**
    - Check for missing entries
    - Identify duplicates
    - Validate data consistency
    - Flag anomalies for review
    
    #### 👁️ Monitoring
    
    **Airflow UI Features**
    - DAG execution status
    - Task success/failure tracking
    - Detailed execution logs
    - Performance metrics
    
    **Alerting System**
    - Email notifications for failures
    - Slack integration for team updates
    - Custom alert thresholds
    - Error reporting
    
    This would ensure the dashboard remains live, fresh, and reliable with minimal manual effort.
    """)