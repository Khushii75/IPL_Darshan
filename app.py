import streamlit as st
import pandas as pd
import joblib

# =_=_=_=_=_=_=_=_=_=_=_=_=_=
# PAGE CONFIG

st.set_page_config(page_title="IPL Darshan", layout="wide")
#changess
st.markdown("""
<style>

/* 🔥 Main Background */
.stApp {
    background: linear-gradient(135deg, #000000, #0f0f0f);
    color: white;
}

/* 🏆 Title Styling */
h1 {
    color: #FFD700;
    text-shadow: 0 0 10px rgba(255,215,0,0.6);
}

/* 📦 Input Fields */
div[data-baseweb="select"] > div {
    background-color: #1a1a1a !important;
    border: 1px solid #FFD700 !important;
    border-radius: 10px !important;
    color: white !important;
}

/* 🔘 Buttons */
.stButton > button {
    background: linear-gradient(90deg, #FFD700, #C9A200);
    color: black;
    font-weight: bold;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
    transition: 0.3s ease;
}

/* ✨ Button Hover */
.stButton > button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 15px #FFD700;
}

/* 📢 Info Box */
.stAlert {
    background-color: #1a1a1a !important;
    border-left: 5px solid #FFD700 !important;
    color: white !important;
}

/* ✅ Success Box */
.stSuccess {
    background-color: #1a1a1a !important;
    border-left: 5px solid #FFD700 !important;
    color: #FFD700 !important;
    font-weight: bold;
}

/* 🎯 Radio Buttons */
.stRadio > div {
    color: #FFD700;
}

/* 🧾 Subheaders */
h2, h3 {
    color: #FFD700;
}

/* ✨ Glow Divider */
hr {
    border: 1px solid #FFD700;
}

/* 📊 Number Inputs */
input {
    background-color: #1a1a1a !important;
    color: white !important;
    border: 1px solid #FFD700 !important;
    border-radius: 8px !important;
}

</style>
""", unsafe_allow_html=True)
#till here

st.title("🏆 IPL Match Prediction System")

# =_=_=_=_=_=_=_=_=_=_=_=_=_=
# LOAD DATA

matches = pd.read_csv("dataset/matches_new.csv")

matches['date'] = pd.to_datetime(
    matches['date'],
    errors='coerce'
)
matches = matches.dropna(subset=['date'])
matches = matches.sort_values("date")

# =_=_=_=_=_=_=_=_=_=_=_=_=_=
# LOAD MODELS

prematch_model = joblib.load("models/prematch_model.pkl")
features = joblib.load("models/features.pkl")
first_innings = joblib.load("models/first_innings.pkl") 
second_innings = joblib.load("models/second_innings.pkl")
second_features = joblib.load("models/second_features.pkl")

# =_=_=_=_=_=_=_=_=_=_=_=_=_=
# FEATURE ENGINEERING FUNCTION

def generate_features(team1, team2, venue):

    matches_played = pd.concat(
        [matches['team1'], matches['team2']]
    ).value_counts()

    wins = matches['winner'].value_counts()

    win_rate = (wins / matches_played).fillna(0.5)

    win_rate_diff = (
        win_rate.get(team1, 0.5)
        - win_rate.get(team2, 0.5)
    )

    # Head-to-Head
    h2h = matches[
        ((matches['team1'] == team1) &
         (matches['team2'] == team2)) |
        ((matches['team1'] == team2) &
         (matches['team2'] == team1))
    ]

    h2h_rate = (
        (h2h['winner'] == team1).sum()
        - (h2h['winner'] == team2).sum()
    )

    # Recent Form
    form1 = (
        matches[(matches['team1'] == team1) |
                (matches['team2'] == team1)]
        .tail(5)['winner'].eq(team1).sum()
    )

    form2 = (
        matches[(matches['team1'] == team2) |
                (matches['team2'] == team2)]
        .tail(5)['winner'].eq(team2).sum()
    )

    form_rate = form1 - form2

    # Venue Performance
    venue_matches = matches[matches['venue'] == venue]

    def venue_wr(team):
        total = venue_matches[
            (venue_matches['team1'] == team) |
            (venue_matches['team2'] == team)
        ].shape[0]

        wins = venue_matches[
            venue_matches['winner'] == team
        ].shape[0]

        return wins / total if total > 0 else 0.5

    venue_rate = venue_wr(team1) - venue_wr(team2)

    return win_rate_diff, form_rate, h2h_rate, venue_rate


# =_=_=_=_=_=_=_=_=_=_=_=_=_=
# USER INPUT

teams = sorted(matches['team1'].dropna().unique())
venues = sorted(matches['venue'].dropna().unique())

col1, col2 = st.columns(2)

with col1:
    team1 = st.selectbox("Team 1", teams)

with col2:
    team2 = st.selectbox(
        "Team 2",
        [t for t in teams if t != team1]
    )

toss_winner = st.selectbox("Toss Winner", [team1, team2])
venue = st.selectbox("Venue", venues)
toss_decision = st.radio("Toss Decision", ["bat", "field"])


# =_=_=_=_=_=_=_=_=_=_=_=_=_=
# DECIDE MATCH ORDER

if toss_decision == "bat":
    first_batting = toss_winner
    first_bowling = team1 if toss_winner == team2 else team2
else:
    first_bowling = toss_winner
    first_batting = team1 if toss_winner == team2 else team2

st.info(f"🏏 {first_batting} will bat first")


# =_=_=_=_=_=_=_=_=_=_=_=_=_=
# PREDICTION

if st.button("Predict Match"):

    win_rate_diff, form_rate, h2h_rate, venue_rate = generate_features(
        team1, team2, venue
    )

    # Raw input
    input_df = pd.DataFrame({
        'team1': [team1],
        'team2': [team2],
        'toss_winner': [toss_winner],
        'toss_decision': [toss_decision],
        'venue': [venue],
        'win_rate_diff': [win_rate_diff],
        'form_diff': [form_rate],
        'h2h_diff': [h2h_rate],
        'venue_diff': [venue_rate],
    })


    # APPLY SAME ENCODING AS TRAINING

    input_df = pd.get_dummies(input_df)


    # ADD MISSING COLUMNS

    for col in features:
        if col not in input_df.columns:
            input_df[col] = 0

    # Keep only trained features
    input_df = input_df[features]


    # PREDICTION

    prob = prematch_model.predict_proba(input_df)[0]

    st.success(f"{team1} Win Probability: {prob[1]*100:.2f}%")
    st.success(f"{team2} Win Probability: {prob[0]*100:.2f}%")
    st.session_state["match_ready"] = True




# =_=_=_=_=_=_=_=_=_=_=_=_=_=
# INNINGS SELECTION

if st.session_state.get("match_ready"):

    st.header("Choose Innings")

    c1,c2=st.columns(2)

    if c1.button("1st Innings"):
        st.session_state.innings=1

    if c2.button("2nd Innings"):
        st.session_state.innings=2



# =_=_=_=_=_=_=_=_=_=_=_=_=_=
# FIRST INNINGS PREDICTION

if st.session_state.get("innings") == 1:

    st.subheader("🏏 1st Innings Score Prediction")

    st.write(f"Batting Team: {first_batting}")
    st.write(f"Bowling Team: {first_bowling}")


    col1, col2 = st.columns(2)

    with col1:
        current_score = st.number_input(
            "Current Score", 0, 300, step=1
        )

        wickets = st.number_input(
            "Wickets Fallen", 0, 10, step=1
        )

    with col2:
        balls_played = st.number_input(
            "Balls Played", 1, 120, step=1
        )

    # Feature Engineering

    ball_left = 120 - balls_played

    current_rr = (
        current_score * 6 / balls_played
        if balls_played > 0 else 0
    )

    # Prediction

    if st.button("Predict Final Score"):

        X = pd.DataFrame({
            'current_score': [current_score],
            'total_wicket': [wickets],
            'ball_left': [ball_left],
            'current_rr': [current_rr]
        })

        prediction = first_innings.predict(X)[0]

        lower = int(prediction - 5)
        upper = int(prediction + 5)

        st.success(f"🏏 Predicted Final Score: {lower} - {upper}")



# =_=_=_=_=_=_=_=_=_=_=_=_=_=
# SECOND INNINGS LIVE PREDICTION

if st.session_state.get("innings")==2:

    st.subheader("2nd Innings Live Predictor")

    # Teams for 2nd innings
    batting_team = first_bowling
    bowling_team = first_batting

    st.write(f"Batting Team: {batting_team}")
    st.write(f"Bowling Team: {bowling_team}")

    target=st.number_input("Target",1,300)
    runs=st.number_input("Current Runs",0,300)
    wickets=st.number_input("Wickets Fallen",0,10)
    balls=st.number_input("Balls Bowled",1,120)

    if st.button("Predict Winning Probability"):

        balls_left=120-balls
        runs_left=target-runs

        crr=(runs*6)/balls if balls>0 else 0
        rrr=(runs_left*6)/balls_left if balls_left>0 else 0

        X=pd.DataFrame({
            'batting_team':[batting_team],
            'bowling_team':[bowling_team],
            'venue':[venue],
            'runs_left':[runs_left],
            'balls_left':[balls_left],
            'wickets_left':[10-wickets],
            'target':[target],
            'crr':[crr],
            'rrr':[rrr]
        })

        X=pd.get_dummies(X)

        for col in second_features:
            if col not in X.columns:
                X[col]=0

        X=X[second_features]

        prob=second_innings.predict_proba(X)[0]

        st.success(f"{batting_team} Win % : {prob[1]*100:.2f}")
        st.success(f"{bowling_team} Win % : {prob[0]*100:.2f}")

