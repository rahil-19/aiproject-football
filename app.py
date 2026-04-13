import joblib
import pandas as pd
import streamlit as st
import base64

model, home_avg, away_avg = joblib.load("model.pkl")


def get_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

team_logos = {
    "Arsenal": "logos/Arsenal.png",
    "Chelsea": "logos/Chelsea.png",
    "Man City": "logos/Man City.png",
    "Man United": "logos/Man United.png",
    "Aston Villa": "logos/Aston Villa.png",
    "Bournemouth": "logos/Bournemouth.png",
    "Brentford": "logos/Brentford.png",
    "Brighton": "logos/Brighton.png",
    "Burnley": "logos/Burnley.png",
    "Crystal Palace": "logos/Crystal Palace.png",
    "Everton": "logos/Everton.png",
    "Fulham": "logos/Fulham.png",
    "Ipswich": "logos/Ipswich.png",
    "Leeds": "logos/Leeds.png",
    "Leicester": "logos/Leicester.png",
    "Liverpool": "logos/Liverpool.png",
    "Luton": "logos/Luton.png",
    "Newcastle": "logos/Newcastle.png",
    "Nott'm Forest": "logos/Nott'm Forest.png",
    "Sheffield United": "logos/Sheffield.png",
    "Southampton": "logos/Southampton.png",
    "Sunderland": "logos/Sunderland.png",
    "Tottenham": "logos/Tottenham.png",
    "West Ham": "logos/West Ham.png",
    "Wolves": "logos/Wolves.png"
}

def predict_match(home, away):
    import pandas as pd
    
    X_new = pd.DataFrame([[home_avg[home], away_avg[away]]],
                         columns=['HomeAttack', 'AwayAttack'])

    pred = model.predict(X_new)[0]
    prob = model.predict_proba(X_new)[0]

    return pred, prob

teams = list(home_avg.keys())

st.title("Premier League Predictor")

home = st.selectbox("Home Team", teams)
away = st.selectbox("Away Team", teams)

if st.button("Predict"):
    pred, prob = predict_match(home, away)

    
    prob_dict = dict(zip(model.classes_, prob))

    home_prob = float(prob_dict.get('H', 0))
    draw_prob = float(prob_dict.get('D', 0))
    away_prob = float(prob_dict.get('A', 0))
    
    st.subheader("Prediction")

    if pred == 'H':
        st.success(f"{home} is likely to WIN")
    elif pred == 'A':
        st.success(f"{away} is likely to WIN")
    else:
        st.warning("Match likely to be a DRAW")

    st.subheader("Win Probabilities")

    home_pct = home_prob * 100
    draw_pct = draw_prob * 100
    away_pct = away_prob * 100

    home_logo = team_logos.get(home, "")
    away_logo = team_logos.get(away, "")

    home_logo_base64 = get_base64(home_logo)
    away_logo_base64 = get_base64(away_logo)

    bar_html = f"""
                <div style="display: flex; align-items: center; justify-content: center; gap: 20px; width: 100%; padding: 20px 0;">

                    <!-- Home Team -->
                    <div style="display: flex; flex-direction: column; align-items: center; width: 120px;">
                        <!--<img src="{home_logo}" width="60" style="margin-bottom: 5px;">-->
                        <img src="data:image/png;base64,{home_logo_base64}" width="50">
                        <span style="font-weight: bold; color: white; font-family: verdana;">{home}</span>
                    </div>

                    <!-- Probability Bar -->
                    <div style="flex-grow: 1; max-width: 800px; background-color: #e0e0e0; border-radius: 15px; overflow: hidden; font-family: verdana;">
                        <div style="display: flex; height: 40px; font-size: 14px; font-weight: bold;">
                            
                            <div style="width: {home_pct}%; background-color: #28a745; color: white; display: flex; align-items: center; justify-content: center;">
                                {home_pct:.1f}%
                            </div>
                            
                            <div style="width: {draw_pct}%; background-color: #ffc107; color: black; display: flex; align-items: center; justify-content: center;">
                                {draw_pct:.1f}%
                            </div>
                            
                            <div style="width: {away_pct}%; background-color: #007bff; color: white; display: flex; align-items: center; justify-content: center;">
                                {away_pct:.1f}%
                            </div>

                        </div>
                    </div>

                    <!-- Away Team -->
                    <div style="display: flex; flex-direction: column; align-items: center; width: 120px;">
                        <!--<img src="{away_logo}" width="60" style="margin-bottom: 5px;">-->
                        <img src="data:image/png;base64,{away_logo_base64}" width="50">
                        <span style="font-weight: bold; color: white; font-family: verdana;">{away}</span>
                    </div>

                </div>
                """
    
    st.components.v1.html(bar_html, height=200)


