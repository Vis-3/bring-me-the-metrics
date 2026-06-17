"""
Metal Music Intelligence Pipeline - Portfolio Visualizations
Four interactive Plotly charts saved as HTML files.

    uv run python -m visualizations.charts
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

OUT = "visualizations/output"
os.makedirs(OUT, exist_ok=True)

SUBGENRE_COLORS = {
    "metalcore":            "#E63946",
    "deathcore":            "#1D3557",
    "melodic metalcore":    "#FF6B6B",
    "progressive metal":    "#2A9D8F",
    "symphonic metal":      "#8338EC",
    "djent":                "#FB8500",
    "nu-metal":             "#023E8A",
    "black metal":          "#212529",
    "melodic death metal":  "#457B9D",
}

EXCLUDE_ARTISTS = {
    "justin bieber", "lady gaga", "taylor swift", "rihanna",
    "beyonce", "katy perry", "ariana grande", "drake",
    "kanye west", "eminem",
}


def cast_numeric(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def drop_non_metal(df):
    if "artist_name" in df.columns:
        mask = ~df["artist_name"].str.lower().isin(EXCLUDE_ARTISTS)
        df = df[mask].copy()
    return df


def load_data():
    legacy   = pd.read_csv("data/mart_album_legacy.csv")
    health   = pd.read_csv("data/mart_subgenre_health.csv")
    features = pd.read_csv("data/mart_artist_features.csv")
    preds    = pd.read_csv("breakout_predictions.csv")

    legacy   = cast_numeric(legacy,   ["listeners", "formed_year",
                                        "years_since_last_release", "play_count",
                                        "total_albums", "last_album_year"])
    health   = cast_numeric(health,   ["total_listeners", "avg_listeners",
                                        "median_listeners", "max_listeners",
                                        "breakout_pct", "golden_era_decade",
                                        "total_artists", "breakout_artists",
                                        "country_count"])
    features = cast_numeric(features, ["current_listeners", "formed_year",
                                        "plays_per_listener", "band_age_years",
                                        "total_albums", "is_breakout"])
    preds    = cast_numeric(preds,    ["breakout_probability", "current_listeners"])

    legacy   = drop_non_metal(legacy)
    features = drop_non_metal(features)
    preds    = drop_non_metal(preds)

    return legacy, health, features, preds


# Chart 1: Golden Era Scatter
def chart_golden_era(features: pd.DataFrame) -> None:
    df = features.dropna(subset=["formed_year", "current_listeners"]).copy()
    df = df[df["current_listeners"] > 50_000]
    df["size"] = df["plays_per_listener"].clip(0, 200).fillna(20)

    fig = go.Figure()

    for subgenre, color in SUBGENRE_COLORS.items():
        sub = df[df["subgenre"] == subgenre]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["formed_year"],
            y=sub["current_listeners"],
            mode="markers",
            name=subgenre.title(),
            marker=dict(
                color=color,
                size=sub["size"].clip(6, 28),
                opacity=0.75,
                line=dict(width=0.5, color="white"),
            ),
            text=sub["artist_name"],
            customdata=sub[["subgenre", "plays_per_listener", "total_albums"]],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Subgenre: %{customdata[0]}<br>"
                "Listeners: %{y:,.0f}<br>"
                "Formed: %{x}<br>"
                "Plays/Listener: %{customdata[1]:.1f}<br>"
                "Albums: %{customdata[2]:.0f}"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=dict(
            text="The Golden Era - When Were Metal's Biggest Bands Formed?",
            font=dict(size=22, color="#1a1a2e"),
            x=0.5,
        ),
        xaxis=dict(title="Year Formed", showgrid=True,
                   gridcolor="#f0f0f0", range=[1975, 2025]),
        yaxis=dict(title="Current Listeners", type="log",
                   showgrid=True, gridcolor="#f0f0f0",
                   tickformat=",.0s",
                   range=[np.log10(50_000), np.log10(df["current_listeners"].max() * 1.5)]),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(title="Subgenre", bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#e0e0e0", borderwidth=1),
        font=dict(family="Inter, Arial, sans-serif"),
        height=600,
        annotations=[dict(
            text="Dot size = plays per listener (engagement depth)",
            x=0.01, y=0.01, xref="paper", yref="paper",
            showarrow=False, font=dict(size=11, color="#888"),
        )],
    )

    fig.write_html(f"{OUT}/golden_era.html")
    print("golden_era.html")


# Chart 2: Engagement vs Scale
def chart_engagement_vs_scale(features: pd.DataFrame) -> None:
    df = features.dropna(subset=["plays_per_listener", "current_listeners"]).copy()
    df = df[
        (df["current_listeners"] > 50_000) &
        (df["plays_per_listener"] > 0) &
        (df["plays_per_listener"] < 500)
    ]

    med_listeners  = df["current_listeners"].median()
    med_engagement = df["plays_per_listener"].median()
    max_l = df["current_listeners"].max() * 1.3
    max_e = df["plays_per_listener"].max() * 1.1

    def quadrant(row):
        hi_eng = row["plays_per_listener"] >= med_engagement
        hi_lis = row["current_listeners"]  >= med_listeners
        if hi_eng and hi_lis:
            return "Giants"
        if hi_eng and not hi_lis:
            return "Cult Heroes"
        if not hi_eng and hi_lis:
            return "Mainstream"
        return "Struggling"

    df["quadrant"] = df.apply(quadrant, axis=1)

    fig = go.Figure()

    # Shaded quadrant backgrounds
    for x0, x1, y0, y1, color in [
        (med_engagement, max_e,          med_listeners, max_l,          "rgba(42,157,143,0.07)"),
        (0,              med_engagement,  med_listeners, max_l,          "rgba(251,133,0,0.07)"),
        (med_engagement, max_e,          50_000,        med_listeners,  "rgba(230,57,70,0.07)"),
        (0,              med_engagement,  50_000,        med_listeners,  "rgba(173,181,189,0.07)"),
    ]:
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                      fillcolor=color, line_width=0)

    for subgenre, color in SUBGENRE_COLORS.items():
        sub = df[df["subgenre"] == subgenre]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["plays_per_listener"],
            y=sub["current_listeners"],
            mode="markers",
            name=subgenre.title(),
            marker=dict(color=color, size=7, opacity=0.72,
                        line=dict(width=0.5, color="white")),
            text=sub["artist_name"],
            customdata=sub[["subgenre", "band_age_years", "quadrant"]],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Subgenre: %{customdata[0]}<br>"
                "Listeners: %{y:,.0f}<br>"
                "Plays/Listener: %{x:.1f}<br>"
                "Band age: %{customdata[1]:.0f} yrs<br>"
                "Quadrant: %{customdata[2]}"
                "<extra></extra>"
            ),
        ))

    fig.add_vline(x=med_engagement, line_dash="dash", line_color="#ccc", line_width=1.5)
    fig.add_hline(y=med_listeners,  line_dash="dash", line_color="#ccc", line_width=1.5)

    for label, x, y, color in [
        ("Giants",      med_engagement * 1.8, max_l * 0.75,          "#2A9D8F"),
        ("Cult Heroes", med_engagement * 0.3, max_l * 0.75,          "#E63946"),
        ("Mainstream",  med_engagement * 0.3, med_listeners * 0.55,  "#FB8500"),
        ("Struggling",  med_engagement * 1.8, med_listeners * 0.55,  "#ADB5BD"),
    ]:
        fig.add_annotation(x=x, y=y, text=f"<b>{label}</b>",
                            showarrow=False, font=dict(size=13, color=color),
                            opacity=0.7)

    fig.update_layout(
        title=dict(
            text="Engagement vs Scale - Loyal Fans vs Casual Reach",
            font=dict(size=22, color="#1a1a2e"),
            x=0.5,
        ),
        xaxis=dict(title="Plays per Listener (Engagement Depth)",
                   showgrid=True, gridcolor="#f0f0f0"),
        yaxis=dict(title="Current Listeners", type="log",
                   showgrid=True, gridcolor="#f0f0f0",
                   tickformat=",.0s",
                   range=[np.log10(50_000), np.log10(max_l)]),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(title="Subgenre", bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#e0e0e0", borderwidth=1),
        font=dict(family="Inter, Arial, sans-serif"),
        height=620,
    )

    fig.write_html(f"{OUT}/engagement_vs_scale.html")
    print("engagement_vs_scale.html")


# Chart 3: Subgenre Health
def chart_subgenre_health(health: pd.DataFrame) -> None:
    df = health.sort_values("total_listeners", ascending=True).copy()
    df["subgenre_label"] = df["subgenre"].str.title()
    df["color"] = df["subgenre"].map(SUBGENRE_COLORS).fillna("#888")

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Total Listeners by Subgenre",
                        "Breakout Rate vs Golden Era"),
        column_widths=[0.55, 0.45],
        horizontal_spacing=0.12,
    )

    fig.add_trace(go.Bar(
        y=df["subgenre_label"],
        x=df["total_listeners"],
        orientation="h",
        marker_color=df["color"],
        text=df["total_listeners"].apply(lambda x: f"{x/1e6:.1f}M"),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Total Listeners: %{x:,.0f}<extra></extra>",
        showlegend=False,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df["golden_era_decade"],
        y=df["breakout_pct"],
        mode="markers+text",
        marker=dict(
            color=df["color"],
            size=df["total_artists"].clip(5, 60),
            opacity=0.85,
            line=dict(width=1, color="white"),
        ),
        text=df["subgenre_label"],
        textposition="top center",
        textfont=dict(size=9),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Golden Era: %{x}s<br>"
            "Breakout Rate: %{y:.1f}%<br>"
            "<extra></extra>"
        ),
        showlegend=False,
    ), row=1, col=2)

    fig.update_layout(
        title=dict(
            text="Subgenre Health - Scale, Breakout Rate and Golden Era",
            font=dict(size=20, color="#1a1a2e"),
            x=0.5,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, Arial, sans-serif"),
        height=500,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
    fig.update_xaxes(title_text="Total Listeners", row=1, col=1)
    fig.update_xaxes(title_text="Golden Era (Formation Decade)", row=1, col=2)
    fig.update_yaxes(title_text="Breakout Rate (%)", row=1, col=2)

    fig.write_html(f"{OUT}/subgenre_health.html")
    print("subgenre_health.html")


# Chart 4: Breakout Rankings
def chart_breakout_rankings(preds: pd.DataFrame) -> None:
    underground = (
        preds[preds["current_tier"] == "Underground"]
        .sort_values("breakout_probability", ascending=False)
        .head(20)
        .copy()
    )
    underground = underground.sort_values("breakout_probability", ascending=True)
    underground["color"] = underground["subgenre"].map(SUBGENRE_COLORS).fillna("#888")
    underground["prob_pct"] = underground["breakout_probability"] * 100
    underground["label"] = underground.apply(
        lambda r: f"{r['artist_name']} ({r['subgenre'].title()})", axis=1
    )

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=underground["label"],
        x=underground["prob_pct"],
        orientation="h",
        marker=dict(color=underground["color"], line=dict(width=0)),
        text=underground["prob_pct"].apply(lambda x: f"{x:.1f}%"),
        textposition="outside",
        customdata=underground[["current_listeners", "top_breakout_factor"]],
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Breakout Probability: %{x:.1f}%<br>"
            "Current Listeners: %{customdata[0]:,.0f}<br>"
            "Key Factor: %{customdata[1]}"
            "<extra></extra>"
        ),
    ))

    seen = set()
    for _, row in underground.iterrows():
        sg = row["subgenre"].title()
        if sg not in seen:
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(color=row["color"], size=10),
                name=sg, showlegend=True,
            ))
            seen.add(sg)

    fig.update_layout(
        title=dict(
            text="Breakout Predictor - Top 20 Underground Bands Most Likely to Break Out",
            font=dict(size=20, color="#1a1a2e"),
            x=0.5,
        ),
        xaxis=dict(
            title="Breakout Probability (%)",
            showgrid=True,
            gridcolor="#f0f0f0",
            range=[0, underground["prob_pct"].max() * 1.25],
        ),
        yaxis=dict(showgrid=False),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, Arial, sans-serif"),
        legend=dict(title="Subgenre", bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#e0e0e0", borderwidth=1),
        height=680,
        margin=dict(l=220),
    )

    fig.write_html(f"{OUT}/breakout_rankings.html")
    print("breakout_rankings.html")


def main():
    print("Loading data...")
    legacy, health, features, preds = load_data()

    print("\nGenerating charts...")
    chart_golden_era(features)
    chart_engagement_vs_scale(features)
    chart_subgenre_health(health)
    chart_breakout_rankings(preds)

    print(f"\nAll charts saved to {OUT}/")


if __name__ == "__main__":
    main()
