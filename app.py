import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Analisi Serie Storiche - Ipertensione",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Analisi di Serie Storiche su Dati di Ipertensione")

st.markdown("""
Applicazione sviluppata nell'ambito del progetto di **Analisi di Serie Storiche su 
Dati di Ipertensione**. Consente di esplorare il dataset, visualizzare le serie 
temporali, effettuare previsioni mediante il modello selezionato nella Fase 2 ed esportare i 
risultati dell'analisi.

**Funzionalità principali**:
- 📂 Caricare un file CSV
- 📈 Visualizzare la serie temporale
- 🔮 Effettuare previsioni mediante il modello selezionato nella Fase 2
- 💾 Esportare grafici e risultati
""")

st.subheader("📂 Caricamento del dataset")
uploaded_file = st.file_uploader(
    "Carica il file CSV con la stessa struttura del dataset originale",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Carica un file CSV per iniziare.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as error:
    st.error(f"Errore durante la lettura del file: {error}")
    st.stop()

required_columns = [
    "Patient_ID",
    "Timestamp",
    "BP_Systolic"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error(
        "Il file non contiene le colonne obbligatorie: "
        + ", ".join(missing_columns)
    )
    st.stop()

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    errors="coerce"
)

invalid_dates = df["Timestamp"].isna().sum()

if invalid_dates > 0:
    st.warning(
        f"Sono state trovate {invalid_dates} date non valide. "
        "Le righe corrispondenti verranno escluse."
    )

df = df.dropna(subset=["Timestamp"])
df = df.sort_values(["Patient_ID", "Timestamp"])

st.success(
    f"File caricato correttamente: "
    f"{df.shape[0]} righe e {df.shape[1]} colonne."
)

st.subheader("Anteprima del dataset")

st.dataframe(
    df.head(20),
    use_container_width=True
)

with st.expander("Visualizza tutte le colonne"):
    st.write(list(df.columns)) 

st.subheader("📈 Visualizzazione interattiva della serie temporale")

# Se il file contiene più pazienti, permette di selezionarne uno.
patient_ids = sorted(df["Patient_ID"].dropna().unique().tolist())

selected_patient = st.selectbox(
    "Paziente",
    options=patient_ids,
    index=0,
    help="Seleziona il paziente di cui visualizzare la serie temporale."
)

patient_df = df[
    df["Patient_ID"] == selected_patient
].copy()

# Variabili cliniche visualizzabili, con etichette più leggibili.
variable_labels = {
    "BP_Systolic": "Pressione sistolica",
    "BP_Diastolic": "Pressione diastolica",
    "Heart_Rate": "Frequenza cardiaca",
    "SpO2": "Saturazione di ossigeno",
    "Respiration_Rate": "Frequenza respiratoria",
    "Body_Temperature": "Temperatura corporea",
    "Blood_Glucose": "Glicemia",
    "Weight": "Peso",
    "BMI": "Indice di massa corporea",
    "Activity_Level": "Livello di attività",
    "Sleep_Pattern": "Qualità del sonno"
}

units = {
    "BP_Systolic": "mmHg",
    "BP_Diastolic": "mmHg",
    "Heart_Rate": "bpm",
    "SpO2": "%",
    "Respiration_Rate": "atti/min",
    "Body_Temperature": "°F",
    "Blood_Glucose": "mg/dL",
    "Weight": "kg",
    "BMI": "kg/m²"
}

# Mantiene soltanto le variabili realmente presenti nel CSV.
available_variables = [
    column
    for column in variable_labels
    if column in patient_df.columns
    and pd.api.types.is_numeric_dtype(patient_df[column])
]

selected_variable = st.selectbox(
    "Variabile clinica",
    options=available_variables,
    format_func=lambda column: variable_labels[column],
    index=(
        available_variables.index("BP_Systolic")
        if "BP_Systolic" in available_variables
        else 0
    )
)

selected_label = variable_labels[selected_variable]
selected_unit = units.get(selected_variable, "")

min_date = patient_df["Timestamp"].min().date()
max_date = patient_df["Timestamp"].max().date()

selected_dates = st.date_input(
    "Intervallo temporale",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
    help="Seleziona la data iniziale e la data finale da visualizzare."
)

if len(selected_dates) == 2:
    start_date, end_date = selected_dates

    if start_date > end_date:
        st.error("La data iniziale non può essere successiva alla data finale.")
        st.stop()

    filtered_df = patient_df[
        (patient_df["Timestamp"].dt.date >= start_date)
        & (patient_df["Timestamp"].dt.date <= end_date)
    ][
        ["Timestamp", selected_variable]
    ].dropna().copy()

    if filtered_df.empty:
        st.warning(
            "Nessuna osservazione disponibile per il paziente, "
            "la variabile e l'intervallo selezionati."
        )

    else:
        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric(
            "Osservazioni",
            f"{len(filtered_df):,}".replace(",", ".")
        )

        col2.metric(
            "Valore medio",
            f"{filtered_df[selected_variable].mean():.2f} {selected_unit}"
        )

        col3.metric(
            "Minimo",
            f"{filtered_df[selected_variable].min():.2f} {selected_unit}"
        )

        col4.metric(
            "Massimo",
            f"{filtered_df[selected_variable].max():.2f} {selected_unit}"
        )

        col5.metric(
            "Dev. standard",
            f"{filtered_df[selected_variable].std():.2f} {selected_unit}"
        )

        selected_period_days = (end_date - start_date).days + 1

        if selected_period_days > 31:
            plot_df = (
                filtered_df
                .set_index("Timestamp")[selected_variable]
                .resample("h")
                .mean()
                .dropna()
                .reset_index()
            )

            st.caption(
                "Per intervalli temporali superiori a 31 giorni, la serie viene "
                "automaticamente aggregata mediante media oraria al fine di "
                "migliorare la leggibilità del grafico."
            )
        
        else:
            plot_df = filtered_df

        y_axis_label = (
            f"{selected_label} ({selected_unit})"
            if selected_unit
            else selected_label
        )

        fig = px.line(
            plot_df,
            x="Timestamp",
            y=selected_variable,
            title=f"Andamento temporale — {selected_label}",
            labels={
                "Timestamp": "Data e ora",
                selected_variable: y_axis_label
            },
            render_mode="webgl"
        )

        mean_value = plot_df[selected_variable].mean()
        
        fig.add_hline(
            y=mean_value,
            line_dash="dash",
            line_color="darkorange",
            line_width=1,
            annotation_text=f"Media: {mean_value:.2f} {selected_unit}",
            annotation_position="bottom right",
            annotation_font_color="darkorange"
        )

        fig.update_layout(
            xaxis_title="Data e ora",
            yaxis_title=y_axis_label,
            hovermode="x unified"
        )

        fig.update_traces(
            hovertemplate=(
                "<b>%{x|%d/%m/%Y %H:%M}</b><br>"
                + selected_label
                + ": %{y:.2f}"
                + (f" {selected_unit}" if selected_unit else "")
                + "<extra></extra>"
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

st.subheader("🔮 Previsione della pressione sistolica")

st.info("""
### Modello utilizzato

**Prophet**

La previsione viene effettuata esclusivamente sulla variabile **BP_Systolic**, utilizzata 
per il confronto e la selezione dei modelli nella Fase 2.

Il modello viene riaddestrato automaticamente sull'intera serie temporale caricata e genera 
previsioni orarie oltre l'ultima osservazione disponibile.
""")

forecast_hours = st.slider(
    "Orizzonte di previsione (ore)",
    min_value=1,
    max_value=336,
    value=24,
    step=1,
    help=(
        "24 ore corrispondono a 1 giorno, "
        "168 ore a 7 giorni e 336 ore a 14 giorni."
    )
)

run_forecast = st.button(
    "Genera previsione",
    type="primary"
)

if run_forecast:

    with st.spinner("Addestramento del modello e generazione della previsione..."):

        # Selezione della variabile utilizzata nella Fase 2
        prophet_df = patient_df[
            ["Timestamp", "BP_Systolic"]
        ].copy()

        prophet_df = prophet_df.dropna(
            subset=["Timestamp", "BP_Systolic"]
        )

        prophet_df = prophet_df.sort_values("Timestamp")

        # Resampling orario, coerente con il preprocessing della Fase 2
        prophet_df = (
            prophet_df
            .set_index("Timestamp")["BP_Systolic"]
            .resample("h")
            .mean()
            .interpolate(method="time")
            .dropna()
            .reset_index()
        )

        # Prophet richiede le colonne ds e y
        prophet_df.columns = ["ds", "y"]

        if len(prophet_df) < 48:
            st.error(
                "La serie contiene un numero insufficiente di osservazioni "
                "per addestrare il modello Prophet."
            )
            st.stop()

        # Stessa configurazione utilizzata nella Fase 2
        model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=True,
            interval_width=0.95
        )

        # Addestramento sull'intera serie disponibile
        model.fit(prophet_df)

        # Generazione delle sole date future, con frequenza oraria
        future = model.make_future_dataframe(
            periods=int(forecast_hours),
            freq="h",
            include_history=False
        )

        forecast = model.predict(future)

        # Selezione e rinomina dei risultati principali
        forecast_future = forecast[
            ["ds", "yhat", "yhat_lower", "yhat_upper"]
        ].copy()

        forecast_future = forecast_future.rename(
            columns={
                "ds": "Timestamp",
                "yhat": "Previsione",
                "yhat_lower": "Limite_inferiore",
                "yhat_upper": "Limite_superiore"
            }
        )

        forecast_future[
            ["Previsione", "Limite_inferiore", "Limite_superiore"]
        ] = forecast_future[
            ["Previsione", "Limite_inferiore", "Limite_superiore"]
        ].round(2)

    # Salvataggio dei risultati in session_state, così restano disponibili
    # anche dopo un rerun dell'app causato da altre interazioni dell'utente
    st.session_state["forecast_future"] = forecast_future
    st.session_state["prophet_df"] = prophet_df
    st.session_state["forecast_hours"] = forecast_hours
    st.session_state["selected_patient"] = selected_patient


# La visualizzazione dei risultati legge sempre da session_state, non
# direttamente dall'esito del click, così la previsione resta visibile
# anche dopo un rerun (es. cambio di paziente o intervallo di date sopra)
if "forecast_future" in st.session_state:

    forecast_future = st.session_state["forecast_future"]
    prophet_df_result = st.session_state["prophet_df"]
    forecast_hours_result = st.session_state["forecast_hours"]

    st.success(
        f"Previsione completata per le prossime "
        f"{forecast_hours_result} ore."
    )

    st.subheader("Risultati della previsione")

    st.dataframe(
        forecast_future,
        use_container_width=True,
        hide_index=True
    )

    # Per rendere il grafico leggibile si mostrano gli ultimi 7 giorni
    # osservati insieme alle previsioni future
    historical_window = max(168, int(forecast_hours_result))

    history_to_plot = prophet_df_result.tail(
        historical_window
    ).copy()

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(
        history_to_plot["ds"],
        history_to_plot["y"],
        label="Dati osservati",
        linewidth=1
    )

    ax.plot(
        forecast_future["Timestamp"],
        forecast_future["Previsione"],
        label="Previsione Prophet",
        linewidth=2
    )

    ax.fill_between(
        forecast_future["Timestamp"],
        forecast_future["Limite_inferiore"],
        forecast_future["Limite_superiore"],
        alpha=0.2,
        label="Intervallo di previsione (95%)"
    )

    ax.axvline(
        prophet_df_result["ds"].max(),
        linestyle="--",
        linewidth=2,
        color="red",
        label="Inizio della previsione"
    )

    ax.set_title("Previsione della pressione sistolica con Prophet")
    ax.set_xlabel("Data e ora")
    ax.set_ylabel("Pressione sistolica (mmHg)")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()

    st.pyplot(fig)

    st.caption(
        "Le previsioni mostrano un andamento periodico coerente con la "
        "stagionalità giornaliera identificata durante la Fase 2 del progetto. "
        "Per orizzonti di previsione superiori a 24 ore il modello tende a "
        "riprodurre tale andamento ciclico."
    )

    # Salvataggio della figura per l'esportazione PNG
    st.session_state["forecast_figure"] = fig

    st.subheader("💾 Esportazione dei risultati")

    col_csv, col_png = st.columns(2)

    with col_csv:
        csv_data = forecast_future.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="⬇️ Scarica previsioni (CSV)",
            data=csv_data,
            file_name=f"forecast_patient_{st.session_state['selected_patient']}.csv",
            mime="text/csv"
        )

    with col_png:
        import io

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        buffer.seek(0)

        st.download_button(
            label="⬇️ Scarica grafico (PNG)",
            data=buffer,
            file_name=f"forecast_patient_{st.session_state['selected_patient']}.png",
            mime="image/png"
        )