# 🩺 Analisi di Serie Temporali su Dati di Ipertensione

Progetto sviluppato per il corso di **Information Systems and Business Intelligence (ISBI)**.

L'obiettivo del progetto è realizzare una pipeline completa di Data Science applicata all'analisi di dati di monitoraggio della pressione arteriosa, dalla selezione del dataset fino allo sviluppo di un'applicazione interattiva per utenti non tecnici.

---

## Struttura del repository

```
├── PROGETTO_ISBI.ipynb        # Notebook Google Colab (Fase 2)
├── app.py                     # Applicazione Streamlit (Fase 3)
├── requirements.txt           # Librerie Python necessarie
├── README.md                  # Documentazione del progetto
└── Relazione_Fase1_ISBI.pdf  # Relazione sulla scelta del dataset (Fase 1)
```

---

## Dataset

**Fonte**

Synthetic Health Data (Anomaly): 100 Patients, 10-Min

Kaggle — autore: *shrishagrawal*

https://www.kaggle.com/datasets/shrishagrawal/synthetic-health-dataanomaly-100patients10-min

Il dataset contiene dati sintetici di monitoraggio cardiovascolare continuo relativi a 100 pazienti, con rilevazioni ogni 10 minuti per un periodo di circa un anno.

Le motivazioni della scelta del dataset, la descrizione delle variabili e le considerazioni etiche sono riportate nella relazione della **Fase 1**.

---

## Scelte implementative

Durante la Fase 2 l'analisi statistica è stata condotta sul **Patient_ID = 1**, selezionato dal dataset originale contenente 100 pazienti.

Per garantire una maggiore fluidità dell'applicazione Streamlit durante il deploy, è stato utilizzato come dataset di esempio un file CSV contenente esclusivamente le osservazioni del paziente analizzato nella Fase 2.

L'applicazione è stata comunque progettata per supportare dataset contenenti più pazienti. Infatti, quando il file caricato include più soggetti, è disponibile un menu che consente di selezionare interattivamente il paziente da analizzare.

---

## Funzionalità dell'applicazione

L'applicazione Streamlit permette di:

- caricare un file CSV con la stessa struttura del dataset originale;
- selezionare il paziente da analizzare;
- scegliere la variabile clinica da visualizzare;
- filtrare l'intervallo temporale di interesse;
- consultare statistiche descrittive della serie selezionata;
- visualizzare la serie temporale in maniera interattiva;
- generare previsioni mediante il modello Prophet;
- visualizzare il grafico della previsione con relativo intervallo di previsione;
- esportare le previsioni in formato CSV;
- esportare il grafico della previsione in formato PNG.

---

## Notebook Google Colab

Il notebook sviluppato nella Fase 2 comprende:

- caricamento del dataset;
- preprocessing dei dati;
- gestione dei valori mancanti;
- parsing delle date;
- resampling temporale;
- analisi esplorativa;
- statistiche descrittive;
- decomposizione della serie temporale;
- test di stazionarietà (ADF e KPSS);
- analisi ACF e PACF;
- implementazione dei modelli:
  - ARIMA
  - SARIMA
  - Prophet
- confronto tra i modelli;
- valutazione mediante MAE, RMSE e MAPE su un test set temporalmente separato.

---

## Modello di previsione

L'applicazione utilizza **Prophet**, risultato il modello con le migliori prestazioni predittive durante la Fase 2 del progetto.

La selezione è stata effettuata confrontando ARIMA, SARIMA e Prophet mediante le metriche:

- MAE;
- RMSE;
- MAPE.

Le metriche sono state calcolate su un **test set temporalmente separato di 14 giorni, corrispondente a 336 osservazioni orarie**.

Nel notebook della Fase 2, Prophet è stato addestrato esclusivamente sul training set, mentre gli ultimi 14 giorni sono stati esclusi dall'addestramento e utilizzati per valutare le prestazioni predittive.

Nell'applicazione Streamlit, invece, Prophet viene **addestrato nuovamente sull'intera serie temporale caricata dall'utente**, includendo anche le osservazioni che nel notebook costituivano il test set. Questa scelta è coerente con il diverso obiettivo dell'applicazione: non più confrontare i modelli, ma generare previsioni future utilizzando tutte le informazioni disponibili.

---

## Come eseguire il notebook

Il notebook è stato sviluppato per Google Colab.

1. Aprire il notebook.
2. Caricare il dataset scaricato da Kaggle.
3. Eseguire tutte le celle in ordine.

---

## Come eseguire l'applicazione

## 1. Clonare il repository

```bash
git clone https://github.com/<username>/<repository>.git
```

oppure scaricare il repository in formato ZIP.

---

## 2. Creare un ambiente virtuale (consigliato)

Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Installare le dipendenze

```bash
pip install -r requirements.txt
```

---

## 4. Avviare Streamlit

```bash
streamlit run app.py
```

L'applicazione sarà disponibile all'indirizzo

```
http://localhost:8501
```

---

## Formato del file CSV richiesto

Il file CSV caricato deve contenere almeno le colonne `Patient_ID`, `Timestamp`
e `BP_Systolic`. Le altre colonne del dataset originale sono opzionali e, se
presenti, possono essere selezionate per la visualizzazione.

---

## Link del progetto

Notebook Google Colab

> Inserire il link pubblico

Applicazione Streamlit

> Inserire il link dopo il deploy su Streamlit Community Cloud

Repository GitHub

> Inserire il link GitHub

---

## Tecnologie utilizzate

- Python
- Pandas
- NumPy
- Matplotlib
- Plotly
- Prophet
- Statsmodels
- Scikit-learn
- Streamlit
- Google Colab

---

## Note

L'applicazione è stata progettata con finalità dimostrative e didattiche. Le previsioni generate non hanno alcun valore diagnostico o clinico e non devono essere utilizzate a supporto di decisioni mediche.

---

## Autore

**Alessandra Cosentino**

Corso di **Information Systems and Business Intelligence (ISBI)**

Università degli Studi di Napoli Federico II