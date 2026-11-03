# Preprocessing offline dei dataset

Script che hanno generato i dati in `data/datasets/<Dataset>/Preprocessed/`, usati
dall'orchestrator come input di ogni job. Sono archiviati **così come sono**, per
tracciabilità: non fanno parte di FLIGHT, non vengono eseguiti dalla pipeline e sono
esclusi da ruff (`scripts/ruff.toml`).

| Dataset      | Script                                 | Output                                  |
|--------------|----------------------------------------|-----------------------------------------|
| GEFCom2012   | `GEFCom2012_01_prepare_data.py`        | `GEFCom2012/Preprocessed/data_clean.feather` |
| ASHRAE       | `ASHRAE_01_prepare_data_casestudy.py`  | `ASHRAE/Preprocessed/data_clean.feather`     |
| PlantVillage | `PlantVillage_00_preprocessing.py`     | `PlantVillage/Preprocessed/<malattia>/*.JPG` |

La corrispondenza con i file presenti in `data/datasets` è stata verificata su colonne,
numero di righe/immagini, range dei target e schema dei nomi file.

## Riprodurli

Non girano senza modifiche:

- dipendono da moduli non archiviati: `utils.mem.reduce_mem_usage` (GEFCom2012) e
  `ashrae.utils` (`DATA_PATH`, `timer`, `load_data`) per ASHRAE;
- usano percorsi relativi a una vecchia cartella `scripts_dev` e ai dataset originali
  (`datasets/GEFCom2012/Load/`, i CSV Kaggle di ASHRAE letti tramite
  `ashrae.utils.load_data`, `Plant_leave_diseases_dataset_without_augmentation` di
  PlantVillage);
- dipendenze extra: `astral`, `pytz`, `holidays` (ASHRAE), `Pillow`, `tqdm`,
  `matplotlib`.

## Cosa fanno

### GEFCom2012

- Legge `Load_history.csv` (20 zone, orario, 2004 – giugno 2008) e lo porta in formato
  lungo (`timestamp`, `zone_id`, `load`); scarta i timestamp senza dati in nessuna zona.
- `is_holiday` dalla `Holiday_List.csv` ufficiale (con correzione manuale del
  31/12/2004).
- Calendario: codifica ciclica sin/cos di ora, giorno del mese, giorno della settimana,
  giorno dell'anno; inoltre `day`, `month`, `year_diff = anno - 2004`.
- Carichi ≤ 0 posti a 1, poi **target `load_scaled = log10(load)`**.
- `ts_block`: per ogni zona si apre un nuovo blocco a ogni buco > 1 h nella serie
  (180 blocchi, 9 per zona). Il client lo usa per non costruire finestre/lag a cavallo
  dei buchi.

### ASHRAE

Basato sul notebook pubblico
<https://github.com/vdhar1992/ASHRAE_Great_energy_predictorIII_CaseStudy>.

- Meteo: aggiunge le righe orarie mancanti per sito, riempie i NaN con ffill/bfill,
  calcola `wind_direction_x/y` e `relative_humidity` (da `dew_temperature`).
- Calendario: stessa codifica ciclica di GEFCom2012.
- Edifici: superficie in m² e poi `square_meters_log10`; rimossi `year_built` e
  `floor_count`.
- Rimosse le letture uguali a 0 (circa 2 M righe) e gli edifici 1099 e 778.
- **Target `meter_reading_scaled = log10(meter_reading)`**, senza standardizzazione.
- `is_holiday` per paese del sito (US, UK, Canada, Irlanda; libreria `holidays`).
- `isdaytime` da alba/tramonto reali di ogni sito (`astral`).
- Riduzione dei dtype (float16/int8, `primary_use` categorica); rimossi
  `dew_temperature`, `dayofyear`, `day`, `month`.

Risultato: 18,3 M righe, 1.447 edifici, 16 siti, 0 NaN.

### PlantVillage

- Parte dalla versione *without augmentation*, escluso `Background_without_leaves`.
- Raggruppa le cartelle coltura+malattia per **sola malattia**, tenendo 8 classi:
  `healthy`, `rust`, `Early_blight`, `Black_rot`, `Late_blight`, `Bacterial_spot`,
  `Citrus_greening`, `Powdery_mildew`. Le altre 12 malattie sono commentate e quindi
  escluse (37.074 immagini contro le 54.305 originali).
- Il sottocampionamento della classe `healthy` è commentato: non applicato.
- Ogni immagine: resize da 256×256 a 224×224, normalizzazione min-max **per singola
  immagine** a 0–255, salvataggio in JPEG come `<Coltura>_<nome originale>.JPG`.

## Bug e anomalie noti

I dati in `data/datasets` sono stati generati con questi bug: correggerli richiede di
rigenerare i file.

1. **ASHRAE – `wind_direction_x/y` errate.** Lo script calcola
   `wind_direction * cos(wind_direction)` (e analogo con `sin`), cioè moltiplica per
   l'angolo in gradi. Valori tra circa −275 e 360 invece di [−1, 1]. La forma corretta è
   `cos/sin(wind_direction)`, eventualmente moltiplicata per `wind_speed`.
2. **ASHRAE – riempimento meteo che attraversa i siti.**
   `weather.groupby("site_id")[na_cols].ffill().bfill()`: il `ffill` è per sito, ma il
   `bfill` è applicato al DataFrame intero, quindi i NaN iniziali di un sito possono
   essere riempiti con valori del sito successivo. Effetto limitato.
3. **ASHRAE – feature meteo non scalate.** `air_temperature`, `cloud_coverage`,
   `precip_depth_1_hr`, `wind_direction`, `wind_speed` restano in unità originali. A
   runtime FLIGHT standardizza solo `sea_level_pressure`
   (`flightorchestrator/.../data_load/datasets/ASHRAE.py`).
4. **GEFCom2012 – separatore delle migliaia.** Il CSV usa la virgola come separatore
   delle migliaia e lo script la sostituisce con il punto, quindi i carichi risultano
   divisi per 1000. Dopo il `log10` è solo uno shift costante di −3: innocuo, ma
   `load_scaled` non è il log del carico in kW.
5. **PlantVillage – `dict_disease_prepro.pkl` obsoleto.** Il pickle in
   `data/datasets/PlantVillage/Preprocessed/` viene da una versione precedente dello
   script (nomi senza prefisso della coltura) e contiene percorsi duplicati (es.
   `healthy`: 15.084 voci, 6.232 uniche). FLIGHT non lo usa (il loader legge le
   cartelle), ma è fuorviante. Inoltre lo script corrente costruisce `dict_prepro` con
   i nomi originali e non con quelli effettivamente salvati.
6. **PlantVillage – collisioni di nome.** Il file salvato prende sempre l'estensione
   `.JPG`: due originali come `image (77).jpg` e `image (77).JPG` della stessa coltura
   finiscono sullo stesso file. Su disco mancano così 2 immagini (`healthy` 15.083 su
   15.084, `Late_blight` 2.908 su 2.909).
7. **PlantVillage – controllo inefficace.** `if disease not in diseases` usa la
   variabile del ciclo precedente invece di `matched_disease`, quindi non scarta mai
   nulla; le cartelle senza corrispondenza vengono comunque saltate dal successivo
   `if matched_disease`.
