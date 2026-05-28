"""
================================================================================
Step 2 — Preprocessing Pipeline
Sessile Drop Contact Angle Images
================================================================================

Dataset:  40 immagini RGB (1280x960 px)
          3 liquidi (H2O, CH2I2, Octanol) x ~13 substrati
Autori:   [nome]
Corso:    CO4 I4LS — ZHAW
================================================================================

Pipeline steps:
  1. Crop timestamp         (bottom 13% dell'immagine)
  2. Selezione canale       (canale RGB con varianza massima nella ROI superiore)
  3. Rilevamento substrato  (Sobel-y peak nella metà inferiore)
  4. Background subtraction (morphological closing + divisione normalizzata)
  5. CLAHE                  (clip=1.5, tile=16x16)
  6. Bilateral filter       (d=9, sigmaColor=sigmaSpace=75)
  7. Blank detection        (varianza ROI sopra substrato < soglia)

Requirements:
  pip install opencv-python numpy matplotlib
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


# ================================================================================
# COSTANTI GLOBALI
# Calibrate sull'intero dataset di 40 immagini
# ================================================================================

TIMESTAMP_FRACTION  = 0.13    # bottom 13% contiene il testo timestamp
CLAHE_CLIP_LIMIT    = 1.5     # conservativo: non amplifica rumore
CLAHE_TILE_GRID     = (16, 16)
BILATERAL_D         = 9
BILATERAL_SIGMA_C   = 75
BILATERAL_SIGMA_S   = 75
MORPH_BG_KERNEL     = 101     # kernel per background subtraction (px)
BLANK_VAR_THRESHOLD = 25.0    # varianza sotto la quale → nessuna goccia
                               # calibrato su: i2-fuoc, test.jpg


# ================================================================================
# STEP 2.1 — Crop del timestamp
# ================================================================================

def crop_timestamp(img: np.ndarray,
                   fraction: float = TIMESTAMP_FRACTION) -> np.ndarray:
    """
    Rimuove il banner inferiore con timestamp (ora e data).

    Motivazione:
        Il testo bianco in basso a sinistra è presente in TUTTE le 40 immagini
        e occupa circa il 13% dell'altezza (verificato su tutto il dataset).
        Il testo ad alto contrasto verrebbe interpretato come foreground da
        qualsiasi thresholder → falsi positivi nelle maschere binarie.

    Args:
        img      : immagine BGR o grayscale (np.ndarray)
        fraction : frazione dell'altezza da rimuovere dal basso

    Returns:
        Immagine croppata (senza il banner inferiore)
    """
    h = img.shape[0]
    return img[:int(h * (1 - fraction)), :]


# ================================================================================
# STEP 2.2 — Selezione del canale ottimale
# ================================================================================

def extract_best_channel(img_bgr: np.ndarray) -> np.ndarray:
    """
    Seleziona il canale RGB che massimizza il contrasto goccia/sfondo.

    Motivazione:
        Il dataset presenta sfondi di colore radicalmente diversi:
          - Sfondo beige/neutro (maggioranza): grayscale standard è sufficiente
          - Sfondo verde (lotus): il canale R separa meglio la goccia scura
          - Substrato dorato (metall): il canale B enfatizza il bordo
        Invece di hard-codare il canale per ogni substrato, si seleziona
        automaticamente il canale con varianza massima nella ROI superiore
        (sfondo + goccia), che è un proxy robusto per "maggior informazione
        sul contorno della goccia".

    Args:
        img_bgr : immagine in formato BGR (np.ndarray uint8)

    Returns:
        Immagine grayscale del canale selezionato (np.ndarray uint8)
    """
    h = img_bgr.shape[0]
    roi_h = int(h * 0.6)  # analizza solo il 60% superiore (sfondo + goccia)

    channels = cv2.split(img_bgr)  # restituisce [B, G, R]
    variances = [np.var(ch[:roi_h, :]) for ch in channels]
    best_ch_idx = int(np.argmax(variances))

    return channels[best_ch_idx]


# ================================================================================
# STEP 2.3 — Rilevamento della riga del substrato
# ================================================================================

def detect_substrate_line(gray: np.ndarray) -> int:
    """
    Individua la riga y corrispondente alla superficie superiore del substrato.

    Algoritmo:
        Gradiente verticale di Sobel (dy) applicato alla metà inferiore
        dell'immagine. La transizione sfondo → substrato genera il picco
        di energia orizzontale più elevato. Un smoothing con finestra
        mobile di 15 righe rende il rilevamento robusto su substrati
        texturizzati (tessuto, suie).

    Motivazione:
        Lavorare solo sulla ROI sopra il substrato è essenziale per:
          - Evitare che il substrato metallico brillante inquini l'istogramma
            del CLAHE e la background subtraction
          - Circoscrivere la blank detection alla zona effettiva della goccia
          - Migliorare la qualità del preprocessing su immagini come
            h2o-fog e oct-fog dove il substrato è molto luminoso

    Args:
        gray : immagine grayscale (np.ndarray uint8)

    Returns:
        Indice di riga (int) della superficie del substrato
    """
    h = gray.shape[0]
    # Ignora il terzo superiore (solo sfondo, nessun substrato)
    search_region = gray[h // 3:, :]

    sobel_y = cv2.Sobel(search_region.astype(np.float32),
                        cv2.CV_32F, dx=0, dy=1, ksize=5)
    row_energy = np.sum(np.abs(sobel_y), axis=1)

    # Smoothing per robustezza su substrati texturizzati
    row_energy_smooth = np.convolve(row_energy, np.ones(15) / 15, mode='same')
    substrate_row = int(np.argmax(row_energy_smooth)) + h // 3

    return substrate_row


# ================================================================================
# STEP 2.4 — Background subtraction morfologica
# ================================================================================

def subtract_background(gray: np.ndarray,
                         kernel_size: int = MORPH_BG_KERNEL) -> np.ndarray:
    """
    Rimuove il gradiente di illuminazione non uniforme tramite
    morphological closing seguito da divisione normalizzata.

    Motivazione critica per questo dataset:
        Diverse immagini presentano gradienti di illuminazione estremi
        che rendono impossibile qualsiasi threshold globale:
          - h2o-fog-5/100, oct-fog-100: illuminazione da sotto crea un
            gradiente verticale enorme (valori ~50 in cima, ~230 in fondo)
          - oct-lotus, h20-lotus: sfondo verde con sfumatura irregolare
          - h2o-tessuto, oct-tessuto: substrato fibroso con molti picchi
            locali di intensità che mimano il bordo della goccia

        Il morphological closing con kernel ellittico grande (101x101 px,
        circa 8% della larghezza dell'immagine) approssima il campo di
        illuminazione locale senza essere disturbato dalla goccia, che
        occupa al massimo il 30% dell'area.

        La divisione normalizzata (divide + scale=255) produce
        un'immagine dove l'intensità è relativa allo sfondo locale,
        indipendentemente dall'illuminazione assoluta.

    Args:
        gray        : immagine grayscale (np.ndarray uint8)
        kernel_size : dimensione del kernel morfologico (default 101 px)

    Returns:
        Immagine normalizzata (np.ndarray uint8)
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    normalized = cv2.divide(
        gray.astype(np.float32),
        background.astype(np.float32),
        scale=255.0
    )
    return np.clip(normalized, 0, 255).astype(np.uint8)


# ================================================================================
# STEP 2.5 — Contrast Enhancement (CLAHE)
# ================================================================================

def apply_clahe(gray: np.ndarray,
                clip_limit: float = CLAHE_CLIP_LIMIT,
                tile_grid: tuple = CLAHE_TILE_GRID) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Motivazione:
        Perché CLAHE e non equalizzazione globale:
          - L'equalizzazione globale satura le zone già ben contrastate
            (metall, lotus) e amplifica eccessivamente il rumore nelle
            zone flat (sfondo beige uniforme).
          - CLAHE opera su tile locali (16x16 = ~80x60 px per immagine
            1280x960) → migliora selettivamente le zone a basso contrasto
            dove si trovano i bordi delle gocce piatte (oct-fog, i2-fog).

        clipLimit=1.5: valore conservativo scelto perché i bordi delle
        gocce piatte (octanol su substrati idrofili) hanno contrasto
        tipicamente inferiore a 5 DN → serve amplificazione moderata,
        non aggressiva, per evitare di amplificare anche il rumore.

        tileGridSize=(16,16): tile da ~80x60 px, dimensione sufficiente
        a contenere l'intera goccia in 1-2 tile adiacenti.

    Args:
        gray       : immagine grayscale (np.ndarray uint8)
        clip_limit : limite di amplificazione del contrasto
        tile_grid  : dimensione della griglia di tile (tuple)

    Returns:
        Immagine con contrasto migliorato (np.ndarray uint8)
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(gray)


# ================================================================================
# STEP 2.6 — Denoising con Bilateral Filter
# ================================================================================

def denoise(gray: np.ndarray) -> np.ndarray:
    """
    Denoising edge-aware con filtro bilaterale.

    Motivazione:
        Perché bilateral filter e non Gaussian blur:
          - Il Gaussian blur degrada i bordi delle gocce a basso contrasto
            (oct-fog, i2-fog): il gradiente al bordo, già debole (~3-5 DN),
            viene ulteriormente attenuato → fallimento della segmentazione.
          - Il filtro bilaterale media i pixel omogenei (sfondo uniforme)
            ma PRESERVA le discontinuità (bordi della goccia) perché
            la funzione peso tiene conto sia della distanza spaziale
            (sigmaSpace) sia della differenza di intensità (sigmaColor).

        Parametri d=9, sigmaColor=sigmaSpace=75:
          - d=9: finestra di 9 px di diametro, sufficiente per il rumore
            di shot presente nelle immagini poco illuminate
          - sigma=75: trade-off tra smoothing del rumore e preservazione
            dei bordi tenuissimi tipici delle gocce di octanol

    Args:
        gray : immagine grayscale (np.ndarray uint8)

    Returns:
        Immagine denoisata (np.ndarray uint8)
    """
    return cv2.bilateralFilter(
        gray,
        d=BILATERAL_D,
        sigmaColor=BILATERAL_SIGMA_C,
        sigmaSpace=BILATERAL_SIGMA_S
    )


# ================================================================================
# STEP 2.7 — Rilevamento immagini blank
# ================================================================================

def detect_blank_image(gray: np.ndarray,
                        substrate_y: int,
                        threshold: float = BLANK_VAR_THRESHOLD) -> bool:
    """
    Classifica l'immagine come blank (nessuna goccia presente) se la
    varianza dell'intensità nella ROI sopra il substrato è < threshold.

    Immagini blank nel dataset:
        - i2-fuoc : CH2I2 su suie, liquido completamente spalmato (angolo ~0)
        - test.jpg: immagine di calibrazione senza goccia

    La soglia BLANK_VAR_THRESHOLD = 25.0 è conservativa: preferisce
    falsi negativi (classifica blank come goccia) a falsi positivi
    (scarta immagini valide). I casi border-line come oct-fog-5 hanno
    varianza bassa ma comunque superiore a 25.

    Args:
        gray        : immagine grayscale preprocessata (np.ndarray uint8)
        substrate_y : riga del substrato (da detect_substrate_line)
        threshold   : soglia di varianza (default 25.0)

    Returns:
        True se l'immagine è blank, False altrimenti
    """
    roi = gray[:substrate_y, :]
    return float(np.var(roi)) < threshold


# ================================================================================
# PIPELINE COMPLETA — Singola immagine
# ================================================================================

def preprocess_image(img_path: str) -> dict:
    """
    Applica l'intera pipeline di preprocessing a una singola immagine.

    Sequenza degli step:
        1. Crop timestamp       (rimuove bottom 13%)
        2. Canale ottimale      (max varianza nella ROI superiore)
        3. Detect substrato     (Sobel-y peak nella metà inferiore)
        4. BG subtraction       (morphological closing + divisione)
        5. CLAHE                (clip=1.5, tile=16x16)
        6. Bilateral filter     (d=9, sigma=75)
        7. Blank detection      (varianza ROI < 25.0)

    Args:
        img_path : percorso all'immagine grezza (.jpg o .png)

    Returns:
        dict con le chiavi:
          'preprocessed'  : np.ndarray uint8, immagine pronta per segmentazione
          'gray_original' : np.ndarray uint8, grayscale senza preprocessing
          'substrate_y'   : int, riga stimata della superficie del substrato
          'is_blank'      : bool, True se nessuna goccia rilevata
          'path'          : str, percorso originale dell'immagine
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Impossibile leggere l'immagine: {img_path}")

    # Step 1 — Crop timestamp
    img_cropped = crop_timestamp(img)

    # Step 2 — Canale ottimale
    gray = extract_best_channel(img_cropped)
    gray_original = gray.copy()

    # Step 3 — Rilevamento substrato
    substrate_y = detect_substrate_line(gray)

    # Step 4 — Background subtraction
    bg_subtracted = subtract_background(gray)

    # Step 5 — CLAHE
    enhanced = apply_clahe(bg_subtracted)

    # Step 6 — Bilateral filter
    denoised = denoise(enhanced)

    # Step 7 — Blank detection
    is_blank = detect_blank_image(denoised, substrate_y)

    # Step 8 — Crop ROI sopra substrato
    roi = crop_above_substrate(denoised, substrate_y)

    # Step 9 — Normalizzazione polarità (goccia sempre scura)
    roi_normalized = normalize_drop_polarity(roi)

    return {
        'preprocessed':      denoised,        # immagine intera preprocessata
        'roi':               roi_normalized,  # ROI sopra substrato, pronta per segmentazione
        'gray_original':     gray_original,
        'substrate_y':       substrate_y,
        'is_blank':          is_blank,
        'path':              img_path,
    }


# ================================================================================
# BATCH PROCESSING — Intero dataset
# ================================================================================

def preprocess_dataset(input_dir: str,
                        output_dir: str,
                        extensions: tuple = ('.jpg', '.png')) -> list:
    """
    Applica la pipeline a tutto il dataset e salva le immagini preprocessate.

    Le immagini vengono salvate in output_dir con suffisso _preprocessed.png.
    Le immagini classificate come blank vengono comunque salvate (non scartate)
    ma flaggate nel campo 'is_blank' del risultato.

    Args:
        input_dir  : cartella con le immagini originali
        output_dir : cartella di output per le immagini preprocessate
        extensions : tuple di estensioni da processare

    Returns:
        Lista di dict (uno per immagine) con tutti i metadati di preprocessing.

    Esempio:
        results = preprocess_dataset('data/raw', 'data/preprocessed')
    """
    in_path  = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    img_files = sorted([f for f in in_path.iterdir()
                         if f.suffix.lower() in extensions])

    print(f"Trovate {len(img_files)} immagini in '{input_dir}'")
    print("-" * 60)

    results = []
    for img_file in img_files:
        result = preprocess_image(str(img_file))

        out_file = out_path / (img_file.stem + "_preprocessed.png")
        cv2.imwrite(str(out_file), result['preprocessed'])
        result['output_path'] = str(out_file)
        results.append(result)

        tag = "BLANK" if result['is_blank'] else f"substrate_y={result['substrate_y']}"
        print(f"  {img_file.name:<30} → {tag}")

    n_blank = sum(r['is_blank'] for r in results)
    print("-" * 60)
    print(f"Totale: {len(results)} immagini | "
          f"{n_blank} blank | "
          f"{len(results) - n_blank} con goccia")

    return results


# ================================================================================
# VISUALIZZAZIONE — Per il Jupyter Notebook
# ================================================================================

def visualize_preprocessing(result: dict,
                              figsize: tuple = (14, 4)) -> None:
    """
    Visualizza side-by-side: originale | preprocessata | linea substrato.

    Da usare nel Jupyter Notebook per documentare le scelte di preprocessing
    e mostrare il miglioramento apportato da ogni step della pipeline.

    Args:
        result  : dict restituito da preprocess_image()
        figsize : dimensione della figura matplotlib
    """
    img_name = Path(result['path']).name
    orig     = result['gray_original']
    prep     = result['preprocessed']
    sub_y    = result['substrate_y']

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(img_name + (" [BLANK]" if result['is_blank'] else ""),
                 fontsize=11, fontweight='bold')

    axes[0].imshow(orig, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Originale (grayscale)')
    axes[0].axis('off')

    axes[1].imshow(prep, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title('Preprocessata\n(BG-sub + CLAHE + Bilateral)')
    axes[1].axis('off')

    overlay = cv2.cvtColor(prep, cv2.COLOR_GRAY2BGR)
    cv2.line(overlay, (0, sub_y), (overlay.shape[1], sub_y), (0, 0, 255), 2)
    axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f'Substrato rilevato\n(y = {sub_y} px)')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_pipeline_steps(img_path: str,
                               figsize: tuple = (18, 4)) -> None:
    """
    Visualizza ogni step intermedio della pipeline per una singola immagine.
    Utile per giustificare nel notebook la necessità di ogni step.

    Args:
        img_path : percorso all'immagine da analizzare
        figsize  : dimensione della figura matplotlib
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Impossibile leggere: {img_path}")

    img_name = Path(img_path).name

    # Esegui i singoli step
    cropped     = crop_timestamp(img)
    gray        = extract_best_channel(cropped)
    sub_y       = detect_substrate_line(gray)
    bg_sub      = subtract_background(gray)
    clahe_out   = apply_clahe(bg_sub)
    final       = denoise(clahe_out)

    steps = [
        (gray,      "Step 2: Canale ottimale"),
        (bg_sub,    "Step 4: BG subtraction"),
        (clahe_out, "Step 5: CLAHE"),
        (final,     "Step 6: Bilateral filter"),
    ]

    fig, axes = plt.subplots(1, len(steps), figsize=figsize)
    fig.suptitle(f"Pipeline steps — {img_name}", fontsize=11, fontweight='bold')

    for ax, (img_step, title) in zip(axes, steps):
        ax.imshow(img_step, cmap='gray', vmin=0, vmax=255)
        ax.set_title(title, fontsize=9)
        ax.axhline(y=sub_y, color='red', linewidth=1, linestyle='--',
                   label=f'substrate y={sub_y}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()


# ================================================================================
# ENTRY POINT
# ================================================================================

if __name__ == "__main__":

    # --- Batch processing dell'intero dataset ---
    results = preprocess_dataset(
        input_dir  = "data/raw",
        output_dir = "data/preprocessed"
    )

    # --- Visualizzazione esempi rappresentativi nel notebook ---
    # Seleziona un campione che copre i diversi scenari di contrasto
    representative_cases = [
        'h2o-rain-100',   # caso standard ben contrastato
        'oct-fog-5',      # caso difficile: angolo basso, contrasto minimo
        'h20-lotus',      # sfondo verde, goccia sferica staccata
        'i2-metall',      # substrato metallico, inversione contrasto
        'i2-fuoc',        # blank: nessuna goccia
    ]

    for r in results:
        name = Path(r['path']).stem
        if any(case in name for case in representative_cases):
            print(f"\n--- {name} | blank={r['is_blank']} | substrate_y={r['substrate_y']} ---")
            visualize_preprocessing(r)

    # --- Visualizzazione steps intermedi per un caso difficile ---
    for r in results:
        if 'oct-fog-5' in r['path']:
            visualize_pipeline_steps(r['path'])
            break


# ================================================================================
# STEP 2.8 — Crop ROI verticale (sopra il substrato)
# ================================================================================

def crop_above_substrate(img: np.ndarray,
                          substrate_y: int,
                          margin: int = 5) -> np.ndarray:
    """
    Ritaglia l'immagine mantenendo solo la zona sopra il substrato.

    Motivazione:
        Dai risultati di segmentazione automatica sul dataset si osserva che
        l'errore più frequente è la segmentazione del substrato stesso invece
        della goccia. Questo accade perché:
          - Il substrato metallico (metall, print) ha intensità molto variabile
            e genera bordi forti che dominano qualsiasi threshold
          - Il substrato fibroso (tessuto) ha texture che mima il contorno
            della goccia
          - Il substrato luminoso (teflon, fog) crea un gradiente verticale
            che sposta la distribuzione dell'istogramma

        Rimuovere completamente il substrato dalla ROI prima della
        segmentazione elimina questi falsi positivi senza perdere
        alcuna informazione sulla goccia.

        Il margine di 5 px garantisce che il punto di contatto della goccia
        con il substrato (contact line) sia incluso nella ROI.

    Args:
        img         : immagine grayscale preprocessata (np.ndarray uint8)
        substrate_y : riga del substrato (da detect_substrate_line)
        margin      : pixel aggiuntivi sotto substrate_y da includere

    Returns:
        ROI ritagliata (np.ndarray uint8) — solo zona sopra il substrato
    """
    cut = min(substrate_y + margin, img.shape[0])
    return img[:cut, :]


# ================================================================================
# STEP 2.9 — Polarity detection e normalizzazione del contrasto
# ================================================================================

def normalize_drop_polarity(roi: np.ndarray) -> np.ndarray:
    """
    Standardizza la polarità dell'immagine in modo che la goccia sia
    SEMPRE più scura dello sfondo (convenzione: goccia = pixel scuri).

    Motivazione:
        Nel dataset esistono due scenari di contrasto opposti:

        Scenario A — goccia più scura dello sfondo (maggioranza):
            metall, lotus, fuoc, rain, print, teflon (alcuni casi)
            → threshold diretto: pixel scuri = goccia

        Scenario B — goccia più chiara dello sfondo:
            h2o-teflon (sovraesposto), h2o-fog, oct-fog, i2-teflon
            → senza inversione, il threshold segmenta lo sfondo
              invece della goccia

        Questo è la causa principale degli errori osservati in:
          - h2o-teflon: segmenta il substrato luminoso
          - h2o-plexig: segmenta la linea del substrato
          - h2o-fog:    segmenta l'area chiara sotto invece della goccia

        Algoritmo di rilevamento automatico della polarità:
        Si analizza una striscia orizzontale centrale della ROI
        (tra il 30% e il 70% della larghezza) nella zona dove
        statisticamente la goccia è presente (metà superiore della ROI).
        Se la mediana dei pixel in quella zona è più alta della mediana
        globale dell'immagine → la zona centrale è più chiara dello sfondo
        → la goccia è chiara → inverti.

    Args:
        roi : ROI grayscale sopra il substrato (np.ndarray uint8)

    Returns:
        ROI con polarità normalizzata (goccia scura, sfondo chiaro)
    """
    h, w = roi.shape

    # Zona centrale dove statisticamente si trova la goccia
    x_start = w // 3
    x_end   = 2 * w // 3
    y_end   = h // 2

    center_zone  = roi[:y_end, x_start:x_end]
    median_center = float(np.median(center_zone))
    median_global = float(np.median(roi))

    # Se la zona centrale è significativamente più chiara dello sfondo
    # → la goccia è chiara → invertiamo per standardizzare
    if median_center > median_global + 10:
        return cv2.bitwise_not(roi)

    return roi
