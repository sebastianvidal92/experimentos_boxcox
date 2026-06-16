# -*- coding: utf-8 -*-
"""Genera report/informe_preproc.tex a partir de results/summary_preproc.csv.
Informe SEPARADO del de Box-Cox: compara Raw vs Box-Cox vs HistEq vs CLAHE con
LDA y QDA. Requiere las figuras report/figures/preproc_*.png (ver
src/make_figures_preproc.py)."""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root (parent of src/)
RESULTS_DIR = os.path.join(ROOT, "results")
REPORT_DIR = os.path.join(ROOT, "report")

d = pd.read_csv(os.path.join(RESULTS_DIR, "summary_preproc.csv"))
pw = pd.read_csv(os.path.join(RESULTS_DIR, "pairwise_preproc.csv"))
raw_lam = pd.read_csv(os.path.join(RESULTS_DIR, "results_preproc.csv"))
lam = raw_lam[raw_lam.condition == "boxcox"].groupby("set").lam.mean().round(2).to_dict()

SETS = [("Vertical", "Verticales"), ("Horizontal", "Horizontales"),
        ("Left", "Oblicuas (izq.\\,$\\to$\\,der.)"), ("Big", "Grandes")]
METHODS = ["LDA", "QDA"]
COND_KEYS = ["raw", "boxcox", "histeq", "clahe"]
COND_TEX = {"raw": "Raw", "boxcox": "Box--Cox", "histeq": "HistEq", "clahe": "CLAHE"}
METRICS = ["IoU", "Dice", "Precision", "Recall"]


def stars(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "$^{***}$"
    if p < 0.01:
        return "$^{**}$"
    if p < 0.05:
        return "$^{*}$"
    return ""


def get_block(s, m, metric):
    b = d[(d.set == s) & (d.method == m) & (d.metric == metric)]
    raw = (float(b.iloc[0].raw_mean), float(b.iloc[0].raw_std))
    tf = {r["transform"]: (float(r.tf_mean), float(r.tf_std), float(r.p_perm))
          for _, r in b.iterrows()}
    return raw, tf


def metric_table(metric, caption, label):
    out = ["\\begin{table}[H]\\centering\\small",
           f"\\caption{{{caption}}}\\label{{{label}}}",
           "\\begin{tabular}{l cccc}", "\\toprule",
           "M\\'etodo & Raw & Box--Cox & HistEq & CLAHE \\\\", "\\midrule"]
    for skey, sname in SETS:
        out.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{sname}}} "
                   f"($\\bar\\lambda={lam[skey]:.2f}$)}} \\\\")
        for mth in METHODS:
            raw, tf = get_block(skey, mth, metric)
            means = {"raw": raw[0], "boxcox": tf["boxcox"][0],
                     "histeq": tf["histeq"][0], "clahe": tf["clahe"][0]}
            best = max(means, key=means.get)

            def cell(key, mean, std, p=None):
                body = f"${mean:.1f}\\pm{std:.1f}$" + (stars(p) if p is not None else "")
                return f"\\textbf{{{body}}}" if key == best else body

            cells = [cell("raw", raw[0], raw[1]),
                     cell("boxcox", *tf["boxcox"]),
                     cell("histeq", *tf["histeq"]),
                     cell("clahe", *tf["clahe"])]
            out.append(f"\\quad {mth} & " + " & ".join(cells) + " \\\\")
        out.append("\\addlinespace")
    out[-1] = "\\bottomrule"
    out += ["\\end{tabular}", "\\end{table}"]
    return "\n".join(out)


def pairwise_table(pair, caption, label):
    """Tabla de comparación directa A vs B: Δ = media(A) − media(B), pp, con stars."""
    out = ["\\begin{table}[H]\\centering\\small",
           f"\\caption{{{caption}}}\\label{{{label}}}",
           "\\begin{tabular}{l cccc}", "\\toprule",
           "M\\'etodo & IoU & Dice & Precision & Recall \\\\", "\\midrule"]
    for skey, sname in SETS:
        out.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{sname}}}}} \\\\")
        for mth in METHODS:
            cells = []
            for metric in METRICS:
                r = pw[(pw["pair"] == pair) & (pw["set"] == skey) &
                       (pw["method"] == mth) & (pw["metric"] == metric)].iloc[0]
                cells.append(f"${r.diff_mean:+.1f}$" + stars(r.p_perm))
            out.append(f"\\quad {mth} & " + " & ".join(cells) + " \\\\")
        out.append("\\addlinespace")
    out[-1] = "\\bottomrule"
    out += ["\\end{tabular}", "\\end{table}"]
    return "\n".join(out)


def summary_table():
    """Promedio sobre los 4 sets + conteo de 'mejor por fila'."""
    df = raw_lam.groupby(["method", "condition"])[METRICS].mean() * 100
    out = ["\\begin{table}[H]\\centering\\small",
           "\\caption{Promedio sobre los cuatro conjuntos (200 im\\'agenes) de cada "
           "m\\'etrica, por clasificador y preprocesamiento (\\%). En \\textbf{negrita} "
           "el mejor de cada fila.}\\label{tab:avg}",
           "\\begin{tabular}{l l cccc}", "\\toprule",
           "Clasificador & M\\'etrica & Raw & Box--Cox & HistEq & CLAHE \\\\", "\\midrule"]
    for mth in METHODS:
        for metric in METRICS:
            vals = {c: float(df.loc[(mth, c), metric]) for c in COND_KEYS}
            best = max(vals, key=vals.get)
            cells = [(f"\\textbf{{{vals[c]:.1f}}}" if c == best else f"{vals[c]:.1f}")
                     for c in COND_KEYS]
            lead = f"{mth}" if metric == METRICS[0] else ""
            out.append(f"{lead} & {metric} & " + " & ".join(cells) + " \\\\")
        out.append("\\addlinespace")
    out[-1] = "\\bottomrule"
    out += ["\\end{tabular}", "\\end{table}"]
    return "\n".join(out)


def win_counts():
    """Veces que cada preprocesamiento es el mejor (4 sets x 2 clf x 4 metricas)."""
    win = {c: 0 for c in COND_KEYS}
    for skey, _ in SETS:
        for mth in METHODS:
            for metric in METRICS:
                raw, tf = get_block(skey, mth, metric)
                means = {"raw": raw[0], "boxcox": tf["boxcox"][0],
                         "histeq": tf["histeq"][0], "clahe": tf["clahe"][0]}
                win[max(means, key=means.get)] += 1
    return win


W = win_counts()
tbl_iou = metric_table("IoU", "\\emph{Intersection over Union} (IoU, \\%) de la clase grieta, "
    "media\\,$\\pm$\\,desv.\\ est.\\ sobre las 50 im\\'agenes de cada conjunto. Significancia "
    "del test pareado de permutaci\\'on \\emph{frente a Raw}.", "tab:iou")
tbl_dice = metric_table("Dice", "Coeficiente Dice/F1 (\\%) de la clase grieta, "
    "media\\,$\\pm$\\,desv.\\ est. Significancia frente a Raw.", "tab:dice")
tbl_prec = metric_table("Precision", "\\emph{Precision} (\\%) de la clase grieta, "
    "media\\,$\\pm$\\,desv.\\ est. Significancia frente a Raw.", "tab:prec")
tbl_rec = metric_table("Recall", "\\emph{Recall} (\\%) de la clase grieta, "
    "media\\,$\\pm$\\,desv.\\ est. Significancia frente a Raw.", "tab:rec")
tbl_avg = summary_table()
pair_he = pairwise_table("boxcox_vs_histeq",
    "Comparaci\\'on directa \\textbf{Box--Cox vs HistEq}: $\\Delta=$ media(Box--Cox)"
    "$-$media(HistEq) en puntos porcentuales, pareada imagen con imagen "
    "(positivo $\\Rightarrow$ Box--Cox mejor). Significancia del test de permutaci\\'on.",
    "tab:bc_he")
pair_cl = pairwise_table("boxcox_vs_clahe",
    "Comparaci\\'on directa \\textbf{Box--Cox vs CLAHE}: $\\Delta=$ media(Box--Cox)"
    "$-$media(CLAHE) en puntos porcentuales, pareada imagen con imagen "
    "(positivo $\\Rightarrow$ Box--Cox mejor). Significancia del test de permutaci\\'on.",
    "tab:bc_cl")

doc = r"""\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[margin=2.4cm]{geometry}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{float}
\graphicspath{{figures/}}
\setlength{\parskip}{0.5em}
\setlength{\parindent}{0pt}

\title{\vspace{-2.5em}Comparaci\'on de preprocesamientos (Box--Cox, ecualizaci\'on de
histograma y CLAHE) en la segmentaci\'on de grietas con LDA y QDA}
\author{}
\date{}

\begin{document}
\maketitle
\vspace{-3em}

Este informe compara tres t\'ecnicas de \textbf{prefiltrado por contraste} ---
\textbf{Box--Cox}, \textbf{ecualizaci\'on de histograma} (HistEq) y \textbf{CLAHE}
(\emph{Contrast Limited Adaptive Histogram Equalization})--- frente al uso de la
imagen \textbf{sin transformar} (\emph{Raw}), en la segmentaci\'on de grietas a nivel
de p\'ixel con dos clasificadores sensibles a la distribuci\'on: \textbf{LDA} y
\textbf{QDA}. El objetivo es cuantificar c\'omo cada preprocesamiento afecta la
calidad de la segmentaci\'on y el compromiso \emph{precision}--\emph{recall}.

\section*{1. Datos}
Conjunto de grietas de hormig\'on de \"Ozgenel \& Sorgu\c{c}. Cuatro conjuntos de
\textbf{50 im\'agenes} seg\'un la geometr\'ia de la fractura: \textbf{verticales},
\textbf{horizontales}, \textbf{oblicuas} (de izquierda a derecha) y \textbf{grandes}
(la grieta ocupa una porci\'on amplia de la imagen). Cada imagen RGB tiene su
m\'ascara binaria; todas se redimensionan a $256\times256$ y la m\'ascara se
binariza. En total, $4\times50=200$ im\'agenes etiquetadas.

\section*{2. Metodolog\'ia}
\textbf{Preprocesamientos.} Cada imagen se convierte a escala de grises
($0{,}299R+0{,}587G+0{,}114B$) y se obtiene un \'unico canal de intensidad bajo cada
condici\'on:
\begin{itemize}
  \item \textbf{Raw:} intensidad gris cruda normalizada a $[0,1]$ (\emph{baseline}).
  \item \textbf{Box--Cox:} transformaci\'on de Box--Cox con $\lambda$ estimado por
  m\'axima verosimilitud \textbf{por imagen}, seguida de \emph{histogram stretching}
  a $[0,1]$.
  \item \textbf{HistEq:} ecualizaci\'on \emph{global} del histograma
  (\texttt{cv2.equalizeHist}).
  \item \textbf{CLAHE:} ecualizaci\'on adaptativa con l\'imite de contraste
  ($\mathtt{clipLimit}=2{,}0$, mosaico $8\times8$).
\end{itemize}
Las transformaciones son \textbf{por imagen} (cada una con sus propios par\'ametros).

\textbf{Clasificadores.} LDA y QDA a nivel de p\'ixel; la \emph{feature} es la
intensidad del p\'ixel. \textbf{Protocolo por imagen}: para cada imagen se entrena con
una muestra estratificada de hasta $6000$ de sus p\'ixeles (\textbf{los mismos
\'indices en las cuatro condiciones}, de modo que la comparaci\'on es pareada) y se
predice la imagen completa.

\textbf{M\'etricas} (clase grieta): IoU, Dice/F1, \emph{Precision} y \emph{Recall}.
Se reporta \textbf{media\,$\pm$\,desviaci\'on est\'andar sobre las 50 im\'agenes} de
cada conjunto. La \emph{Accuracy} se omite por ser enga\~nosa (la grieta es
$\sim$1\,\% de los p\'ixeles).

\textbf{Significancia.} Cada transformaci\'on se compara \textbf{contra Raw} mediante
un \textbf{test pareado de permutaci\'on} (sign-flip) y el de Wilcoxon, pareando
imagen con imagen, m\'as \textbf{intervalos de confianza al 95\,\% por bootstrap}
(remuestreando \emph{im\'agenes}). Significancia: $^{*}p<0{,}05$, $^{**}p<0{,}01$,
$^{***}p<0{,}001$. En las tablas, \textbf{en negrita} el mejor valor de cada fila.

\section*{3. Resultados}
""" + tbl_avg + "\n\n" + tbl_iou + "\n\n" + tbl_dice + "\n\n" + tbl_prec + "\n\n" + tbl_rec + r"""

\medskip
\textbf{Comparaci\'on directa entre transformaciones.} Las tablas anteriores comparan
cada preprocesamiento contra \emph{Raw}. Las dos siguientes contrastan las
transformaciones \textbf{entre s\'i} (pareando imagen con imagen), que es la
comparaci\'on relevante para decidir entre ellas.

""" + pair_he + "\n\n" + pair_cl + r"""

\textbf{Box--Cox supera a HistEq} de forma masiva y significativa en IoU, Dice y
\emph{precision} en los ocho casos (entre $+15$ y $+53$ puntos, $p<0{,}001$), lo que
confirma que HistEq es el preprocesamiento m\'as d\'ebil; HistEq solo iguala o supera
en \emph{recall} de manera inconsistente. \textbf{Frente a CLAHE} las diferencias son
mucho menores: Box--Cox tiende a mayor \emph{precision} (y a mayor IoU/Dice en grietas
horizontales), mientras que CLAHE consigue mayor \emph{recall}; en IoU/Dice ambos
resultan \textbf{estad\'isticamente equivalentes} en la mayor\'ia de los conjuntos
(diferencias peque\~nas y no significativas, salvo en horizontales, donde gana
Box--Cox).

\section*{4. Lectura de los resultados}
El conteo de ``mejor por fila'' sobre las 32 combinaciones (4 conjuntos $\times$ 2
clasificadores $\times$ 4 m\'etricas) es: \textbf{Raw """ + str(W["raw"]) + r"""},
CLAHE """ + str(W["clahe"]) + r""", HistEq """ + str(W["histeq"]) + r""" y Box--Cox
""" + str(W["boxcox"]) + r""". Las conclusiones principales:
\begin{itemize}
  \item \textbf{Ninguna transformaci\'on mejora la calidad global de forma
  consistente.} \emph{Raw} es el mejor en IoU, Dice y \emph{precision} en casi todos
  los casos. Las tres t\'ecnicas siguen el patr\'on cl\'asico: \textbf{suben el
  \emph{recall} a costa de la \emph{precision}}.
  \item \textbf{HistEq es, con diferencia, la peor.} La ecualizaci\'on global
  destruye IoU, Dice y \emph{precision} (la \emph{precision} cae a $\sim$10--13\,\%,
  todas con $p<0{,}001$) y es \textbf{muy inestable}: la desviaci\'on del \emph{recall}
  entre im\'agenes ronda el 35\,\% (frente a 11--19\,\% en las dem\'as), porque satura
  el contraste y a menudo etiqueta como grieta gran parte del fondo.
  \item \textbf{CLAHE y Box--Cox son comparables y mucho m\'as suaves.} CLAHE entrega
  el \emph{recall} m\'as alto y estable (mejor detecci\'on); Box--Cox conserva algo
  mejor la \emph{precision}. Ambas son el preprocesamiento de elecci\'on si la
  prioridad es \textbf{no pasar por alto la fractura}. No obstante, el promedio oculta
  variabilidad por imagen: \textbf{en algunos casos Box--Cox supera claramente a CLAHE}
  en calidad (IoU/Dice), recuperando mejor la red completa de la grieta
  (v\'ease la Figura~\ref{fig:bcwin}).
  \item \textbf{El clasificador importa.} En \textbf{QDA}, CLAHE y Box--Cox apenas
  alteran IoU/Dice (cambios peque\~nos, a menudo no significativos): son ``seguras''
  para ganar \emph{recall} sin sacrificar calidad. En \textbf{LDA}, en cambio, todas
  las transformaciones reducen IoU/Dice de forma significativa.
  \item \textbf{\'Unico caso favorable en calidad:} QDA con Box--Cox en grietas
  \emph{horizontales} (IoU $+2{,}4$, Dice $+3{,}5$; $p<0{,}05$), donde el aumento de
  \emph{recall} compensa la p\'erdida de \emph{precision}.
\end{itemize}

\section*{5. Ejemplos}
Un ejemplo por tipo de grieta (clasificador QDA). De izquierda a derecha: imagen RGB,
m\'ascara real y segmentaci\'on bajo Raw, Box--Cox, HistEq y CLAHE. Se aprecia c\'omo
HistEq \textbf{inunda} la predicci\'on de falsos positivos (\emph{recall} alto pero
\emph{precision} muy baja), mientras CLAHE y Box--Cox engrosan moderadamente la grieta
respecto de Raw. La \'ultima figura (con t\'itulos en IoU/Dice) muestra un caso en que
\textbf{Box--Cox supera claramente a CLAHE} en calidad, recordando que el promedio
oculta variabilidad por imagen.

\begin{figure}[H]\centering
\includegraphics[width=\linewidth]{preproc_Vertical.png}
\caption{Grieta vertical.}
\end{figure}

\begin{figure}[H]\centering
\includegraphics[width=\linewidth]{preproc_Horizontal.png}
\caption{Grieta horizontal.}
\end{figure}

\begin{figure}[H]\centering
\includegraphics[width=\linewidth]{preproc_Left.png}
\caption{Grieta oblicua.}
\end{figure}

\begin{figure}[H]\centering
\includegraphics[width=\linewidth]{preproc_Big.png}
\caption{Grieta grande: HistEq satura la imagen de falsos positivos.}
\end{figure}

\begin{figure}[H]\centering
\includegraphics[width=\linewidth]{preproc_BoxCoxWin.png}
\caption{Caso favorable a Box--Cox frente a CLAHE (grieta grande, QDA; t\'itulos en
IoU/Dice). Box--Cox recupera la red completa de la grieta y \textbf{supera tanto a
CLAHE como a Raw} en calidad, mientras CLAHE queda m\'as fragmentada y ruidosa. Ilustra
que, pese a su comportamiento medio similar, ambos preprocesamientos no son
intercambiables imagen a imagen.}\label{fig:bcwin}
\end{figure}

\section*{6. Conclusi\'on}
Sobre 200 im\'agenes con medidas de incertidumbre y tests pareados, \textbf{ninguno de
los tres prefiltrados mejora la calidad global de la segmentaci\'on respecto de usar la
imagen sin transformar}; todos intercambian \emph{precision} por \emph{recall}. La
\textbf{ecualizaci\'on global de histograma (HistEq) es claramente perjudicial} y debe
descartarse. \textbf{CLAHE} y \textbf{Box--Cox} son alternativas suaves y de efecto
similar: \'utiles cuando la prioridad es \textbf{maximizar la detecci\'on}
(\emph{recall}), especialmente con \textbf{QDA}, donde no degradan apreciablemente
IoU/Dice. Si la prioridad es una delimitaci\'on fina de la grieta (mayor IoU/Dice),
conviene usar la imagen \textbf{Raw} o combinar CLAHE/Box--Cox con un
post-procesamiento que recupere \emph{precision}.

\end{document}
"""

os.makedirs(REPORT_DIR, exist_ok=True)
out = os.path.join(REPORT_DIR, "informe_preproc.tex")
with open(out, "w") as f:
    f.write(doc)
print("Escrito", out)
