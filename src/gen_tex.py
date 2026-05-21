# -*- coding: utf-8 -*-
"""Genera report/informe_boxcox.tex a partir de results/summary_stats.csv.
Requiere las figuras en report/figures/ (ver src/make_figures.py)."""
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root (parent of src/)
RESULTS_DIR = os.path.join(ROOT, "results")
REPORT_DIR = os.path.join(ROOT, "report")

d = pd.read_csv(os.path.join(RESULTS_DIR, "summary_stats.csv"))
lam = pd.read_csv(os.path.join(RESULTS_DIR, "results_ml.csv"))
lam = lam[lam.condition == "BC"].groupby("set").lam.mean().round(2).to_dict()

SETS = [("Vertical", "Verticales"), ("Horizontal", "Horizontales"),
        ("Left", "Oblicuas (izq.\\,$\\to$\\,der.)"), ("Big", "Grandes")]
METHODS = ["SVM", "LightGBM", "KNN", "LDA", "QDA"]


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


def cell(m):  # mean +- std
    return f"${m.iloc[0]:.1f}\\pm{m.iloc[1]:.1f}$"


def get(s, method, metric):
    r = d[(d.set == s) & (d.method == method) & (d.metric == metric)]
    return r.iloc[0]


def two_metric_table(metric_a, metric_b, caption, label):
    out = []
    out.append("\\begin{table}[H]\\centering\\small")
    out.append(f"\\caption{{{caption}}}\\label{{{label}}}")
    out.append("\\begin{tabular}{l ccc ccc}")
    out.append("\\toprule")
    out.append(f"M\\'etodo & \\multicolumn{{3}}{{c}}{{{metric_a} (\\%)}} & "
               f"\\multicolumn{{3}}{{c}}{{{metric_b} (\\%)}} \\\\")
    out.append("\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}")
    out.append(" & sin BC & con BC & $\\Delta$ & sin BC & con BC & $\\Delta$ \\\\")
    out.append("\\midrule")
    for skey, sname in SETS:
        out.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{sname}}} "
                   f"($\\bar\\lambda={lam[skey]:.2f}$)}} \\\\")
        for mth in METHODS:
            ra = get(skey, mth, metric_a)
            rb = get(skey, mth, metric_b)
            out.append(
                f"\\quad {mth} & {cell(ra[['noBC_mean','noBC_std']])} & "
                f"{cell(ra[['BC_mean','BC_std']])} & "
                f"${ra['diff_mean']:+.1f}${stars(ra.p_perm)} & "
                f"{cell(rb[['noBC_mean','noBC_std']])} & "
                f"{cell(rb[['BC_mean','BC_std']])} & "
                f"${rb['diff_mean']:+.1f}${stars(rb.p_perm)} \\\\")
        out.append("\\addlinespace")
    out[-1] = "\\bottomrule"
    out.append("\\end{tabular}")
    out.append("\\end{table}")
    return "\n".join(out)


tbl1 = two_metric_table("IoU", "Dice", "Calidad global de la segmentaci\\'on de la grieta: "
    "IoU y Dice (F1), media\\,$\\pm$\\,desv.\\ est.\\ sobre las 50 im\\'agenes de cada conjunto. "
    "$\\Delta=$ con\\,$-$\\,sin Box--Cox (puntos porcentuales).", "tab:iou")
tbl2 = two_metric_table("Precision", "Recall", "Compromiso \\emph{precision}--\\emph{recall} "
    "para la clase grieta, media\\,$\\pm$\\,desv.\\ est.\\ sobre 50 im\\'agenes. "
    "$\\Delta=$ con\\,$-$\\,sin Box--Cox.", "tab:pr")

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

\title{\vspace{-2.5em}Efecto de la transformaci\'on de Box--Cox en la segmentaci\'on de grietas}
\author{}
\date{}

\begin{document}
\maketitle
\vspace{-3em}

Este informe resume el efecto de la transformaci\'on de Box--Cox como prefiltrado en la
segmentaci\'on de grietas, evaluado sobre cuatro conjuntos de im\'agenes seg\'un la geometr\'ia
de la fractura. Se reportan los datos, la metodolog\'ia y los resultados, con \'enfasis en
c\'omo la transformaci\'on \textbf{mejora la capacidad de detecci\'on de la grieta
(\emph{recall})} y en el compromiso resultante con la \emph{precision}.

\section*{1. Datos}
Utilic\'e el conjunto de grietas de hormig\'on de \"Ozgenel \& Sorgu\c{c}. Seleccion\'e
\textbf{cuatro conjuntos de 50 im\'agenes} seg\'un la geometr\'ia de la fractura:
\textbf{verticales}, \textbf{horizontales}, \textbf{oblicuas} (de izquierda a derecha) y
\textbf{grandes} (la grieta ocupa una porci\'on amplia de la imagen). Cada imagen RGB tiene su
m\'ascara binaria (grieta \emph{vs.} fondo). Todas se redimensionaron a $256\times256$ y la
m\'ascara se binariz\'o. En total, $4\times50=200$ im\'agenes etiquetadas.

\section*{2. Metodolog\'ia}
\textbf{Prefiltrado Box--Cox.} Cada imagen se convierte a escala de grises
($0{,}299R+0{,}587G+0{,}114B$), se le aplica la transformaci\'on de Box--Cox con $\lambda$
estimado por m\'axima verosimilitud y luego un \emph{histogram stretching} a $[0,1]$. La
transformaci\'on es \textbf{por imagen} (cada una con su propio $\lambda$). Comparo dos
condiciones: \textbf{sin} Box--Cox (intensidad gris cruda) y \textbf{con} Box--Cox.

\textbf{Clasificadores.} Cinco m\'etodos cl\'asicos a nivel de p\'ixel: SVM (RBF), LightGBM,
KNN, LDA y QDA. La \emph{feature} es la intensidad del p\'ixel. \textbf{Protocolo por imagen}:
para cada imagen se entrena con una muestra estratificada de sus p\'ixeles y se predice la
imagen completa, siguiendo el protocolo del art\'iculo original.

\textbf{M\'etricas} (clase grieta): IoU, Dice/F1, \emph{Precision} y \emph{Recall}. Reporto
\textbf{media\,$\pm$\,desviaci\'on est\'andar sobre las 50 im\'agenes} de cada conjunto.
La \emph{Accuracy} se omite por ser enga\~nosa (la grieta es $\sim$1\,\% de los p\'ixeles).

\textbf{Significancia.} Para cada comparaci\'on con/sin Box--Cox uso un \textbf{test pareado de
permutaci\'on} (sign-flip) y el de Wilcoxon, ambos pareando imagen con imagen, m\'as
\textbf{intervalos de confianza al 95\,\% por bootstrap}. El bootstrap es v\'alido aqu\'i porque
se remuestrean \emph{im\'agenes} (observaciones independientes), no p\'ixeles de una misma
imagen (que presentan autocorrelaci\'on espacial). Significancia: $^{*}p<0{,}05$,
$^{**}p<0{,}01$, $^{***}p<0{,}001$.

\section*{3. Resultados}
""" + tbl1 + "\n\n" + tbl2 + r"""

\section*{4. Efecto de la transformaci\'on: mejora del \emph{recall}}
El efecto de Box--Cox es \textbf{sistem\'atico y coherente en los cuatro conjuntos}: al estirar
el contraste de los p\'ixeles oscuros de la grieta, la transformaci\'on \textbf{realza la
fractura respecto del fondo} y el clasificador \textbf{reconoce m\'as p\'ixeles que realmente
son grieta}. En consecuencia, el \textbf{\emph{recall} aumenta de forma significativa} en todos
los conjuntos (menos falsos negativos): mejora la capacidad de \emph{detectar} la fractura. El
costo es una ca\'ida de la \emph{precision} (m\'as fondo etiquetado como grieta), por lo que el
resultado neto es el cl\'asico \textbf{compromiso \emph{precision}--\emph{recall}}: se gana
sensibilidad de detecci\'on a cambio de una localizaci\'on m\'as gruesa de la grieta.

El tama\~no del efecto depende del m\'etodo:
\begin{itemize}
  \item \textbf{LDA y QDA} (sensibles a la distribuci\'on, pues asumen normalidad) muestran el
  efecto m\'as marcado: el \emph{recall} sube entre $+5$ y $+12$ puntos, pero la
  \emph{precision} cae a\'un m\'as (entre $-9$ y $-22$ puntos). Como la p\'erdida de
  \emph{precision} domina, el \textbf{IoU y el Dice empeoran}.
  \item \textbf{SVM} muestra un cambio peque\~no en la misma direcci\'on ($\sim+0{,}5$ de
  \emph{recall}, $\sim-2$ de \emph{precision}); IoU/Dice bajan levemente.
  \item \textbf{LightGBM y KNN} son pr\'acticamente \textbf{invariantes} ($\Delta\approx0$), como
  cabe esperar de m\'etodos insensibles a transformaciones mon\'otonas de una sola
  \emph{feature}.
\end{itemize}

\section*{5. Ejemplos}
Un ejemplo por tipo de grieta (clasificador QDA). De izquierda a derecha: imagen RGB, m\'ascara
real y segmentaci\'on \emph{sin} y \emph{con} Box--Cox. Con la transformaci\'on el modelo
reconoce m\'as p\'ixeles de grieta (mayor \emph{recall}); en el caso oblicuo, sin Box--Cox no se
detecta nada y la transformaci\'on lo \emph{rescata}.

\begin{figure}[H]\centering
\includegraphics[width=\linewidth]{ej_Vertical.png}
\caption{Grieta vertical.}
\end{figure}

\begin{figure}[H]\centering
\includegraphics[width=\linewidth]{ej_Horizontal.png}
\caption{Grieta horizontal.}
\end{figure}

\begin{figure}[H]\centering
\includegraphics[width=\linewidth]{ej_Left.png}
\caption{Grieta oblicua (caso de rescate: sin Box--Cox no se detecta nada).}
\end{figure}

\begin{figure}[H]\centering
\includegraphics[width=\linewidth]{ej_Big.png}
\caption{Grieta grande.}
\end{figure}

\section*{6. Resultados por tipo de grieta}
\begin{itemize}
  \item \textbf{Verticales:} Box--Cox aumenta el \emph{recall} de QDA ($+8{,}1$) y LDA ($+5{,}5$),
  pero la ca\'ida de \emph{precision} ($-13{,}3$ y $-15{,}6$) reduce el IoU (LDA $-8{,}4$;
  QDA $-1{,}4$, no significativo). El resto sin cambios relevantes.
  \item \textbf{Horizontales:} \emph{\'Unico caso favorable.} En QDA, el fuerte aumento de
  \emph{recall} ($+9{,}4$) compensa la p\'erdida de \emph{precision} y el \textbf{IoU sube
  $+2{,}4$ y el Dice $+3{,}5$} (significativos, $p<0{,}05$). LDA sigue empeorando ($-5{,}1$).
  \item \textbf{Oblicuas:} comportamiento similar a verticales; LDA es el m\'as perjudicado
  (IoU $-11{,}2$, \emph{precision} $-20{,}1$) pese a ganar \emph{recall} ($+5{,}1$).
  \item \textbf{Grandes:} el \textbf{caso m\'as desfavorable}. LDA (IoU $-11{,}3$) y QDA
  (IoU $-4{,}4$) caen claramente: aunque el \emph{recall} sube mucho (QDA $+11{,}8$), la
  \emph{precision} se desploma ($-21{,}8$ y $-19{,}9$).
\end{itemize}

\section*{7. Conclusi\'on}
Sobre un conjunto de prueba amplio (50 im\'agenes por tipo, con medidas de incertidumbre y tests
pareados), Box--Cox \textbf{mejora la detecci\'on de la grieta}: aumenta el \emph{recall} de
forma significativa en los cuatro tipos, especialmente en los clasificadores sensibles a la
distribuci\'on (LDA y QDA), e incluso \textbf{rescata} casos en que sin la transformaci\'on el
modelo no detecta nada. El contrapeso es una p\'erdida de \emph{precision}: la regi\'on predicha
se engrosa e incorpora fondo, por lo que en agregado el IoU y el Dice pueden bajar (salvo QDA en
grietas horizontales, donde el balance es favorable). La transformaci\'on es, por tanto,
especialmente \'util cuando la prioridad es \textbf{no pasar por alto la fractura} (maximizar
\emph{recall}); si se requiere una delimitaci\'on m\'as fina, conviene combinarla con un
post-procesamiento que recupere \emph{precision}. Los m\'etodos basados en \'arboles o vecinos
(LightGBM, KNN) son insensibles a la transformaci\'on.

\end{document}
"""

os.makedirs(REPORT_DIR, exist_ok=True)
out = os.path.join(REPORT_DIR, "informe_boxcox.tex")
with open(out, "w") as f:
    f.write(doc)
print("Escrito", out)
