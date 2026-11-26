"""Estilo único dos gráficos do relatório: traço fino, grade discreta, paleta categórica fixa."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

RAIZ = Path(__file__).resolve().parents[3]
FIGURAS = RAIZ / "reports" / "figures"
RESULTADOS = RAIZ / "reports" / "resultados"

# Ordem fixa (validada para daltonismo); a cor segue a entidade, nunca a posição.
SERIE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TINTA, TINTA_2, MUDA, GRADE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
AZUL_SEQ = LinearSegmentedColormap.from_list(
    "azul", ["#f7f9fc", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)


def estilo() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "font.size": 10.5,
            "axes.edgecolor": GRADE,
            "axes.labelcolor": TINTA_2,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlecolor": TINTA,
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRADE,
            "grid.linewidth": 0.8,
            "xtick.color": TINTA_2,
            "ytick.color": TINTA_2,
            "lines.linewidth": 2,
            "legend.frameon": False,
            "figure.facecolor": "white",
        }
    )


def salvar(fig, nome: str) -> Path:
    FIGURAS.mkdir(parents=True, exist_ok=True)
    caminho = FIGURAS / nome
    fig.savefig(caminho, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return caminho
