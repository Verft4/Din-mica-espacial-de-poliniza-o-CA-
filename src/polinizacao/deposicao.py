"""Calibração da deposição de pólen por visita, relativa a *Apis*, em função do tamanho do corpo.

Dados: Földesi et al. (2020), meta-análise com razões-log de deposição por visita (abelha / *Apis*), e
Cecala et al. (2020), deposição em melancia (checagem externa). Só o efeito *relativo* do tamanho é calibrado;
o limiar de polinização não é, pois esses dados não contêm pegamento de fruto ou semente.

A meta-regressão é de efeitos aleatórios, com tau² por momentos (Viechtbauer) e IC por bootstrap de estudos
(os efeitos de um mesmo estudo compartilham o comparador *Apis* e não são independentes). A validação deixa um
estudo inteiro de fora.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- dados
@dataclass(frozen=True)
class Observacoes:
    estudo: np.ndarray
    ordem: np.ndarray
    taxon: np.ndarray
    cultura: np.ndarray       # True se cultivo
    bl: np.ndarray            # razão de comprimento do corpo (NaN se faltante)
    itd: np.ndarray           # razão da distância intertegular (NaN se faltante)
    lnr: np.ndarray
    v_lnr: np.ndarray
    lnr_delta: np.ndarray
    v_delta: np.ndarray

    def __len__(self) -> int:
        return len(self.estudo)

    def subconjunto(self, mascara: np.ndarray) -> "Observacoes":
        return Observacoes(*(getattr(self, f)[mascara] for f in self.__dataclass_fields__))


def _num(s: str) -> float:
    s = s.strip()
    return float("nan") if s in ("", "NA") else float(s)


def carregar_foldesi(caminho: str | Path) -> Observacoes:
    """Lê ``Metadata.csv`` (separador ';', codificação Windows-1252)."""
    with open(caminho, newline="", encoding="latin-1") as f:
        linhas = [r for r in csv.DictReader(f, delimiter=";") if r.get("studyID")]
    col = lambda k: [r[k] for r in linhas]  # noqa: E731
    num = lambda k: np.array([_num(x) for x in col(k)])  # noqa: E731
    return Observacoes(
        estudo=np.array(col("studyID")), ordem=np.array(col("Poll_order")), taxon=np.array(col("Pollinator")),
        cultura=np.array([x == "crop" for x in col("type")]), bl=num("BLratio"), itd=num("ITD_ratio"),
        lnr=num("lnR"), v_lnr=num("V_lnR"), lnr_delta=num("lnRdelta"), v_delta=num("Var_lnRdelta"),
    )


def subconjunto_abelhas(obs: Observacoes, tamanho: str = "itd") -> tuple[Observacoes, dict]:
    """Hymenoptera com a medida de tamanho presente e positiva. Informa o que foi excluído."""
    x = getattr(obs, tamanho)
    abelha = obs.ordem == "Hymenoptera"
    ok = abelha & np.isfinite(x) & (x > 0) & np.isfinite(obs.lnr_delta) & np.isfinite(obs.v_delta) & (obs.v_delta > 0)
    info = {
        "n_total": len(obs), "n_himenoptera": int(abelha.sum()), "n_usado": int(ok.sum()),
        "n_estudos": int(len(np.unique(obs.estudo[ok]))),
        "excluidos_tamanho_ausente": int((abelha & ~np.isfinite(x)).sum()),
        "excluidos_tamanho_nulo": int((abelha & np.isfinite(x) & (x <= 0)).sum()),
    }
    return obs.subconjunto(ok), info


# --------------------------------------------------------------------------- meta-regressão
@dataclass(frozen=True)
class Ajuste:
    beta: np.ndarray
    tau2: float
    cov: np.ndarray
    nomes: tuple[str, ...]


def _matriz(x: np.ndarray, intercepto: bool) -> np.ndarray:
    return np.column_stack([np.ones_like(x), x]) if intercepto else x.reshape(-1, 1)


def ajustar_metarregressao(x, y, v, intercepto: bool = True) -> Ajuste:
    """Efeitos aleatórios; ``x`` já em ln(razão de tamanho). ``intercepto=False`` ancora em *Apis* (x = 0 → 0)."""
    x, y, v = (np.asarray(a, dtype=float) for a in (x, y, v))
    X = _matriz(x, intercepto)
    k, p = X.shape
    w0 = 1.0 / v
    A0 = X.T @ (w0[:, None] * X)
    b0 = np.linalg.solve(A0, X.T @ (w0 * y))
    q = float(np.sum(w0 * (y - X @ b0) ** 2))
    tr = np.trace(np.linalg.solve(A0, X.T @ ((w0 ** 2)[:, None] * X)))
    denom = float(w0.sum() - tr)
    tau2 = max(0.0, (q - (k - p)) / denom) if denom > 0 else 0.0
    w = 1.0 / (v + tau2)
    A = X.T @ (w[:, None] * X)
    beta = np.linalg.solve(A, X.T @ (w * y))
    nomes = ("intercepto", "inclinacao") if intercepto else ("inclinacao",)
    return Ajuste(beta=beta, tau2=tau2, cov=np.linalg.inv(A), nomes=nomes)


def prever(aj: Ajuste, x) -> tuple[np.ndarray, np.ndarray]:
    """Média prevista de lnR e variância da média prevista."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    X = _matriz(x, len(aj.beta) == 2)
    return X @ aj.beta, np.einsum("ij,jk,ik->i", X, aj.cov, X)


def bootstrap_estudos(estudo, x, y, v, intercepto: bool, n_boot: int, seed: int) -> np.ndarray:
    """Coeficientes sob reamostragem de estudos inteiros (com reposição)."""
    rng = np.random.default_rng(seed)
    estudos = np.unique(estudo)
    idx = {s: np.flatnonzero(estudo == s) for s in estudos}
    saida = []
    for _ in range(n_boot):
        sel = rng.choice(estudos, size=len(estudos), replace=True)
        ii = np.concatenate([idx[s] for s in sel])
        if len(np.unique(x[ii])) < 2 and intercepto:
            continue
        try:
            saida.append(ajustar_metarregressao(x[ii], y[ii], v[ii], intercepto).beta)
        except np.linalg.LinAlgError:
            continue
    return np.array(saida)


# --------------------------------------------------------------------------- validação por estudo
MODELOS = ("M0", "M1", "M2", "M3")


def validar_loso(estudo, x, y, v) -> dict:
    """Deixa cada estudo de fora; prevê seus efeitos com M0 (lnR=0), M1 (intercepto), M2 (reta), M3 (pela origem).

    Erro quadrático médio ponderado por 1/(V + tau2 de M1 no conjunto completo), igual para todos os modelos.
    Cobertura: fração de efeitos retidos dentro do intervalo preditivo de 95%.
    """
    estudo, x, y, v = (np.asarray(a) for a in (estudo, x, y, v))
    tau2_ref = ajustar_metarregressao(x, y, v, True).tau2
    tau2_m1 = _ajuste_intercepto(y, v).tau2
    w_ref = 1.0 / (v + tau2_m1)
    pred = {m: np.zeros(len(y)) for m in MODELOS}
    sd = {m: np.zeros(len(y)) for m in MODELOS}
    for s in np.unique(estudo):
        te = estudo == s
        tr = ~te
        # M1: média ponderada simples
        a1 = _ajuste_intercepto(y[tr], v[tr])
        a2 = ajustar_metarregressao(x[tr], y[tr], v[tr], True)
        a3 = ajustar_metarregressao(x[tr], y[tr], v[tr], False)
        pred["M0"][te] = 0.0
        sd["M0"][te] = np.sqrt(v[te] + tau2_m1)
        pred["M1"][te] = a1.beta[0]
        sd["M1"][te] = np.sqrt(v[te] + a1.tau2 + a1.cov[0, 0])
        for nome, aj in (("M2", a2), ("M3", a3)):
            m, var = prever(aj, x[te])
            pred[nome][te] = m
            sd[nome][te] = np.sqrt(v[te] + aj.tau2 + var)
    out = {"tau2_completo_m2": tau2_ref, "tau2_m1_completo": tau2_m1, "n": int(len(y)),
           "n_estudos": int(len(np.unique(estudo))), "w_ref": w_ref}
    for m in MODELOS:
        e = y - pred[m]
        out[m] = {
            "mse_ponderado": float(np.sum(w_ref * e ** 2) / np.sum(w_ref)),
            "rmse_simples": float(np.sqrt(np.mean(e ** 2))),
            "cobertura95": float(np.mean(np.abs(e) <= 1.96 * sd[m])),
            "erro": e,
        }
    return out


def _ajuste_intercepto(y, v) -> Ajuste:
    y, v = np.asarray(y, float), np.asarray(v, float)
    w0 = 1.0 / v
    mu0 = float(np.sum(w0 * y) / w0.sum())
    q = float(np.sum(w0 * (y - mu0) ** 2))
    k = len(y)
    denom = float(w0.sum() - np.sum(w0 ** 2) / w0.sum())
    tau2 = max(0.0, (q - (k - 1)) / denom) if denom > 0 else 0.0
    w = 1.0 / (v + tau2)
    mu = float(np.sum(w * y) / w.sum())
    return Ajuste(beta=np.array([mu]), tau2=tau2, cov=np.array([[1.0 / w.sum()]]), nomes=("intercepto",))


def diferenca_loso_ic(estudo, erro_a, erro_b, w, n_boot: int = 4000, seed: int = 7) -> tuple[float, float, float]:
    """Diferença de MSE ponderado (a - b) com IC 95% por bootstrap de estudos. Positivo = b melhor."""
    estudo = np.asarray(estudo)
    estudos = np.unique(estudo)
    num = np.array([np.sum(w[estudo == s] * (erro_a[estudo == s] ** 2 - erro_b[estudo == s] ** 2)) for s in estudos])
    den = np.array([np.sum(w[estudo == s]) for s in estudos])
    rng = np.random.default_rng(seed)
    dif = []
    for _ in range(n_boot):
        i = rng.integers(0, len(estudos), len(estudos))
        dif.append(num[i].sum() / den[i].sum())
    return float(num.sum() / den.sum()), float(np.quantile(dif, 0.025)), float(np.quantile(dif, 0.975))


# --------------------------------------------------------------------------- Cecala et al. (2020)
@dataclass(frozen=True)
class VisitasMelancia:
    grupo: np.ndarray
    genero: np.ndarray
    tratamento: np.ndarray
    graos: np.ndarray
    duracao: np.ndarray


def carregar_cecala(caminho: str | Path) -> VisitasMelancia:
    with open(caminho, newline="", encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    n = lambda k: np.array([float(r[k]) if r[k] != "" else np.nan for r in linhas])  # noqa: E731
    return VisitasMelancia(
        grupo=np.array([r["grupo"] for r in linhas]), genero=np.array([r["genero"] for r in linhas]),
        tratamento=np.array([r["tratamento"] for r in linhas]), graos=n("graos_estigma"), duracao=n("duracao_s"),
    )


def log_razao_medias(a, b, n_boot: int = 4000, seed: int = 11) -> tuple[float, float, float]:
    """ln(média(a)/média(b)) com IC 95% por bootstrap de visitas. a, b sem NaN."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    est = np.log(a.mean() / b.mean())
    bs = []
    for _ in range(n_boot):
        ma = rng.choice(a, len(a)).mean()
        mb = rng.choice(b, len(b)).mean()
        if ma > 0 and mb > 0:
            bs.append(np.log(ma / mb))
    return float(est), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


# --------------------------------------------------------------------------- ponte para o autômato
def multiplicador(tamanho_relativo: float, expoente: float) -> float:
    """Fator sobre a taxa de deposição: ``tamanho^expoente`` (igual a exp(b·ln s) da meta-regressão)."""
    return float(tamanho_relativo ** expoente)
