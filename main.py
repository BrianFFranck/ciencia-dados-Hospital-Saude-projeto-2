"""Pipeline reprodutível do Vídeo-Caso 2 - Anestesiologia.

Executar a partir desta pasta com: python main.py
Os dados brutos em data/ nunca são alterados.
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
FIG = OUT / "figuras"

# Regras operacionais explícitas. São filtros técnicos conservadores, não limites
# Valores extremos dentro destes limites são sinalizados,mas não eliminados automaticamente.
VALID_RANGE = {
    "FC": (0.0, 300.0),       # zero não é tratado como frequência observada utilizável
    "PAM": (0.0, 300.0),      # valores <= 0 não são utilizáveis como PAM
    "SpO2": (0.0, 100.0),    # percentagem não pode exceder 100%
}
PAM_CUTOFF = 65.0
COVERAGE_THRESHOLD = 0.80
UNIDADES_MONITORIZACAO = {"FC": "bpm", "PAM": "mmHg", "SpO2": "%"}
UNIDADES_LABORATORIO = {"Hemoglobina": "g/dL", "Creatinina": "mg/dL"}

REQUIRED = {
    "doentes": ["doente_id", "idade", "sexo", "altura_cm", "peso_kg"],
    "cirurgias": ["cirurgia_id", "doente_id", "especialidade", "abordagem",
                  "tipo_anestesia", "asa", "urgencia", "duracao_anestesia_min"],
    "laboratorio": ["cirurgia_id", "tempo_relativo_h", "analito", "valor", "unidade"],
    "monitorizacao": ["cirurgia_id", "tempo_min", "parametro", "valor", "unidade"],
}


def carregar_fontes():
    paths = {
        "doentes": DATA / "doentes.csv",
        "cirurgias": DATA / "cirurgias.csv",
        "laboratorio": DATA / "laboratorio.csv",
        "monitorizacao_A": DATA / "monitorizacao_A.csv",
        "monitorizacao_B": DATA / "monitorizacao_B.csv",
    }

    for nome, path in paths.items():            #clause guards
        if not path.exists():
            raise FileNotFoundError(
                f"Ficheiro obrigatório em falta: {path}"
            )

    dados = {
        nome: pd.read_csv(path)
        for nome, path in paths.items()
    }

    for nome in ["doentes", "cirurgias", "laboratorio"]:
        faltam = set(REQUIRED[nome]) - set(dados[nome].columns)

        if faltam:
            raise ValueError(
                f"{nome}: colunas obrigatórias em falta: {sorted(faltam)}"
            )

    for nome in ["monitorizacao_A", "monitorizacao_B"]:
        faltam = set(REQUIRED["monitorizacao"]) - set(dados[nome].columns)

        if faltam:
            raise ValueError(
                f"{nome}: colunas obrigatórias em falta: {sorted(faltam)}"
            )

    return dados


def auditar_fontes(dados):
    linhas = []
    fontes = {k: dados[k] for k in ["doentes", "cirurgias", "laboratorio"]}
    fontes["monitorizacao_A"] = dados["monitorizacao_A"]
    fontes["monitorizacao_B"] = dados["monitorizacao_B"]
    id_col = {"doentes": "doente_id", "cirurgias": "cirurgia_id",
              "laboratorio": "cirurgia_id", "monitorizacao_A": "cirurgia_id",
              "monitorizacao_B": "cirurgia_id"}
    for origem, df in fontes.items():
        idn = df[id_col[origem]].nunique(dropna=True)
        ndup = int(df.duplicated().sum())
        for col in df.columns:
            nmiss = int(df[col].isna().sum())
            linhas.append({
                "origem": origem, "variavel": col, "dtype": str(df[col].dtype),
                "n_linhas_fonte": len(df), "n_colunas_fonte": len(df.columns),
                "n_ids_distintos": int(idn), "n_duplicados_exatos_fonte": ndup,
                "n_observado": int(df[col].notna().sum()), "n_missing": nmiss,
                "percentagem_missing": round(100 * nmiss / len(df), 4) if len(df) else np.nan,
            })
    result = pd.DataFrame(linhas)
    result.to_csv(OUT / "auditoria_fontes.csv", index=False)
    return result


def auditar_duplicados(dados):
    """Analisa duplicados sem criar ficheiros auxiliares."""

    registos = []

    definicoes = [
        ("doentes", dados["doentes"], ["doente_id"]),
        ("cirurgias", dados["cirurgias"], ["cirurgia_id"]),
        (
            "laboratorio",
            dados["laboratorio"],
            ["cirurgia_id", "tempo_relativo_h", "analito"]
        ),
        (
            "monitorizacao_A",
            dados["monitorizacao_A"],
            ["cirurgia_id", "tempo_min", "parametro"]
        ),
        (
            "monitorizacao_B",
            dados["monitorizacao_B"],
            ["cirurgia_id", "tempo_min", "parametro"]
        ),
    ]

    for nome, df, chave in definicoes:
        exatos = int(df.duplicated().sum())

        sem_exatos = df.drop_duplicates()

        dup_log = sem_exatos[
            sem_exatos.duplicated(chave, keep=False)
        ]

        grupos = (
            int(dup_log.groupby(chave, dropna=False).ngroups)
            if not dup_log.empty else 0
        )

        registos.append({
            "fonte": nome,
            "chave_logica": "+".join(chave),
            "duplicados_exatos_excesso": exatos,
            "grupos_chave_logica_com_conteudo_distinto": grupos,
            "linhas_envolvidas_nesses_grupos": len(dup_log)
        })

    return pd.DataFrame(registos)


def normalizar_texto(df, colunas):
    df = df.copy()
    for c in colunas:
        if c in df:
            df[c] = df[c].astype("string").str.strip()
    return df


def normalizar_categorias(dados):
    doentes = normalizar_texto(dados["doentes"], ["sexo"])
    doentes["sexo"] = doentes["sexo"].str.casefold().map(
        {"feminino": "Feminino", "masculino": "Masculino"}
    ).fillna(doentes["sexo"])
    cir = normalizar_texto(dados["cirurgias"],
                           ["especialidade", "abordagem", "tipo_anestesia", "urgencia"])
    maps = {
        "especialidade": {"cirurgia geral": "Cirurgia Geral", "cirurgia torácica": "Cirurgia Torácica",
                           "ginecologia": "Ginecologia", "urologia": "Urologia"},
        "abordagem": {"aberta": "Aberta", "videolaparoscópica": "Videolaparoscópica", "robótica": "Robótica"},
        "tipo_anestesia": {"geral": "Geral", "raquidiana": "Raquidiana"},
        "urgencia": {"urgente": "Urgente", "programada": "Programada"},
    }
    for col, mp in maps.items():
        cir[col] = cir[col].str.casefold().map(mp).fillna(cir[col])
    lab = normalizar_texto(dados["laboratorio"], ["analito", "unidade"])
    lab["analito"] = lab["analito"].str.casefold().map(
        {"hemoglobina": "Hemoglobina", "creatinina": "Creatinina"}
    ).fillna(lab["analito"])
    mon = {}
    for name in ["monitorizacao_A", "monitorizacao_B"]:
        m = normalizar_texto(dados[name], ["parametro", "unidade"])
        m["parametro"] = m["parametro"].str.strip().str.upper().replace({"SPO2": "SpO2"})
        # Mantém a grafia canónica definida no enunciado.
        m["parametro"] = m["parametro"].replace({"FC": "FC", "PAM": "PAM", "SPO2": "SpO2"})
        mon[name] = m
    # Tipos anuláveis para campos discretos com possíveis missing values.
    doentes["idade"] = pd.to_numeric(doentes["idade"], errors="raise").astype("Int64")
    cir["asa"] = pd.to_numeric(cir["asa"], errors="raise").astype("Int64")
    cir["duracao_anestesia_min"] = pd.to_numeric(cir["duracao_anestesia_min"], errors="raise").astype("Int64")
    #
    anom_asa = cir.loc[
        cir.asa.notna() & ~cir.asa.between(1, 5),
        ["cirurgia_id", "asa"]
    ].copy()

    return doentes, cir, lab, mon, anom_asa



def remover_duplicados_seguros(df, nome, chave):
    """Remove só linhas integralmente iguais; falha se a chave conflitar."""
    antes = len(df)
    limpo = df.drop_duplicates().copy()
    repeticoes_exatas = antes - len(limpo)
    conflitos = limpo[limpo.duplicated(chave, keep=False)]
    if not conflitos.empty:
        n_grupos = conflitos.groupby(chave, dropna=False).ngroups
        raise ValueError(
            f"{nome}: {n_grupos} grupos repetem a chave lógica {chave} com conteúdo diferente. "
            "É necessária decisão clínica/documental antes de continuar."
        )
    return limpo, repeticoes_exatas


def validar_ids_fontes(dados, cirurgias):
    """Verifica a integridade referencial sem exportar CSV auxiliar."""

    ids_cir = set(cirurgias.cirurgia_id.dropna().astype(str))

    fontes = {
        "laboratorio": dados["laboratorio"],
        "monitorizacao_A": dados["monitorizacao_A"],
        "monitorizacao_B": dados["monitorizacao_B"],
    }

    linhas = []
    desconhecidos = []

    for nome, df in fontes.items():

        ids_fonte = set(df.cirurgia_id.dropna().astype(str))

        orfaos = sorted(ids_fonte - ids_cir)
        sem_registo = sorted(ids_cir - ids_fonte)

        linhas.append({
            "fonte": nome,
            "n_ids_fonte": len(ids_fonte),
            "n_ids_cirurgias": len(ids_cir),
            "n_ids_sem_cirurgia_correspondente": len(orfaos),
            "n_cirurgias_sem_registo_na_fonte": len(sem_registo),
            "ids_sem_correspondencia": ";".join(orfaos),
            "cirurgias_sem_registo": ";".join(sem_registo)
        })

        desconhecidos.extend((nome, x) for x in orfaos)

    ids_mon = (
        set(fontes["monitorizacao_A"].cirurgia_id.dropna().astype(str))
        |
        set(fontes["monitorizacao_B"].cirurgia_id.dropna().astype(str))
    )

    linhas.append({
        "fonte": "monitorizacao_A+B",
        "n_ids_fonte": len(ids_mon),
        "n_ids_cirurgias": len(ids_cir),
        "n_ids_sem_cirurgia_correspondente": len(ids_mon - ids_cir),
        "n_cirurgias_sem_registo_na_fonte": len(ids_cir - ids_mon),
        "ids_sem_correspondencia": ";".join(sorted(ids_mon - ids_cir)),
        "cirurgias_sem_registo": ";".join(sorted(ids_cir - ids_mon))
    })

    if desconhecidos:
        detalhe = ", ".join(
            f"{fonte}:{cid}" for fonte, cid in desconhecidos
        )

        raise ValueError(
            "Foram encontrados cirurgia_id sem correspondência: "
            + detalhe
        )

    return pd.DataFrame(linhas)


def auditar_categorias_e_unidades(dados):
    """Devolve as auditorias para serem incluídas no resumo."""

    linhas = []

    configuracao = [
        ("doentes", dados["doentes"], ["sexo"]),
        (
            "cirurgias",
            dados["cirurgias"],
            ["especialidade", "abordagem", "tipo_anestesia", "urgencia", "asa"]
        ),
        ("laboratorio", dados["laboratorio"], ["analito", "unidade"]),
        ("monitorizacao_A", dados["monitorizacao_A"], ["parametro", "unidade"]),
        ("monitorizacao_B", dados["monitorizacao_B"], ["parametro", "unidade"]),
    ]

    for fonte, df, colunas in configuracao:
        for coluna in colunas:

            cont = (
                df[coluna]
                .astype("string")
                .fillna("<NA>")
                .value_counts(dropna=False)
            )

            for categoria, n in cont.items():
                linhas.append({
                    "fonte": fonte,
                    "variavel": coluna,
                    "categoria_ou_unidade": str(categoria),
                    "n_registos": int(n)
                })

    auditoria_categorias = pd.DataFrame(linhas)

    validacoes = []

    for fonte in ["monitorizacao_A", "monitorizacao_B"]:
        df = dados[fonte]

        for parametro, unidade_esperada in UNIDADES_MONITORIZACAO.items():

            sub = df.loc[
                df.parametro.astype("string")
                .str.strip()
                .str.casefold()
                .eq(parametro.casefold())
            ]

            for unidade, n in (
                sub.unidade.astype("string")
                .str.strip()
                .value_counts(dropna=False)
                .items()
            ):
                validacoes.append({
                    "fonte": fonte,
                    "parametro_analito": parametro,
                    "unidade_observada": str(unidade),
                    "unidade_esperada": unidade_esperada,
                    "n_registos": int(n),
                    "coerente": str(unidade) == unidade_esperada
                })

    lab = dados["laboratorio"]

    for analito, unidade_esperada in UNIDADES_LABORATORIO.items():

        sub = lab.loc[
            lab.analito.astype("string")
            .str.strip()
            .str.casefold()
            .eq(analito.casefold())
        ]

        for unidade, n in (
            sub.unidade.astype("string")
            .str.strip()
            .value_counts(dropna=False)
            .items()
        ):
            validacoes.append({
                "fonte": "laboratorio",
                "parametro_analito": analito,
                "unidade_observada": str(unidade),
                "unidade_esperada": unidade_esperada,
                "n_registos": int(n),
                "coerente": str(unidade) == unidade_esperada
            })

    return auditoria_categorias, pd.DataFrame(validacoes)


def preparar_monitorizacao(mon_a, mon_b, cirurgias):
    a = mon_a.copy(); a["origem"] = "A"
    b = mon_b.copy(); b["origem"] = "B"
    combinado = pd.concat([a, b], ignore_index=True)
    if len(combinado) != len(a) + len(b):
        raise AssertionError("A concatenação não preservou o total de linhas.")
    if set(a.cirurgia_id.dropna()) & set(b.cirurgia_id.dropna()):
        warnings.warn("Existem cirurgia_id nos dois blocos; verificar sobreposição.")

    # Validação técnica preservando as colunas originais e registando motivos
    combinado["tempo_min_original"] = combinado["tempo_min"]
    combinado["tempo_min"] = pd.to_numeric(combinado["tempo_min"], errors="coerce")
    tempo_inteiro = combinado.tempo_min.notna() & np.isclose(combinado.tempo_min.fillna(0) % 1, 0)
    tempo_nao_negativo = combinado.tempo_min.ge(0)
    combinado["tempo_valido"] = tempo_inteiro & tempo_nao_negativo
    combinado["valor_original"] = combinado["valor"]
    combinado["valor"] = pd.to_numeric(combinado["valor"], errors="coerce")
    combinado["parametro_reconhecido"] = combinado.parametro.isin(VALID_RANGE)
    combinado["unidade_esperada"] = combinado.parametro.map(UNIDADES_MONITORIZACAO)
    combinado["unidade_coerente"] = combinado.unidade.astype("string").str.strip().eq(combinado.unidade_esperada)
    combinado["valor_utilizavel"] = False
    combinado["motivo_validacao"] = "valor em falta ou não numérico"

    for parametro, (lo, hi) in VALID_RANGE.items():
        if parametro == "SpO2":
            dentro = combinado.valor.between(lo, hi, inclusive="both")
        else:
            dentro = combinado.valor.gt(lo) & combinado.valor.le(hi)
        mask_base = combinado.parametro.eq(parametro)
        mask = mask_base & dentro & combinado.tempo_valido & combinado.unidade_coerente
        combinado.loc[mask, "valor_utilizavel"] = True
        combinado.loc[mask, "motivo_validacao"] = "válido segundo regra técnica operacional"
        combinado.loc[mask_base & combinado.valor.notna() & ~dentro, "motivo_validacao"] = f"valor fora da regra operacional [{lo}, {hi}]"
    combinado.loc[~combinado.parametro_reconhecido, "motivo_validacao"] = "parâmetro não reconhecido"
    combinado.loc[combinado.parametro_reconhecido & ~combinado.unidade_coerente, "motivo_validacao"] = "unidade incompatível ou em falta"
    combinado.loc[~combinado.tempo_valido, "motivo_validacao"] = "tempo inválido: deve ser inteiro e não negativo"
    combinado.loc[combinado.valor.isna(), "motivo_validacao"] = "valor em falta ou não numérico"

    # Auditoria por parâmetro e motivo, nenhuma observação inválida é apagada
    auditoria = combinado.groupby(["parametro", "motivo_validacao"], dropna=False).agg(
        n_registos=("valor", "size"), n_validos=("valor_utilizavel", "sum")
    ).reset_index()
    auditoria["n_invalidos"] = auditoria["n_registos"] - auditoria["n_validos"]


    chave = ["cirurgia_id", "tempo_min", "parametro"]
    # A chave lógica só pode ser única depois de remover duplicados integrais.
    combinado, n_removidos = remover_duplicados_seguros(combinado, "monitorização", chave)
    combinado["flag_extremo_iqr"] = False
    for p, grupo in combinado.groupby("parametro"):
        vals = grupo.loc[grupo.valor.notna(), "valor"]
        if len(vals):
            q1, q3 = vals.quantile([.25, .75]); iqr = q3 - q1
            mask = combinado.parametro.eq(p) & combinado.valor.notna() & (
                (combinado.valor < q1 - 1.5 * iqr) | (combinado.valor > q3 + 1.5 * iqr)
            )
            combinado.loc[mask, "flag_extremo_iqr"] = True

    indexed = combinado.set_index(chave + ["origem"]).sort_index()
    if not indexed.index.is_unique:
        raise AssertionError("MultiIndex não é único após auditoria.")
    # O pivot é feito apenas sobre parâmetros reconhecidos, sem colapsar categorias desconhecidas.
    reconhecidos = combinado.loc[combinado.parametro_reconhecido].copy()
    wide = reconhecidos.pivot(index=["cirurgia_id", "tempo_min"], columns="parametro", values="valor")
    wide.columns.name = None
    wide_reset = wide.reset_index()
    back = wide.stack(future_stack=True).dropna().rename("valor").reset_index()
    back.columns = ["cirurgia_id", "tempo_min", "parametro", "valor"]
    original = reconhecidos.loc[reconhecidos.valor.notna(), chave + ["valor"]].sort_values(chave).reset_index(drop=True)
    back = back.sort_values(chave).reset_index(drop=True)
    pd.testing.assert_frame_equal(original, back, check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-10)
    combinado.to_csv(OUT / "monitorizacao_preparada.csv", index=False)
    return combinado, wide_reset, n_removidos


def preparar_laboratorio(lab, cir_ids):
    lab = lab.copy()
    lab["tempo_relativo_h"] = pd.to_numeric(lab["tempo_relativo_h"], errors="coerce")
    lab["valor"] = pd.to_numeric(lab["valor"], errors="coerce")
    n_antes = len(lab)
    lab = lab.drop_duplicates().copy()
    n_exact = n_antes - len(lab)
    logicos = lab[lab.duplicated(["cirurgia_id", "tempo_relativo_h", "analito"], keep=False)].copy()
    if not logicos.empty:
        logicos["observacao"] = "chave lógica repetida; conteúdo distinto preservado para investigação"
    eleg = lab.loc[lab.tempo_relativo_h.between(-24, 0, inclusive="both")].copy()
    rows, conflitos = [], []
    for cid in cir_ids:
        row = {"cirurgia_id": cid}
        for analito, prefix in [("Hemoglobina", "hemoglobina"), ("Creatinina", "creatinina")]:
            # Selecionar primeiro o instante mais recente, incluindo medições com valor missing.
            sub = eleg.loc[(eleg.cirurgia_id == cid) & (eleg.analito == analito)].copy()
            if sub.empty:
                row[f"{prefix}_preoperatoria"] = np.nan
                row[f"{prefix}_tempo_relativo_h"] = np.nan
                row[f"{prefix}_estado"] = "sem medição elegível"
                continue
            tmin = sub.tempo_relativo_h.max()
            candidatos = sub.loc[sub.tempo_relativo_h.eq(tmin)].copy()
            candidatos_unicos = candidatos[["valor", "unidade"]].drop_duplicates()
            valores_validos = candidatos.valor.notna()
            if not valores_validos.all():
                estado = "missing no instante elegível mais recente"
                valor_selecionado = np.nan
            elif len(candidatos_unicos) > 1:
                estado = "ambíguo: valores/unidades diferentes no instante mais recente"
                valor_selecionado = np.nan
            else:
                unidade = str(candidatos_unicos.iloc[0].unidade).strip()
                unidade_esperada = UNIDADES_LABORATORIO[analito]
                if unidade != unidade_esperada:
                    estado = f"unidade incompatível: {unidade} (esperada {unidade_esperada})"
                    valor_selecionado = np.nan
                else:
                    estado = "selecionado: medição mais próxima de zero"
                    valor_selecionado = float(candidatos_unicos.iloc[0].valor)
            row[f"{prefix}_preoperatoria"] = valor_selecionado
            row[f"{prefix}_tempo_relativo_h"] = float(tmin)
            row[f"{prefix}_estado"] = estado
            if not valores_validos.all() or len(candidatos_unicos) > 1 or "unidade incompatível" in estado:
                for _, x in candidatos.iterrows():
                    conflitos.append({"cirurgia_id": cid, "analito": analito,
                                      "tempo_relativo_h": tmin, "valor": x.valor,
                                      "unidade": x.unidade, "decisao": estado})
        rows.append(row)
    prep = pd.DataFrame(rows)
    if prep.cirurgia_id.duplicated().any():
        raise AssertionError("A tabela laboratorial pivotada deve ter no máximo uma linha por cirurgia.")
    prep.to_csv(OUT / "laboratorio_preoperatorio.csv", index=False)
    n_grupos_conflito = len({(x["cirurgia_id"], x["analito"], x["tempo_relativo_h"]) for x in conflitos})
    return prep, eleg, n_exact, n_grupos_conflito


def calcular_metricas_hemodinamicas(mon, cirurgias):
    ids = cirurgias[["cirurgia_id", "duracao_anestesia_min"]].copy()
    valid = mon.loc[mon.valor_utilizavel].copy()
    pam = valid.loc[valid.parametro.eq("PAM")]
    fc = valid.loc[valid.parametro.eq("FC")]
    spo = valid.loc[valid.parametro.eq("SpO2")]
    pam_g = pam.groupby("cirurgia_id").valor.agg(
        n_minutos_pam_validos="count", pam_mediana="median", pam_minima="min")
    pam_low = pam.assign(_low=pam.valor.lt(PAM_CUTOFF)).groupby("cirurgia_id")._low.sum().rename("minutos_pam_lt65")
    fc_g = fc.groupby("cirurgia_id").valor.median().rename("fc_mediana")
    spo_g = spo.groupby("cirurgia_id").valor.min().rename("spo2_minima")
    resumo = ids.set_index("cirurgia_id").join([pam_g, pam_low, fc_g, spo_g]).reset_index()
    resumo["n_minutos_pam_validos"] = resumo.n_minutos_pam_validos.fillna(0).astype(int)
    resumo["minutos_pam_lt65"] = resumo.minutos_pam_lt65.fillna(0).astype(int)
    resumo["percentagem_pam_lt65"] = np.where(
        resumo.n_minutos_pam_validos.gt(0),
        100 * resumo.minutos_pam_lt65 / resumo.n_minutos_pam_validos, np.nan)
    resumo["cobertura_pam"] = np.where(
        resumo.duracao_anestesia_min.gt(0),
        resumo.n_minutos_pam_validos / resumo.duracao_anestesia_min, np.nan)
    resumo["cobertura_pam_suficiente"] = resumo.cobertura_pam.ge(COVERAGE_THRESHOLD)
    resumo["n_minutos_fc_validos"] = fc.groupby("cirurgia_id").size().reindex(resumo.cirurgia_id).fillna(0).astype(int).values
    resumo["n_minutos_spo2_validos"] = spo.groupby("cirurgia_id").size().reindex(resumo.cirurgia_id).fillna(0).astype(int).values
    if not (resumo["minutos_pam_lt65"] <= resumo["n_minutos_pam_validos"]).all():
        raise AssertionError("Minutos PAM <65 não podem exceder o número de PAM válidas.")
    if not resumo["percentagem_pam_lt65"].dropna().between(0, 100).all():
        raise AssertionError("Percentagem de PAM <65 fora do intervalo 0-100%.")
    if resumo["cobertura_pam"].dropna().gt(1).any():
        warnings.warn("Existem cirurgias com cobertura PAM >100%; verificar duração e grelha temporal.")
    resumo.to_csv(OUT / "resumo_hemodinamico.csv", index=False)
    return resumo


def integrar_dados(cir, doentes, lab_pre, hemo):
    if doentes.doente_id.duplicated().any():
        raise AssertionError("doente_id deve ser único na tabela de doentes.")
    if cir.cirurgia_id.duplicated().any():
        raise AssertionError("cirurgia_id deve ser único na tabela de cirurgias.")
    n0 = len(cir)
    integrado = cir.merge(doentes, on="doente_id", how="left", validate="many_to_one", indicator="_merge_doente")
    n1 = len(integrado)
    if n1 != n0:
        raise AssertionError("Merge doentes/cirurgias alterou inesperadamente o número de cirurgias.")
    if not integrado._merge_doente.eq("both").all():
        raise AssertionError("Existem cirurgias sem correspondência demográfica.")
    integrado = integrado.drop(columns="_merge_doente")
    for df, label in [(lab_pre, "laboratório"), (hemo, "hemodinâmica")]:
        if df.cirurgia_id.duplicated().any():
            raise AssertionError(f"A tabela {label} não tem uma linha por cirurgia.")
    integrado = integrado.merge(lab_pre, on="cirurgia_id", how="left", validate="one_to_one", indicator="_merge_lab")
    n2 = len(integrado)
    lab_match = integrado._merge_lab.value_counts().to_dict()
    integrado = integrado.drop(columns="_merge_lab")
    hemo_aux = hemo.drop(columns=["duracao_anestesia_min"])
    integrado = integrado.merge(hemo_aux, on="cirurgia_id", how="left",
                                validate="one_to_one", indicator="_merge_hemo")
    n3 = len(integrado)
    hemo_match = integrado._merge_hemo.value_counts().to_dict()
    integrado = integrado.drop(columns="_merge_hemo")
    if n3 != n0 or integrado.cirurgia_id.duplicated().any():
        raise AssertionError("Dataset final não preserva uma linha única por cirurgia.")
    integrado.to_csv(OUT / "dados_cirurgicos_integrados.csv", index=False)
    return integrado, {"linhas_cirurgias": n0, "apos_merge_doentes": n1,
                       "apos_merge_lab": n2, "apos_merge_hemo": n3,
                       "merge_lab": lab_match, "merge_hemo": hemo_match}


def gerar_visualizacoes(integrado, mon, resumo):
    sns.set_theme(context="notebook", style="whitegrid")
    # Figura 1: percentagem de missing data por variável e fonte
    aud = pd.read_csv(OUT / "auditoria_fontes.csv")
    aud = aud.loc[~aud.variavel.isin(["doente_id", "cirurgia_id"])].copy()
    ordem_fontes = ["doentes", "cirurgias", "laboratorio", "monitorizacao_A", "monitorizacao_B"]
    aud["ordem"] = aud.origem.map({f: i for i, f in enumerate(ordem_fontes)})
    aud = aud.sort_values(["ordem", "percentagem_missing"], ascending=[True, False])

    # Posição vertical de cada barra: um título por fonte e um espaço entre grupos.
    posicoes, titulos, pos = [], [], 0.0
    for origem, grupo in aud.groupby("origem", sort=False):
        titulos.append((pos, f"{origem} ({grupo.n_linhas_fonte.iloc[0]} linhas)"))
        pos += 1.0
        for _ in range(len(grupo)):
            posicoes.append(pos)
            pos += 1.0
        pos += 0.5
    aud["y"] = posicoes

    cores = dict(zip(ordem_fontes, sns.color_palette("colorblind", len(ordem_fontes))))
    xmax = aud.percentagem_missing.max()
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.barh(aud.y, aud.percentagem_missing, height=0.65, color=aud.origem.map(cores))
    for _, r in aud.iterrows():
        if r.n_missing:
            texto, cor = (
                f"{r.percentagem_missing:.2f}% ({r.n_missing} de {r.n_linhas_fonte})",
                "black",
            )
        else:
            texto, cor = "0% (completa)", "gray"
        ax.text(
            r.percentagem_missing + xmax * 0.012, r.y, texto, va="center", fontsize=8.5, color=cor
        )
    for y_titulo, texto in titulos:
        ax.text(xmax * 0.012, y_titulo, texto, va="center", fontsize=10, fontweight="bold")
    ax.set_yticks(aud.y, aud.variavel)
    ax.invert_yaxis()
    ax.set_xlim(0, xmax * 1.4)
    ax.grid(axis="y", visible=False)
    ax.set(
        xlabel="Valores em falta (% das linhas de cada fonte)",
        ylabel="",
        title="Qualidade dos dados: percentagem de valores em falta por fonte e variável",
    )
    fig.tight_layout()
    fig.savefig(FIG / "fig01_qualidade_dados.png", dpi=300)
    plt.close(fig)

    # Figura 2: episódio com melhor cobertura conjunta dos três parâmetros
    cov = mon.loc[mon.valor_utilizavel].groupby(["cirurgia_id", "parametro"]).size().unstack(fill_value=0)
    cov = cov.reindex(columns=["PAM", "FC", "SpO2"], fill_value=0)
    dur = integrado.set_index("cirurgia_id").duracao_anestesia_min
    cov_ratio = cov.div(dur, axis=0)
    cov_ratio["min_cobertura"] = cov_ratio[["PAM", "FC", "SpO2"]].min(axis=1)
    # Escolhe um episódio com cobertura mais adequada e maior carga observada de PAM baixa, para que a trajetória ilustre também o limiar de 65 mmHg
    candidatos = resumo.set_index("cirurgia_id").join(cov_ratio[["min_cobertura"]])
    candidatos = candidatos.loc[candidatos.min_cobertura.ge(COVERAGE_THRESHOLD)]
    escolhido = candidatos.percentagem_pam_lt65.idxmax()
    serie = mon.loc[(mon.cirurgia_id == escolhido) & mon.valor_utilizavel].pivot(
        index="tempo_min", columns="parametro", values="valor").sort_index()
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for ax, p, unit in zip(axes, ["PAM", "FC", "SpO2"], ["mmHg", "bpm", "%"]):
        if p in serie:
            ax.plot(serie.index, serie[p], linewidth=1.1)
        ax.set_ylabel(f"{p} ({unit})")
        ax.set_title(p, loc="left", fontsize=10)
    if "PAM" in serie and serie.PAM.notna().any():
        axes[0].axhline(PAM_CUTOFF, linestyle="--", linewidth=1, label="Limiar 65 mmHg")
        axes[0].legend(loc="best")
        idxmin = serie.PAM.idxmin(); vmin = serie.PAM.min()
        axes[0].annotate(f"Mínimo: {vmin:g} mmHg", xy=(idxmin, vmin), xytext=(8, 8),
                         textcoords="offset points", arrowprops={"arrowstyle": "->"})
    axes[-1].set_xlabel("Tempo desde o início da cirurgia (min)")
    fig.suptitle(f"Trajetória intraoperatória · {escolhido}", y=1.01)
    fig.tight_layout(); fig.savefig(FIG / "fig02_trajetoria_intraoperatoria.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "fig02_trajetoria_intraoperatoria.svg", bbox_inches="tight"); plt.close(fig)

    # Figura 3: histograma apenas com cobertura PAM >= 80% da duração registada
    d = resumo.loc[resumo.cobertura_pam_suficiente & resumo.percentagem_pam_lt65.notna(), "percentagem_pam_lt65"]
    fig, ax = plt.subplots(figsize=(9, 5))
    if len(d):
        ax.hist(d, bins=min(15, max(5, int(np.sqrt(len(d))))), edgecolor="black")
    else:
        ax.text(.5, .5, "Sem cirurgias que cumpram o critério de cobertura", ha="center", va="center", transform=ax.transAxes)
    ax.set(xlabel="Minutos observados com PAM <65 mmHg (%)", ylabel="Número de cirurgias",
           title=f"Distribuição da carga observada de PAM baixa (cobertura ≥{COVERAGE_THRESHOLD:.0%})")
    fig.tight_layout(); fig.savefig(FIG / "fig03_distribuicao_pam_baixa.png", dpi=300); plt.close(fig)

    # Figura 4: boxplot por especialidade e categorias com dimensão suficiente
    d4 = integrado.loc[
        integrado.cobertura_pam_suficiente & integrado.percentagem_pam_lt65.notna()
    ].copy()
    counts = d4.especialidade.value_counts()
    cats = counts[counts >= 5].index
    d4 = d4.loc[d4.especialidade.isin(cats)]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    if not d4.empty:
        ordem = (
            d4.groupby("especialidade").percentagem_pam_lt65.median().sort_values(ascending=False)
        )
        rotulos = [f"{e}\n(n={counts[e]})" for e in ordem.index]
        sns.boxplot(
            data=d4,
            x="especialidade",
            y="percentagem_pam_lt65",
            order=ordem.index,
            color="#c6dbef",
            width=0.55,
            showfliers=False,  # os outliers já aparecem como pontos do stripplot
            medianprops={"color": "black", "linewidth": 2},
            ax=ax,
        )
        np.random.seed(42)  # o jitter do stripplot é aleatório: seed para figura reprodutível
        sns.stripplot(
            data=d4,
            x="especialidade",
            y="percentagem_pam_lt65",
            order=ordem.index,
            color="#08306b",
            size=3.5,
            alpha=0.6,
            jitter=0.18,
            ax=ax,
        )
        for i, mediana in enumerate(ordem):
            ax.text(i + 0.3, mediana, f"mediana {mediana:.1f}%", va="center", fontsize=8.5)
        ax.set_xticks(range(len(ordem)), rotulos)
        ax.set_ylim(-1, d4.percentagem_pam_lt65.max() * 1.08)
    else:
        ax.text(
            0.5,
            0.5,
            "Sem grupos com dimensão suficiente",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    ax.set(
        xlabel="Especialidade",
        ylabel="Minutos observados com PAM <65 mmHg (%)",
        title=(
            "Carga observada de PAM baixa por especialidade\n"
            f"(cirurgias com cobertura PAM ≥{COVERAGE_THRESHOLD:.0%}; cada ponto = uma cirurgia)"
        ),
    )
    fig.tight_layout()
    fig.savefig(FIG / "fig04_comparacao_grupos.png", dpi=300)
    plt.close(fig)

    # Figura 5: Hb pré-operatória vs carga observada de PAM baixa
    d5 = integrado.loc[integrado.hemoglobina_preoperatoria.notna() &
                       integrado.percentagem_pam_lt65.notna() & integrado.cobertura_pam_suficiente]
    fig, ax = plt.subplots(figsize=(8, 5))
    if len(d5) >= 2:
        sns.regplot(data=d5, x="hemoglobina_preoperatoria", y="percentagem_pam_lt65",
                    scatter_kws={"alpha": .7}, line_kws={"linewidth": 1.5}, ax=ax)
    elif len(d5) == 1:
        sns.scatterplot(data=d5, x="hemoglobina_preoperatoria", y="percentagem_pam_lt65", ax=ax)
    else:
        ax.text(.5, .5, "Sem pares Hb/PAM com cobertura suficiente", ha="center", va="center", transform=ax.transAxes)
    ax.set(xlabel="Hemoglobina pré-operatória (g/dL)", ylabel="Minutos observados com PAM <65 mmHg (%)",
           title="Associação exploratória: Hemoglobina e PAM baixa")
    fig.tight_layout(); fig.savefig(FIG / "fig05_hb_pam.png", dpi=300); plt.close(fig)

    # Figura 6: distribuição normalizada de urgência por especialidade
    tab = pd.crosstab(integrado.especialidade, integrado.urgencia, normalize="index") * 100
    fig, ax = plt.subplots(figsize=(9, 5))
    tab.plot(kind="bar", stacked=True, ax=ax)
    ax.set(xlabel="Especialidade", ylabel="Cirurgias dentro da especialidade (%)",
           title="Composição de urgência por especialidade")
    ax.legend(title="Urgência", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout(); fig.savefig(FIG / "fig06_urgencia_especialidade.png", dpi=300); plt.close(fig)
    return escolhido, len(d), len(d4), len(d5)


def escrever_resumo(dados, doentes, cir, lab_pre, eleg, mon, anom_asa, resumo, integrado,
                    merge_info, removidos_mon, removidos_lab, conflitos_lab, figura_info):
    """Gera o relatório príncipal"""
    escolhido, n_cov, n_grupos, n_hb = figura_info
    lines = [
        "RESUMO DE INTEGRAÇÃO — VÍDEO-CASO 2 (ANESTESIOLOGIA)",
        "====================================================",
        f"Número de doentes únicos: {doentes.doente_id.nunique()}",
        f"Número de cirurgias únicas: {cir.cirurgia_id.nunique()}",
        f"Cirurgias com doente correspondente: {merge_info['apos_merge_doentes']}",
        f"Registos laboratoriais na janela [-24, 0] h (antes da seleção): {len(eleg)}",
        f"Cirurgias com Hemoglobina pré-operatória selecionada: {lab_pre.hemoglobina_preoperatoria.notna().sum()}",
        f"Cirurgias com Creatinina pré-operatória selecionada: {lab_pre.creatinina_preoperatoria.notna().sum()}",
        f"Grupos laboratoriais ambíguos no instante elegível mais recente: {conflitos_lab}",
        f"Observações de monitorização após concatenação e remoção de duplicados exatos: {len(mon)}",
        f"Duplicados exatos removidos da monitorização: {removidos_mon}",
        f"Observações sinalizadas como extremas pelo critério IQR (não removidas): {int(mon.flag_extremo_iqr.sum())}",
        f"Duplicados exatos removidos do laboratório: {removidos_lab}",
        f"Cirurgias com pelo menos uma PAM válida: {(resumo.n_minutos_pam_validos > 0).sum()}",
        f"Cirurgias com cobertura PAM >= {COVERAGE_THRESHOLD:.0%}: {n_cov}",
        f"Cirurgia_id único no dataset final: {integrado.cirurgia_id.is_unique}",
        f"Número final de linhas: {len(integrado)}",
        "",
        "VALIDAÇÃO DAS FONTES E JUNÇÕES",
        "- Integridade referencial: consultar auditoria_ids.csv; IDs sem cirurgia correspondente fazem o pipeline parar",
        "- Categorias e unidades: consultar auditoria_categorias.csv e auditoria_unidades.csv",
        "- Qualidade da monitorização por parâmetro/motivo: consultar auditoria_monitorizacao.csv",
        f"Linhas de cirurgias antes do merge: {merge_info['linhas_cirurgias']}",
        f"Após merge doentes: {merge_info['apos_merge_doentes']}",
        f"Após merge laboratório: {merge_info['apos_merge_lab']} | {merge_info['merge_lab']}",
        f"Após merge hemodinâmica: {merge_info['apos_merge_hemo']} | {merge_info['merge_hemo']}",
        "",
        "REGRAS E LIMITAÇÕES DOCUMENTADAS",
        "- Só foram removidas repetições integralmente iguais, após auditoria da chave lógica",
        "- Conflitos laboratoriais no tempo mais recente não foram resolvidos arbitrariamente; o valor selecionado fica missing e os candidatos são exportados em conflitos_laboratorio.csv",
        "- Janela laboratorial: -24 <= tempo_relativo_h <= 0; seleção do maior tempo (mais próximo de zero)",
        "- Regras técnicas de utilização: FC > 0 e <= 300; PAM > 0 e <= 300; SpO2 entre 0 e 100. São regras operacionais, não limites clínicos universais",
        "- Observações extremas pelo critério IQR são sinalizadas, não removidas automaticamente",
        "- CIR0096 tem ASA=6, fora do intervalo convencional 1-5;  o valor foi preservado e sinalizado em anomalias_categoricas.csv. Existem ainda 4 valores ASA em falta",
        f"Registos com ASA fora do intervalo convencional (1-5): {len(anom_asa)}",
        f"- Cobertura suficiente definida operacionalmente como n_minutos_pam_validos / duracao_anestesia_min >= {COVERAGE_THRESHOLD:.0%}",
        "- A percentagem PAM <65 usa exclusivamente minutos com PAM válida no denominador; lacunas não são consideradas normais",
        f"- Episódio usado na trajetória temporal: {escolhido}",
        "- Figura adicional: composição percentual de urgência por especialidade",
        "- Não se fazem inferências causais nem modelos preditivos",
    ]
    (OUT / "resumo_integracao.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    dados = carregar_fontes()
    auditar_fontes(dados)
    auditar_duplicados(dados)
    doentes, cir, lab, mon_src, anom_asa = normalizar_categorias(dados)

    # Unicidade de entidades: as linhas repetidas só são eliminadas quando integralmente iguais
    doentes, dup_doentes = remover_duplicados_seguros(doentes, "doentes", ["doente_id"])
    cir, dup_cir = remover_duplicados_seguros(cir, "cirurgias", ["cirurgia_id"])
    if doentes.doente_id.duplicated().any() or cir.cirurgia_id.duplicated().any():
        raise AssertionError("IDs de entidade não únicos depois da limpeza.")
    validar_ids_fontes(dados, cir)
    auditar_categorias_e_unidades(dados)
    mon, wide, dup_mon = preparar_monitorizacao(mon_src["monitorizacao_A"], mon_src["monitorizacao_B"], cir)
    lab_pre, eleg, dup_lab, conflitos_lab = preparar_laboratorio(lab, cir.cirurgia_id.tolist())
    hemo = calcular_metricas_hemodinamicas(mon, cir)
    integrado, merge_info = integrar_dados(cir, doentes, lab_pre, hemo)
    figura_info = gerar_visualizacoes(integrado, mon, hemo)
    escrever_resumo(dados, doentes, cir, lab_pre, eleg, mon, anom_asa, hemo, integrado,
                    merge_info, dup_mon, dup_lab, conflitos_lab, figura_info)

    # Registo explícito das repetições nas entidades
    (OUT / "decisoes_duplicados.txt").write_text(
        f"doentes.csv: {dup_doentes} linhas integralmente repetidas removidas\n"
        f"cirurgias.csv: {dup_cir} linhas integralmente repetidas removidas\n"
        f"monitorizacao_A+B: {dup_mon} linhas integralmente repetidas removidas após concatenação\n"
        f"laboratorio.csv: {dup_lab} linhas integralmente repetidas removidas\n"
        "No laboratório, chaves lógicas repetidas com conteúdo distinto foram preservadas e exportadas para duplicados_logicos_laboratorio.csv; se o empate ocorrer na medição pré-operatória mais recente, o valor fica missing e os candidatos são exportados para conflitos_laboratorio.csv\n"
        "Nas tabelas de entidades e monitorização, conflitos de chave lógica impedem a execução até existir uma decisão explícita\n",
        encoding="utf-8")
    print("Pipeline concluído com sucesso.")
    print(f"Dataset final: {len(integrado)} cirurgias; cirurgia_id único={integrado.cirurgia_id.is_unique}")
    print(f"Outputs: {OUT}")

if __name__ == "__main__":
    main()
