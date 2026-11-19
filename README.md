
# Caso #2 — Estabilidade hemodinâmica durante cirurgia

**Ciência de Dados para a Saúde** · Licenciatura em Biomedicina Computacional e Inteligência Artificial

## Grupo

**Grupo_4**

| Nome | Número |
|------|--------|
| [Rafael Alves Pereira] | [nº22507906] |
| [Bernardo Esteves Margarido] | [nº22508039] |
| [Brian Ferrari Franck] | [nº22502756] |
| [Filipe Lopes] | [nº22508487] |

## Objetivo

Integrar cinco fontes perioperatórias pseudonimizadas (doentes, cirurgias, laboratório e dois blocos de monitorização) num dataset analítico com uma linha por cirurgia. A partir dele calculam-se métricas hemodinâmicas e geram-se seis figuras reprodutíveis sobre a estabilidade da PAM durante a cirurgia.

## Requisitos

- Python ≥ 3.10
- pandas ≥ 2.1
- numpy
- matplotlib
- seaborn

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Execução

Na raiz do projeto:

```bash
python main.py
```

Os caminhos são relativos ao próprio `main.py` e o programa não precisa de Internet. Lê `data/`, que nunca é alterada, e recria tudo em `outputs/`. O programa para com uma mensagem explícita se:
- faltar um ficheiro ou uma coluna obrigatória;
- uma chave lógica se repetir com conteúdo diferente;
- existir um `cirurgia_id` sem cirurgia correspondente;
- um merge não respeitar a cardinalidade esperada.

## Estrutura

```
Grupo_4
|-- main.py
|-- README.md
|-- requirements.txt
|-- apresentacao.pdf
|-- data/
|   |-- doentes.csv
|   |-- cirurgias.csv
|   |-- laboratorio.csv
|   |-- monitorizacao_A.csv
|   `-- monitorizacao_B.csv
`-- outputs/                              # gerado por main.py
    |-- auditoria_fontes.csv              # tipos e missing por variável e fonte
    |-- monitorizacao_preparada.csv       # A+B concatenados, com origem e flags de validação
    |-- laboratorio_preoperatorio.csv     # Hb e Creatinina pré-op (1 linha por cirurgia)
    |-- resumo_hemodinamico.csv           # métricas de PAM/FC/SpO2 por cirurgia
    |-- dados_cirurgicos_integrados.csv   # dataset final
    |-- resumo_integracao.txt             # controlos de qualidade e integração
    |-- decisoes_duplicados.txt           # duplicados removidos por fonte
    `-- figuras/
        |-- fig01_qualidade_dados.png
        |-- fig02_trajetoria_intraoperatoria.png  (+ .svg)
        |-- fig03_distribuicao_pam_baixa.png
        |-- fig04_comparacao_grupos.png
        |-- fig05_hb_pam.png
        `-- fig06_urgencia_especialidade.png
```

Etapas do `main.py`, por ordem:

1. `carregar_fontes`
2. `auditar_fontes`
3. `auditar_duplicados`
4. `normalizar_categorias`
5. `remover_duplicados_seguros`
6. `validar_ids_fontes`
7. `preparar_monitorizacao` 
8. `preparar_laboratorio`
9. `calcular_metricas_hemodinamicas`
10. `integrar_dados`
11. `gerar_visualizacoes`
12. `escrever_resumo`

## Unidade de observação do dataset final

`dados_cirurgicos_integrados.csv` tem **uma linha por episódio cirúrgico**. O programa verifica que `cirurgia_id` é único e que o número de linhas é igual ao de `cirurgias.csv`.

## Principais decisões

1. **Limpeza de categorias.** Removem-se espaços e normaliza-se a capitalização em `sexo`, `especialidade`, `abordagem`, `tipo_anestesia`, `urgencia`, `analito` e `parametro`. Só se agregam grafias que são explicitamente equivalentes. `idade`, `asa` e `duracao_anestesia_min` usam `Int64`, que aceita missing. Os valores ASA fora de 1–5 são sinalizados mas mantidos.
2. **Duplicados.** Só se removem linhas totalmente iguais. Se uma chave lógica se repetir com conteúdo diferente, o pipeline pára (doentes, cirurgias, monitorização). As chaves usadas são:
   - doentes: `doente_id`
   - cirurgias: `cirurgia_id`
   - laboratório: `cirurgia_id + tempo_relativo_h + analito`
   - monitorização: `cirurgia_id + tempo_min + parametro`
3. **Monitorização A + B → `pd.concat()`, não `merge()`.** Os dois ficheiros são blocos de exportação com a mesma estrutura. A coluna `origem` (A/B) guarda a proveniência, e o programa confirma que o total de linhas é a soma dos dois blocos.
4. **Validação da monitorização sem apagar nada.** Cada observação fica com `valor_utilizavel` e `motivo_validacao`. Os valores originais são mantidos. Regras técnicas, que não são limites clínicos:
   - FC entre 0 (exclusive) e 300
   - PAM entre 0 (exclusive) e 300
   - SpO2 entre 0 e 100
   - tempo inteiro e ≥ 0
   - unidade coerente com o parâmetro

   Os extremos pelo critério IQR (1,5×) ficam só sinalizados (`flag_extremo_iqr`) e não são excluídos.
5. **Long ↔ wide.** Cria-se um MultiIndex `cirurgia_id / tempo_min / parametro / origem` e o programa verifica que é único. Depois faz-se `pivot()` para wide (FC, PAM e SpO2 em colunas) e `stack()` para voltar a long. O resultado é comparado com o original através de `assert_frame_equal`.
6. **Laboratório reduzido antes do merge.**
   - Janela: `-24 <= tempo_relativo_h <= 0`.
   - Para Hemoglobina e Creatinina escolhe-se o instante mais próximo de 0.
   - O valor fica `NA` em três casos: o valor nesse instante está em falta, há valores ou unidades diferentes no mesmo instante, ou a unidade não é g/dL / mg/dL.
   - O estado de cada caso fica em `<analito>_estado`.
   - O resultado tem no máximo uma linha por cirurgia, o que evita uma junção many-to-many com a monitorização.
7. **Métricas hemodinâmicas.** Usam apenas observações utilizáveis.
   - `percentagem_pam_lt65 = 100 × minutos_pam_lt65 / n_minutos_pam_validos`. O denominador são os minutos com PAM válida, não a duração da anestesia, e os minutos sem PAM não contam como normais.
   - `cobertura_pam = n_minutos_pam_validos / duracao_anestesia_min`. A cobertura é **suficiente se ≥ 80%**. Este critério é usado nas figuras 3–5.
8. **Integração com left joins a partir de `cirurgias`**, para que nenhum episódio se perca:
   - `cirurgias ← doentes`: `validate="many_to_one"`
   - `← laboratório`: `validate="one_to_one"`
   - `← métricas`: `validate="one_to_one"`

   Todos os merges usam `indicator`, e o número de linhas é registado depois de cada um em `resumo_integracao.txt`.
