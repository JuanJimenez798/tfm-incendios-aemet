#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Funde el diario derivado de la observacion horaria (obs_diaria.csv, producido por
log_observacion.py en GitHub Actions) con aemet_limpio.parquet del cuaderno 02.

Filosofia del 02: marcar, no borrar. Las filas que vienen de la observacion horaria
se marcan con flag_obs_horaria=True y NUNCA pisan una fila oficial ya existente:
en cuanto AEMET publique el diario oficial de ese dia, manda el oficial.

Uso en Colab
------------
    !wget -q https://raw.githubusercontent.com/USUARIO/REPO/main/merge_observacion.py
    !wget -q -O obs_diaria.csv https://raw.githubusercontent.com/USUARIO/REPO/main/datos/obs_diaria.csv

    from merge_observacion import funde, comprueba_convencion
    ext = funde("/content/drive/MyDrive/TFM-Incendios/aemet_limpio.parquet",
                "obs_diaria.csv",
                "/content/drive/MyDrive/TFM-Incendios/aemet_limpio_ext.parquet")

Luego, en los cuadernos 08 y 09, apuntar AEMET_LIMPIO a aemet_limpio_ext.parquet.
"""

import pandas as pd

# obs_diaria -> nombres del parquet del 02
RENOMBRA = {"tmed": "tmed_obs"}                  # tmed del parquet se llama tmed_derivada
COLS_METEO = ["prec", "tmax", "tmin", "tmed_derivada", "hrMax", "hrMedia", "hrMin",
              "velmedia", "racha", "sol"]


def carga_obs(ruta_obs, solo_completos=True, prec_0707=False):
    """Lee obs_diaria.csv y lo deja con el esquema de aemet_limpio.parquet."""
    obs = pd.read_csv(ruta_obs, parse_dates=["fecha"], dtype={"indicativo": str})
    base = obs[obs["completo"]] if solo_completos else obs
    sel = base.rename(columns=RENOMBRA).copy()
    if prec_0707:
        sel["prec"] = sel["prec_0707"]
    sel["anio"] = sel["fecha"].dt.year
    sel["mes"] = sel["fecha"].dt.month
    sel["flag_obs_horaria"] = True
    cols = ["indicativo", "fecha", "anio", "mes"] + COLS_METEO + ["flag_obs_horaria"]
    return sel[[c for c in cols if c in sel.columns]]


def funde(ruta_parquet, ruta_obs, ruta_salida, solo_completos=True, prec_0707=False):
    """Anade a aemet_limpio.parquet los dias-estacion que no estan ya en el oficial."""
    oficial = pd.read_parquet(ruta_parquet)
    if "flag_obs_horaria" not in oficial.columns:
        oficial_marcado = oficial.assign(flag_obs_horaria=False)
    else:
        oficial_marcado = oficial

    nuevo = carga_obs(ruta_obs, solo_completos, prec_0707)

    # coordenadas y metadatos desde el propio parquet (la observacion trae lat/lon
    # aparte, pero mantener una sola fuente evita desalineaciones en el cruce)
    meta_cols = [c for c in ["nombre", "provincia", "altitud", "lat", "lon"]
                 if c in oficial_marcado.columns]
    if meta_cols:
        meta = (oficial_marcado.sort_values("fecha")
                               .drop_duplicates("indicativo", keep="last")
                               .set_index("indicativo")[meta_cols])
        con_meta = nuevo.join(meta, on="indicativo")
    else:
        con_meta = nuevo

    ya = set(map(tuple, oficial_marcado[["indicativo", "fecha"]].to_numpy()))
    clave = list(map(tuple, con_meta[["indicativo", "fecha"]].to_numpy()))
    solo_nuevos = con_meta[[k not in ya for k in clave]]

    union = pd.concat([oficial_marcado, solo_nuevos], ignore_index=True)
    ext = union.sort_values(["indicativo", "fecha"]).reset_index(drop=True)
    ext.to_parquet(ruta_salida, index=False)

    print(f"Oficial      : {len(oficial):,} filas, hasta {oficial.fecha.max().date()}")
    print(f"Observacion  : {len(con_meta):,} dias-estacion candidatos")
    print(f"Anadidos     : {len(solo_nuevos):,} (el resto ya estaba en el oficial)")
    print(f"Resultado    : {len(ext):,} filas, hasta {ext.fecha.max().date()} -> {ruta_salida}")
    return ext


def comprueba_convencion(ruta_parquet, ruta_obs):
    """En los dias en que coexisten oficial y observacion, mide que convencion de
    precipitacion casa mejor (00-24 UTC o 07-07 UTC) y el sesgo del resto de variables.
    Esto sustituye a suponerlo: se decide con los datos."""
    oficial = pd.read_parquet(ruta_parquet,
                              columns=["indicativo", "fecha", "prec", "tmax", "tmin",
                                       "hrMedia", "hrMin", "velmedia", "racha"])
    obs = pd.read_csv(ruta_obs, parse_dates=["fecha"], dtype={"indicativo": str})
    solape = obs[obs["completo"]].merge(oficial, on=["indicativo", "fecha"],
                                        suffixes=("_obs", "_ofi"))
    if solape.empty:
        print("Todavia no hay solape entre el oficial y la observacion. "
              "Vuelve a ejecutarlo cuando AEMET publique estos dias.")
        return None

    filas = []
    for col_obs, col_ofi, etiqueta in [("prec_obs", "prec_ofi", "prec 00-24 UTC"),
                                       ("prec_0707", "prec_ofi", "prec 07-07 UTC"),
                                       ("tmax_obs", "tmax_ofi", "tmax"),
                                       ("tmin_obs", "tmin_ofi", "tmin"),
                                       ("hrMedia_obs", "hrMedia_ofi", "hrMedia"),
                                       ("hrMin_obs", "hrMin_ofi", "hrMin"),
                                       ("velmedia_obs", "velmedia_ofi", "velmedia m/s"),
                                       ("racha_obs", "racha_ofi", "racha m/s")]:
        if col_obs not in solape or col_ofi not in solape:
            continue
        par = solape[[col_obs, col_ofi]].dropna()
        if len(par) < 30:
            continue
        err = par[col_obs] - par[col_ofi]
        filas.append(dict(variable=etiqueta, n=len(par),
                          r=round(par[col_obs].corr(par[col_ofi]), 3),
                          sesgo=round(err.mean(), 3), MAE=round(err.abs().mean(), 3)))
    tabla = pd.DataFrame(filas)
    print(f"Solape: {len(solape):,} dias-estacion "
          f"({solape.fecha.min().date()} -> {solape.fecha.max().date()})")
    print(tabla.to_string(index=False))
    print("\nLa fila de prec con menor MAE indica la convencion que usa AEMET "
          "para estas estaciones; usa prec_0707=True en funde() si gana la de 07-07.")
    return tabla
