#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Logger de observacion horaria de AEMET  (endpoint /observacion/convencional/todas).

Por que existe
--------------
Los "valores climatologicos diarios" de AEMET se publican con ~10 dias de retraso.
El endpoint de observacion convencional devuelve las ULTIMAS 24 h de las ~800
estaciones automaticas, asi que ejecutandolo a diario el hueco se cubre desde hoy
hacia adelante. Es informacion IRREVERSIBLE: lo que no se descargue hoy, manana ya
no esta en ese endpoint (aunque acabara apareciendo en los diarios oficiales).

Salidas (dentro de DATOS/)
-------------------------
  obs_horaria/AAAA-MM-DD.csv.gz   crudo horario por fecha UTC de `fint` (nunca se pierde nada)
  obs_diaria.csv                  agregado diario con el esquema de los diarios de AEMET
  obs_estaciones.csv              inventario de idema -> lat/lon/alt/ubi visto en la observacion

Convenciones del endpoint (documentacion oficial de AEMET)
----------------------------------------------------------
  fint   fecha-hora FINAL del periodo de observacion, en UTC; los datos son de la
         hora ANTERIOR a fint.
  prec   precipitacion acumulada en los 60 min anteriores a fint (mm)
  vv     velocidad media del viento en los 10 min anteriores a fint (m/s)
  vmax   racha maxima (viento mantenido 3 s) registrada en los 60 min previos (m/s)
  hr     humedad relativa instantanea en fint (%)
  ta     temperatura instantanea en fint (C)
  tamin  minimo de los 60 valores instantaneos de ta de esa hora (C)
  tamax  maximo de los 60 valores instantaneos de ta de esa hora (C)
  inso   duracion de la insolacion en los 60 min previos (horas)

Asignacion de la hora a un dia
------------------------------
La hora que TERMINA en fint se asigna al dia que contiene fint - 30 min. Asi la
observacion de las 00:00 (que cubre 23-24 h) cae en el dia anterior, que es lo
correcto. Para la precipitacion se calculan las dos convenciones vigentes:
  prec        acumulado 00-24 UTC   (criterio de la red automatica)
  prec_0707   acumulado 07-07 UTC   (dia pluviometrico clasico, asignado al dia que empieza)
Cual de las dos casa con los diarios oficiales se decide midiendolo en el solape
(ver comprueba_convencion() en merge_observacion.py). No se asume.

Unidades: identicas a aemet_limpio.parquet (viento en m/s, prec en mm, temp en C).

Uso
---
  AEMET_API_KEY=xxx python log_observacion.py
  python log_observacion.py --credenciales credenciales.csv --datos datos
"""

import argparse
import gzip
import io
import json
import os
import sys
import time

import pandas as pd
import requests

URL_OBS = "https://opendata.aemet.es/opendata/api/observacion/convencional/todas"

# Campos que se guardan del crudo horario. El resto del JSON se descarta: si algun dia
# hace falta, se anade aqui y a partir de ese momento queda registrado.
CAMPOS = ["idema", "fint", "lat", "lon", "alt", "ubi",
          "ta", "tamin", "tamax", "hr", "vv", "vmax", "dv", "prec", "inso", "pres"]
NUMERICOS = ["lat", "lon", "alt", "ta", "tamin", "tamax", "hr",
             "vv", "vmax", "dv", "prec", "inso", "pres"]

MIN_HORAS_COMPLETO = 22     # horas necesarias para dar un dia por completo


# --------------------------------------------------------------------------- #
# Credenciales
# --------------------------------------------------------------------------- #
def credencial(nombre="AEMET_API_KEY", ruta_csv=None, obligatorio=True):
    """Variable de entorno primero (secreto de GitHub Actions), luego credenciales.csv.

    Formatos admitidos en el CSV (con o sin cabecera, con , o ;, entrecomillado o no):
      largo:  AEMET_API_KEY,eyJhbGci...        una clave por linea
      ancho:  AEMET_API_KEY,ESIOS_API_KEY      cabecera de nombres + una fila de valores
    """
    def es_nombre(x):        # distingue 'AEMET_API_KEY' (nombre) de un token real
        return (0 < len(x) <= 40 and "_" in x and x == x.upper()
                and all(ch.isalnum() or ch == "_" for ch in x))

    v = os.environ.get(nombre)
    if v:
        return v.strip()

    if ruta_csv and os.path.exists(ruta_csv):
        with open(ruta_csv, encoding="utf-8-sig") as fh:
            filas = [[x.strip().strip('"').strip("'")
                      for x in ln.replace(";", ",").rstrip("\n").split(",")]
                     for ln in fh if ln.strip()]
        # ancho: la primera fila son SOLO nombres de clave
        if len(filas) >= 2 and nombre in filas[0] and all(es_nombre(x) for x in filas[0] if x):
            j = filas[0].index(nombre)
            for c in filas[1:]:
                if len(c) > j and c[j]:
                    return c[j]
        # largo: nombre en la primera columna, valor en la segunda
        for c in filas:
            if len(c) >= 2 and c[0] == nombre and c[1] and not es_nombre(c[1]):
                return c[1]

    if obligatorio:
        raise KeyError(f"{nombre}: no esta en el entorno ni en {ruta_csv}")
    return None


# --------------------------------------------------------------------------- #
# Descarga
# --------------------------------------------------------------------------- #
def descarga_observacion(api_key, intentos=5):
    """Devuelve la lista de dicts de las ultimas 24 h. AEMET responde en dos saltos:
    primero un JSON con el campo `datos` (URL temporal) y luego el contenido."""
    ultimo_error = None
    for i in range(intentos):
        try:
            r = requests.get(URL_OBS, params={"api_key": api_key}, timeout=90)
            if r.status_code == 429:                      # limite de peticiones
                time.sleep(20 * (i + 1))
                continue
            r.raise_for_status()
            meta = r.json()
            enlace = meta.get("datos")
            if not enlace:
                ultimo_error = f"respuesta sin 'datos': {meta}"
                time.sleep(5 * (i + 1))
                continue
            r2 = requests.get(enlace, timeout=120)
            r2.raise_for_status()
            # AEMET sirve la observacion en latin-1
            return json.loads(r2.content.decode("latin-1"))
        except Exception as err:                          # red, JSON roto, 5xx...
            ultimo_error = repr(err)
            time.sleep(5 * (i + 1))
    raise RuntimeError(f"AEMET no respondio tras {intentos} intentos. Ultimo error: {ultimo_error}")


def normaliza(registros):
    bruto = pd.DataFrame(registros)
    faltan = [c for c in CAMPOS if c not in bruto.columns]
    for c in faltan:
        bruto[c] = pd.NA
    obs = bruto[CAMPOS].copy()
    obs["fint"] = pd.to_datetime(obs["fint"], errors="coerce")
    for c in NUMERICOS:
        obs[c] = pd.to_numeric(obs[c], errors="coerce")
    obs["idema"] = obs["idema"].astype(str).str.strip()
    limpio = obs.dropna(subset=["idema", "fint"])
    return limpio.sort_values(["idema", "fint"]), faltan


# --------------------------------------------------------------------------- #
# Persistencia del crudo horario (particionado por fecha UTC de fint)
# --------------------------------------------------------------------------- #
def guarda_crudo(obs, dir_horaria):
    """Funde con lo ya guardado y deduplica por (idema, fint), quedandose con lo ultimo.
    Devuelve el numero de filas nuevas por fichero tocado."""
    os.makedirs(dir_horaria, exist_ok=True)
    nuevas = {}
    for fecha, grupo in obs.groupby(obs["fint"].dt.date):
        ruta = os.path.join(dir_horaria, f"{fecha}.csv.gz")
        if os.path.exists(ruta):
            previo = pd.read_csv(ruta, parse_dates=["fint"], dtype={"idema": str})
            union = pd.concat([previo, grupo], ignore_index=True)
        else:
            previo = None
            union = grupo
        fundido = (union.sort_values(["idema", "fint"])
                        .drop_duplicates(["idema", "fint"], keep="last")
                        .reset_index(drop=True))
        fundido.to_csv(ruta, index=False, compression="gzip")
        nuevas[str(fecha)] = len(fundido) - (0 if previo is None else len(previo))
    return nuevas


def carga_crudo(dir_horaria, dias=None):
    """Lee el crudo horario guardado. `dias` limita a los N ficheros mas recientes."""
    if not os.path.isdir(dir_horaria):
        return pd.DataFrame()
    ficheros = sorted(f for f in os.listdir(dir_horaria) if f.endswith(".csv.gz"))
    if dias:
        ficheros = ficheros[-dias:]
    partes = [pd.read_csv(os.path.join(dir_horaria, f), parse_dates=["fint"],
                          dtype={"idema": str}) for f in ficheros]
    if not partes:
        return pd.DataFrame()
    todo = pd.concat(partes, ignore_index=True)
    return todo.drop_duplicates(["idema", "fint"], keep="last")


# --------------------------------------------------------------------------- #
# Agregado diario
# --------------------------------------------------------------------------- #
def agrega_diario(horaria):
    """Horaria cruda -> diario con el esquema de los diarios de AEMET.

    Columnas: indicativo, fecha, tmax, tmin, tmed, tmed_derivada, hrMax, hrMedia,
              hrMin, velmedia, racha, prec, prec_0707, sol, n_horas, completo.
    velmedia/racha en m/s (como aemet_limpio.parquet), prec en mm, temp en C.
    """
    if horaria.empty:
        return pd.DataFrame()

    h = horaria.copy()
    # la hora que TERMINA en fint pertenece al dia que contiene fint - 30 min
    h["dia"] = (h["fint"] - pd.Timedelta(minutes=30)).dt.normalize()
    # dia pluviometrico 07-07 UTC, asignado al dia en que EMPIEZA la ventana
    h["dia_pluv"] = (h["fint"] - pd.Timedelta(hours=7, minutes=30)).dt.normalize()

    # tamin/tamax son los extremos reales de la hora; si faltan, se cae a la instantanea
    h["t_alta"] = h["tamax"].fillna(h["ta"])
    h["t_baja"] = h["tamin"].fillna(h["ta"])

    g = h.groupby(["idema", "dia"])
    diario = pd.DataFrame({
        "tmax":     g["t_alta"].max(),
        "tmin":     g["t_baja"].min(),
        "tmed":     g["ta"].mean().round(2),          # media de las instantaneas horarias
        "hrMax":    g["hr"].max(),
        "hrMedia":  g["hr"].mean().round(1),
        "hrMin":    g["hr"].min(),
        "velmedia": g["vv"].mean().round(2),          # m/s
        "racha":    g["vmax"].max(),                  # m/s
        "prec":     g["prec"].sum(min_count=1),       # 00-24 UTC
        "sol":      g["inso"].sum(min_count=1),
        "n_horas":  g["fint"].size(),
    }).reset_index()

    # tmed_derivada: misma definicion que la columna homonima de aemet_limpio.parquet
    diario["tmed_derivada"] = ((diario["tmax"] + diario["tmin"]) / 2).round(2)

    # precipitacion en la convencion 07-07
    pluv = (h.groupby(["idema", "dia_pluv"])["prec"].sum(min_count=1)
             .rename("prec_0707").reset_index()
             .rename(columns={"dia_pluv": "dia"}))
    con_pluv = diario.merge(pluv, on=["idema", "dia"], how="left")

    con_pluv["completo"] = con_pluv["n_horas"] >= MIN_HORAS_COMPLETO
    salida = con_pluv.rename(columns={"idema": "indicativo", "dia": "fecha"})

    ORDEN = ["indicativo", "fecha", "tmax", "tmin", "tmed", "tmed_derivada",
             "hrMax", "hrMedia", "hrMin", "velmedia", "racha",
             "prec", "prec_0707", "sol", "n_horas", "completo"]
    return salida[ORDEN].sort_values(["fecha", "indicativo"]).reset_index(drop=True)


def inventario(horaria):
    if horaria.empty:
        return pd.DataFrame()
    ult = (horaria.sort_values("fint").drop_duplicates("idema", keep="last")
                  [["idema", "lat", "lon", "alt", "ubi"]])
    return ult.rename(columns={"idema": "indicativo"}).sort_values("indicativo")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Logger de observacion horaria de AEMET")
    ap.add_argument("--datos", default="datos", help="carpeta de salida (por defecto: datos)")
    ap.add_argument("--credenciales", default="credenciales.csv",
                    help="CSV con AEMET_API_KEY si no hay variable de entorno")
    ap.add_argument("--dias-agregado", type=int, default=None,
                    help="recalcular el diario solo con los N dias mas recientes")
    args = ap.parse_args()

    dir_horaria = os.path.join(args.datos, "obs_horaria")
    ruta_diaria = os.path.join(args.datos, "obs_diaria.csv")
    ruta_estac = os.path.join(args.datos, "obs_estaciones.csv")
    os.makedirs(args.datos, exist_ok=True)

    api_key = credencial("AEMET_API_KEY", args.credenciales)

    registros = descarga_observacion(api_key)
    obs, faltan = normaliza(registros)
    if faltan:
        print(f"[aviso] campos ausentes en esta respuesta: {faltan}")
    print(f"Descargadas {len(obs):,} filas horarias | "
          f"{obs.idema.nunique()} estaciones | "
          f"{obs.fint.min()} -> {obs.fint.max()} UTC")

    nuevas = guarda_crudo(obs, dir_horaria)
    print("Crudo horario actualizado:",
          ", ".join(f"{k} (+{v})" for k, v in sorted(nuevas.items())) or "sin cambios")

    horaria = carga_crudo(dir_horaria, args.dias_agregado)
    diario = agrega_diario(horaria)
    diario.to_csv(ruta_diaria, index=False)

    inv = inventario(horaria)
    if not inv.empty:
        inv.to_csv(ruta_estac, index=False)

    completos = diario[diario.completo]
    print(f"\nobs_diaria.csv: {len(diario):,} filas | "
          f"{diario.fecha.min().date()} -> {diario.fecha.max().date()}")
    print(f"  dias-estacion completos (>={MIN_HORAS_COMPLETO} h): {len(completos):,} "
          f"({len(completos) / max(len(diario), 1) * 100:.1f}%)")
    por_dia = (diario.groupby(diario.fecha.dt.date)["completo"].agg(["sum", "size"])
                     .rename(columns={"sum": "estaciones_completas", "size": "estaciones"}))
    print(por_dia.tail(12).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
