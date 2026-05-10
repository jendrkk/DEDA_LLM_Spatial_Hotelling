"""
gebaeude_capacity.py
====================
Building floor-space efficiency factors and employee hard-cap logic derived
from the ALKIS-OK Berlin Gebäudefunktion (GFK) catalogue
(Stand: Juni 2024, Kennung 31001).

Overview
--------
This module provides two related capabilities:

1. **Efficiency factors** — map GFK code → NUF/BGF ratio, the share of
   gross floor area that constitutes net usable floor space.

2. **Employee hard cap** — derive a physical upper bound on how many employees
   a building can accommodate, and enforce that cap against reported IHK
   headcounts (single-company and multi-company cases).

Definitions
-----------
EF (efficiency factor)  = NUF / BGF
    NUF  Netto-Nutzfläche (net usable floor area)
    BGF  Brutto-Grundfläche (gross floor area = footprint × floors)

Hard cap H  = (footprint_m2 × num_floors × EF) / m2_per_employee
    The maximum number of employees that can physically fit in the building
    given its size, function, and layout.

EF Calibration sources
-----------------------
    DIN 277-1:2016  Grundflächen und Rauminhalte im Bauwesen
    gif MF-G 2017   Mietflächenrichtlinie Gewerbeflächen (gif e.V.)
    RICS            Code of Measuring Practice, 6th ed.

m²/employee Calibration sources
---------------------------------
    ArbStättV       Arbeitsstättenverordnung, Anhang 1.2 (min. 8 m²/person)
    gif BR 2023     gif Büroflächenreport 2022/2023 Berlin
    RICS GOCS 2023  RICS Global Occupancy Costs Survey 2023
    StaBu EH        Statistisches Bundesamt, Strukturerhebung Einzelhandel
    IHA 2023        IHA Hotelmarkt Deutschland 2023
    DKG             Deutsche Krankenhausgesellschaft — staffing norms
    BVL             Bundesvereinigung Logistik — Logistikimmobilien report
    DEHOGA          Branchenbericht Gastronomie

Hochhaus (high-rise) EF penalty
---------------------------------
Buildings flagged `hochhaus=True` (≥22 m or ≥8 storeys above ground) receive
an additive EF reduction reflecting the proportionally larger structural and
service core that scales with height.  See HOCHHAUS_PENALTY.

Multi-company cap enforcement
------------------------------
When N IHK registrations share the same building (identified by spatial join
to `gebaeude.gpkg`) and their aggregate reported headcount exceeds the hard
cap H, a **proportional scaling** rule is applied:

    X_i_capped = X_i × min(1.0, H / Σ X_j)

This preserves relative firm sizes while enforcing the physical aggregate
constraint.  For a single company the rule reduces to min(X, H).

Usage
-----
    from hotelling.spatial.gebaeude_capacity import (
        get_efficiency_factor,
        get_m2_per_employee,
        compute_employee_hard_cap,
        apply_hard_cap_single,
        apply_hard_cap_multi,
    )

    # Vectorised efficiency factor
    gdf["efficiency"] = gdf.apply(
        lambda r: get_efficiency_factor(r["gfk"], bool(r["hochhaus"])), axis=1
    )
    gdf["usable_area_m2"] = (
        gdf.geometry.area
        * gdf["anzahl_der_oberirdischen_geschosse"].clip(lower=1)
        * gdf["efficiency"]
    )

    # Employee hard cap per building
    gdf["employee_hard_cap"] = gdf.apply(
        lambda r: compute_employee_hard_cap(
            r.geometry.area,
            r["anzahl_der_oberirdischen_geschosse"],
            r["gfk"],
            bool(r["hochhaus"]),
        ),
        axis=1,
    )

    # Apply cap — single company
    company["empl_capped"] = company.apply(
        lambda r: apply_hard_cap_single(r["empl"], r["employee_hard_cap"]),
        axis=1,
    )

    # Apply cap — multiple companies in same building (group by building id)
    company["empl_capped"] = (
        company.groupby("building_id", group_keys=False)
        .apply(lambda g: apply_hard_cap_multi(g["empl"], g["employee_hard_cap"].iloc[0]))
    )
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Additive EF deduction for high-rise buildings (≥22 m / ≥8 storeys).
#: High-rises sacrifice more floor plate to structural cores, fire-escape
#: shafts, mechanical plant floors, and additional elevator banks.
#: Empirical range: 0.06–0.08; central value 0.07 adopted here.
#: Source: gif MF-G 2017 core-to-plate ratio analysis; RICS CoMP 6th ed.
HOCHHAUS_PENALTY: float = 0.07

#: Minimum EF after applying the hochhaus penalty (absolute floor).
EF_MIN: float = 0.40

# ── Default EF by broad class ────────────────────────────────────────────
EF_DEFAULT_RESIDENTIAL: float = 0.82
EF_DEFAULT_COMMERCIAL:  float = 0.75
EF_DEFAULT_INDUSTRIAL:  float = 0.86
EF_DEFAULT_PUBLIC:      float = 0.70
EF_DEFAULT_UNKNOWN:     float = 0.75

# ── m²/employee defaults by broad class ─────────────────────────────────
M2_DEFAULT_OFFICE:       float = 12.0
M2_DEFAULT_RETAIL:       float = 18.0
M2_DEFAULT_INDUSTRIAL:   float = 40.0
M2_DEFAULT_PUBLIC:       float = 20.0
M2_DEFAULT_RESIDENTIAL:  float = 999.0  # no meaningful employee density
M2_DEFAULT_UNKNOWN:      float = 15.0

# ── Group-level EF fallback (round GFK down to nearest 1000) ─────────────
_GROUP_EF_DEFAULTS: dict[int, float] = {
    1000: EF_DEFAULT_RESIDENTIAL,
    2000: EF_DEFAULT_COMMERCIAL,
    3000: EF_DEFAULT_PUBLIC,
    3200: 0.68,
}

# ── Group-level m²/employee fallback ─────────────────────────────────────
_GROUP_M2_DEFAULTS: dict[int, float] = {
    1000: M2_DEFAULT_RESIDENTIAL,
    2000: M2_DEFAULT_OFFICE,
    3000: M2_DEFAULT_PUBLIC,
    3200: 25.0,
}

# ---------------------------------------------------------------------------
# Human-readable labels
# ---------------------------------------------------------------------------
GFK_LABEL: dict[int, str] = {
    1000: "Wohngebäude (allgemein)",
    1010: "Wohnhaus",
    1020: "Wohnheim",
    1021: "Kinderheim",
    1022: "Seniorenheim",
    1023: "Schwesternwohnheim",
    1024: "Studenten-/Schülerwohnheim",
    1025: "Schullandheim",
    1100: "Gemischt genutztes Gebäude mit Wohnen",
    1110: "Wohngebäude mit Gemeinbedarf",
    1120: "Wohngebäude mit Handel und Dienstleistungen",
    1130: "Wohngebäude mit Gewerbe und Industrie",
    1210: "Land-/forstwirtschaftliches Wohngebäude",
    1220: "Land-/forstwirtschaftliches Wohn- und Betriebsgebäude",
    1223: "Forsthaus",
    1310: "Gebäude zur Freizeitgestaltung",
    1311: "Ferienhaus",
    1312: "Wochenendhaus",
    1313: "Gartenhaus",
    2000: "Gebäude für Wirtschaft oder Gewerbe (allgemein)",
    2010: "Gebäude für Handel und Dienstleistungen",
    2020: "Bürogebäude",
    2030: "Kreditinstitut",
    2040: "Versicherung",
    2050: "Geschäftsgebäude",
    2051: "Kaufhaus",
    2052: "Einkaufszentrum",
    2053: "Markthalle",
    2054: "Laden",
    2055: "Kiosk",
    2060: "Messehalle",
    2070: "Gebäude für Beherbergung (allgemein)",
    2071: "Hotel / Motel / Pension",
    2072: "Jugendherberge",
    2074: "Campingplatzgebäude",
    2080: "Gebäude für Bewirtung (allgemein)",
    2081: "Gaststätte / Restaurant",
    2083: "Kantine",
    2090: "Freizeit- und Vergnügungsstätte",
    2091: "Festsaal",
    2092: "Kino",
    2093: "Kegel- / Bowlinghalle",
    2094: "Spielkasino",
    2100: "Gebäude für Gewerbe und Industrie (allgemein)",
    2111: "Fabrik",
    2120: "Werkstatt",
    2130: "Tankstelle",
    2140: "Gebäude für Vorratshaltung (allgemein)",
    2141: "Kühlhaus",
    2142: "Speichergebäude",
    2143: "Lagerhalle / Lagerschuppen / Lagerhaus",
    2150: "Speditionsgebäude",
    2160: "Gebäude für Forschungszwecke",
    2180: "Gebäude für betriebliche Sozialeinrichtung",
    2200: "Sonstiges Gebäude für Gewerbe und Industrie",
    2211: "Windmühle",
    2212: "Wassermühle",
    2213: "Schöpfwerk",
    2310: "Gebäude für Handel und Dienstleistungen mit Wohnen",
    2320: "Gebäude für Gewerbe und Industrie mit Wohnen",
    2400: "Betriebsgebäude zu Verkehrsanlagen (allgemein)",
    2410: "Betriebsgebäude für Straßenverkehr",
    2411: "Straßenmeisterei",
    2412: "Wartungshalle",
    2420: "Betriebsgebäude für Schienenverkehr",
    2421: "Bahnwärterhaus",
    2422: "Lokschuppen / Wagenhalle",
    2423: "Stellwerk / Blockstelle",
    2424: "Betriebsgebäude des Güterbahnhofs",
    2430: "Betriebsgebäude für Flugverkehr",
    2431: "Flugzeughalle",
    2440: "Betriebsgebäude für Schiffsverkehr",
    2441: "Werft (Halle)",
    2442: "Dock (Halle)",
    2443: "Betriebsgebäude zur Schleuse",
    2444: "Bootshaus",
    2460: "Gebäude zum Parken (allgemein)",
    2461: "Parkhaus",
    2462: "Parkdeck",
    2463: "Garage",
    2464: "Fahrzeughalle",
    2465: "Tiefgarage",
    2500: "Gebäude zur Versorgung (allgemein)",
    2510: "Gebäude zur Wasserversorgung",
    2511: "Wasserwerk",
    2512: "Pumpstation",
    2513: "Wasserbehälter",
    2520: "Gebäude zur Elektrizitätsversorgung",
    2521: "Elektrizitätswerk",
    2522: "Umspannwerk",
    2523: "Umformer",
    2540: "Gebäude für Fernmeldewesen",
    2560: "Gebäude an unterirdischen Leitungen",
    2571: "Gaswerk",
    2580: "Heizwerk",
    2591: "Pumpwerk (nicht Wasserversorgung)",
    2600: "Gebäude zur Entsorgung (allgemein)",
    2610: "Gebäude zur Abwasserbeseitigung",
    2611: "Gebäude der Kläranlage",
    2612: "Toilette",
    2620: "Gebäude zur Abfallbehandlung",
    2621: "Müllbunker",
    2622: "Gebäude zur Müllverbrennung",
    2623: "Gebäude der Abfalldeponie",
    2700: "Gebäude für Land- und Forstwirtschaft (allgemein)",
    2720: "Land-/forstwirtschaftliches Betriebsgebäude",
    2721: "Scheune",
    2723: "Schuppen",
    2724: "Stall",
    2726: "Scheune und Stall",
    2727: "Stall für Tiergroßhaltung",
    2735: "Jagdhaus / Jagdhütte",
    2741: "Treibhaus",
    2742: "Gewächshaus (verschiebbar)",
    3000: "Gebäude für öffentliche Zwecke (allgemein)",
    3010: "Verwaltungsgebäude",
    3011: "Parlament",
    3012: "Rathaus",
    3013: "Post",
    3014: "Zollamt",
    3015: "Gericht",
    3016: "Botschaft / Konsulat",
    3020: "Gebäude für Bildung und Forschung (allgemein)",
    3021: "Allgemeinbildende Schule",
    3022: "Berufsbildende Schule",
    3023: "Hochschulgebäude",
    3024: "Forschungsinstitut",
    3030: "Gebäude für kulturelle Zwecke (allgemein)",
    3031: "Schloss",
    3032: "Theater / Oper",
    3033: "Konzertgebäude",
    3034: "Museum",
    3035: "Rundfunk / Fernsehen",
    3036: "Veranstaltungsgebäude",
    3037: "Bibliothek / Bücherei",
    3038: "Burg / Festung",
    3040: "Gebäude für religiöse Zwecke (allgemein)",
    3041: "Kirche",
    3042: "Synagoge",
    3043: "Kapelle",
    3044: "Gemeindehaus",
    3045: "Gotteshaus",
    3046: "Moschee",
    3050: "Gebäude für Gesundheitswesen (allgemein)",
    3051: "Krankenhaus",
    3052: "Heilanstalt / Pflegeanstalt",
    3060: "Gebäude für soziale Zwecke (allgemein)",
    3061: "Jugendfreizeitheim",
    3062: "Freizeit- / Vereinsheim",
    3063: "Seniorenfreizeitstätte",
    3064: "Obdachlosenheim",
    3065: "Kinderkrippe / Kindergarten / Kindertagesstätte",
    3070: "Gebäude für Sicherheit und Ordnung (allgemein)",
    3071: "Polizei",
    3072: "Feuerwehr",
    3073: "Kaserne",
    3074: "Schutzbunker",
    3075: "Justizvollzugsanstalt",
    3080: "Friedhofsgebäude",
    3081: "Trauerhalle",
    3082: "Krematorium",
    3090: "Empfangsgebäude (allgemein)",
    3091: "Bahnhofsgebäude",
    3092: "Flughafengebäude",
    3094: "Gebäude zum U-Bahnhof",
    3095: "Gebäude zum S-Bahnhof",
    3097: "Gebäude zum Busbahnhof",
    3100: "Gebäude für öffentliche Zwecke mit Wohnen",
    3200: "Gebäude für Erholungszwecke (allgemein)",
    3210: "Gebäude für Sportzwecke (allgemein)",
    3211: "Sport- / Turnhalle",
    3212: "Gebäude zum Sportplatz",
    3220: "Badegebäude (allgemein)",
    3221: "Hallenbad",
    3222: "Gebäude im Freibad",
    3240: "Gebäude für Kurbetrieb",
    3241: "Badegebäude für medizinische Zwecke",
    3242: "Sanatorium",
    3260: "Gebäude im Zoo (allgemein)",
    3261: "Empfangsgebäude des Zoos",
    3262: "Aquarium / Terrarium / Voliere",
    3263: "Tierschauhaus",
    3264: "Stall im Zoo",
    3270: "Gebäude im botanischen Garten (allgemein)",
    3271: "Empfangsgebäude des botanischen Gartens",
    3272: "Gewächshaus (Botanik)",
    3273: "Pflanzenschauhaus",
    3281: "Schutzhütte",
    9998: "Nach Quellenlage nicht zu spezifizieren",
}

# ---------------------------------------------------------------------------
# EF table — NUF/BGF ratio
# ---------------------------------------------------------------------------
# Annotation key:
#   [DIN277]   calibrated against DIN 277-1:2016 typical values
#   [gif-MFG]  calibrated against gif MF-G 2017
#   [EMPT]     building type rarely carries commercial employees;
#              EF kept for spatial accounting but set low
#   [N/A]      no meaningful employee capacity; set very low
#
GFK_BASE_EFFICIENCY: dict[int, float] = {
    # ── 1000 Wohngebäude ─────────────────────────────────────────────────
    1000: 0.84,  # allgemein                              [DIN277]
    1010: 0.85,  # Wohnhaus                               [DIN277]
    1020: 0.78,  # Wohnheim                               [DIN277]
    1021: 0.74,  # Kinderheim                             [DIN277]
    1022: 0.72,  # Seniorenheim                           [DIN277]
    1023: 0.76,  # Schwesternwohnheim                     [DIN277]
    1024: 0.78,  # Studenten-/Schülerwohnheim             [DIN277]
    1025: 0.80,  # Schullandheim                          [DIN277]
    1100: 0.80,  # Gemischt mit Wohnen                    [DIN277]
    1110: 0.80,  # Wohngebäude mit Gemeinbedarf           [DIN277]
    1120: 0.79,  # Wohngebäude mit Handel                 [DIN277]
    1130: 0.78,  # Wohngebäude mit Gewerbe                [DIN277]
    1210: 0.82,  # Land-/forstwirtschaftl. Wohngebäude    [DIN277]
    1220: 0.80,  # Land-/forstwirtschaftl. Wohn+Betrieb   [DIN277]
    1223: 0.82,  # Forsthaus                              [DIN277]
    1310: 0.78,  # Freizeitgestaltung                     [DIN277]
    1311: 0.82,  # Ferienhaus                             [DIN277]
    1312: 0.82,  # Wochenendhaus                          [DIN277]
    1313: 0.88,  # Gartenhaus — single open room          [DIN277]
    # ── 2000 Wirtschaft & Gewerbe ────────────────────────────────────────
    2000: 0.75,  # allgemein                              [gif-MFG]
    # Key: 2010 is the catch-all for most Berlin offices;
    # hochhaus flag drops this to 0.75 − 0.07 = 0.68
    2010: 0.75,  # Handel + Dienstleistungen (offices)    [gif-MFG]
    2020: 0.76,  # Bürogebäude (explicitly labelled)      [gif-MFG]
    2030: 0.73,  # Kreditinstitut — vault overhead        [gif-MFG]
    2040: 0.74,  # Versicherung                           [gif-MFG]
    2050: 0.78,  # Geschäftsgebäude                       [gif-MFG]
    2051: 0.80,  # Kaufhaus                               [gif-MFG]
    2052: 0.80,  # Einkaufszentrum                        [gif-MFG]
    2053: 0.83,  # Markthalle                             [gif-MFG]
    2054: 0.82,  # Laden                                  [gif-MFG]
    2055: 0.88,  # Kiosk — single room                    [gif-MFG]
    2060: 0.82,  # Messehalle                             [gif-MFG]
    2070: 0.63,  # Beherbergung allgemein                 [DIN277]
    2071: 0.62,  # Hotel/Motel/Pension                    [DIN277]
    2072: 0.66,  # Jugendherberge                         [DIN277]
    2074: 0.72,  # Campingplatzgebäude                    [DIN277]
    2080: 0.72,  # Bewirtung allgemein                    [DIN277]
    2081: 0.73,  # Gaststätte/Restaurant                  [DIN277]
    2083: 0.76,  # Kantine                                [DIN277]
    2090: 0.68,  # Freizeit/Vergnüg.                      [DIN277]
    2091: 0.65,  # Festsaal — stage + foyer               [DIN277]
    2092: 0.68,  # Kino                                   [DIN277]
    2093: 0.74,  # Kegel-/Bowlinghalle                    [DIN277]
    2094: 0.70,  # Spielkasino                            [DIN277]
    2100: 0.84,  # Gewerbe/Industrie allgemein            [DIN277]
    2111: 0.85,  # Fabrik                                 [DIN277]
    2120: 0.82,  # Werkstatt                              [DIN277]
    2130: 0.78,  # Tankstelle                             [DIN277]
    2140: 0.89,  # Vorratshaltung allgemein               [DIN277]
    2141: 0.88,  # Kühlhaus — insulated walls             [DIN277]
    2142: 0.90,  # Speichergebäude                        [DIN277]
    2143: 0.91,  # Lagerhalle                             [DIN277]
    2150: 0.85,  # Speditionsgebäude                      [DIN277]
    2160: 0.68,  # Forschungsgebäude — lab infrastructure [DIN277]
    2180: 0.72,  # Betriebliche Sozialeinrichtung         [DIN277]
    2200: 0.78,  # Sonstiges Gewerbe/Industrie            [DIN277]
    2211: 0.60,  # Windmühle                              [EMPT]
    2212: 0.60,  # Wassermühle                            [EMPT]
    2213: 0.65,  # Schöpfwerk                             [EMPT]
    2310: 0.76,  # Handel+Dienstleistungen mit Wohnen     [DIN277]
    2320: 0.78,  # Gewerbe+Industrie mit Wohnen           [DIN277]
    2400: 0.75,  # Betriebsgeb. Verkehrsanlagen allg.     [DIN277]
    2410: 0.74,  # Betriebsgeb. Straßenverkehr            [DIN277]
    2411: 0.72,  # Straßenmeisterei                       [DIN277]
    2412: 0.84,  # Wartungshalle                          [DIN277]
    2420: 0.74,  # Betriebsgeb. Schienenverkehr           [DIN277]
    2421: 0.80,  # Bahnwärterhaus                         [DIN277]
    2422: 0.88,  # Lokschuppen/Wagenhalle                 [DIN277]
    2423: 0.72,  # Stellwerk                              [DIN277]
    2424: 0.76,  # Betriebsgeb. Güterbahnhof              [DIN277]
    2430: 0.74,  # Betriebsgeb. Flugverkehr               [DIN277]
    2431: 0.90,  # Flugzeughalle — hangar                 [DIN277]
    2440: 0.74,  # Betriebsgeb. Schiffsverkehr            [DIN277]
    2441: 0.88,  # Werft (Halle)                          [DIN277]
    2442: 0.85,  # Dock (Halle)                           [DIN277]
    2443: 0.72,  # Betriebsgeb. Schleuse                  [DIN277]
    2444: 0.80,  # Bootshaus                              [DIN277]
    2460: 0.82,  # Parken allgemein                       [N/A]
    2461: 0.82,  # Parkhaus                               [N/A]
    2462: 0.85,  # Parkdeck                               [N/A]
    2463: 0.88,  # Garage                                 [N/A]
    2464: 0.84,  # Fahrzeughalle                          [N/A]
    2465: 0.85,  # Tiefgarage                             [N/A]
    2500: 0.65,  # Versorgung allgemein                   [EMPT]
    2510: 0.65,  # Wasserversorgung                       [EMPT]
    2511: 0.60,  # Wasserwerk                             [EMPT]
    2512: 0.62,  # Pumpstation                            [EMPT]
    2513: 0.30,  # Wasserbehälter — mostly tank volume    [N/A]
    2520: 0.60,  # Elektrizitätsversorgung                [EMPT]
    2521: 0.58,  # Elektrizitätswerk                      [EMPT]
    2522: 0.55,  # Umspannwerk — transformer-dominated    [N/A]
    2523: 0.50,  # Umformer                               [N/A]
    2540: 0.65,  # Fernmeldewesen                         [EMPT]
    2560: 0.60,  # Gebäude an unterird. Leitungen         [EMPT]
    2571: 0.62,  # Gaswerk                                [EMPT]
    2580: 0.62,  # Heizwerk                               [EMPT]
    2591: 0.58,  # Pumpwerk                               [EMPT]
    2600: 0.65,  # Entsorgung allgemein                   [EMPT]
    2610: 0.62,  # Abwasserbeseitigung                    [EMPT]
    2611: 0.60,  # Kläranlage                             [EMPT]
    2612: 0.40,  # Toilette                               [N/A]
    2620: 0.65,  # Abfallbehandlung                       [EMPT]
    2621: 0.80,  # Müllbunker                             [EMPT]
    2622: 0.65,  # Müllverbrennung                        [EMPT]
    2623: 0.60,  # Abfalldeponie                          [EMPT]
    2700: 0.82,  # Land-/Forstwirtschaft allgemein        [DIN277]
    2720: 0.80,  # Betriebsgebäude Landwirtschaft         [DIN277]
    2721: 0.92,  # Scheune — open barn                    [DIN277]
    2723: 0.88,  # Schuppen                               [DIN277]
    2724: 0.88,  # Stall                                  [DIN277]
    2726: 0.90,  # Scheune und Stall                      [DIN277]
    2727: 0.88,  # Stall für Tiergroßhaltung              [DIN277]
    2735: 0.80,  # Jagdhaus/Jagdhütte                     [DIN277]
    2741: 0.82,  # Treibhaus                              [DIN277]
    2742: 0.80,  # Gewächshaus (verschiebbar)             [DIN277]
    # ── 3000 Öffentliche Zwecke ──────────────────────────────────────────
    3000: 0.70,  # allgemein                              [DIN277]
    3010: 0.72,  # Verwaltungsgebäude                     [gif-MFG]
    3011: 0.65,  # Parlament                              [DIN277]
    3012: 0.68,  # Rathaus                                [DIN277]
    3013: 0.73,  # Post                                   [DIN277]
    3014: 0.70,  # Zollamt                                [DIN277]
    3015: 0.68,  # Gericht                                [DIN277]
    3016: 0.68,  # Botschaft/Konsulat                     [DIN277]
    3020: 0.70,  # Bildung/Forschung allgemein            [DIN277]
    3021: 0.71,  # Allgemeinbildende Schule               [DIN277]
    3022: 0.70,  # Berufsbildende Schule                  [DIN277]
    3023: 0.68,  # Hochschule                             [DIN277]
    3024: 0.66,  # Forschungsinstitut — lab-heavy         [DIN277]
    3030: 0.65,  # Kulturell allgemein                    [DIN277]
    3031: 0.62,  # Schloss                                [DIN277]
    3032: 0.60,  # Theater/Oper — stage + fly tower       [DIN277]
    3033: 0.62,  # Konzertgebäude                         [DIN277]
    3034: 0.65,  # Museum                                 [DIN277]
    3035: 0.68,  # Rundfunk/Fernsehen                     [DIN277]
    3036: 0.65,  # Veranstaltungsgebäude                  [DIN277]
    3037: 0.72,  # Bibliothek                             [DIN277]
    3038: 0.58,  # Burg/Festung — thick walls             [DIN277]
    3040: 0.58,  # Religiös allgemein                     [DIN277]
    3041: 0.58,  # Kirche                                 [DIN277]
    3042: 0.60,  # Synagoge                               [DIN277]
    3043: 0.60,  # Kapelle                                [DIN277]
    3044: 0.70,  # Gemeindehaus                           [DIN277]
    3045: 0.62,  # Gotteshaus                             [DIN277]
    3046: 0.62,  # Moschee                                [DIN277]
    3050: 0.56,  # Gesundheitswesen allgemein             [DIN277]
    3051: 0.55,  # Krankenhaus — lowest EF                [DIN277]
    3052: 0.57,  # Heilanstalt/Pflegeanstalt              [DIN277]
    3060: 0.70,  # Soziale Zwecke allgemein               [DIN277]
    3061: 0.72,  # Jugendfreizeitheim                     [DIN277]
    3062: 0.73,  # Freizeit-/Vereinsheim                  [DIN277]
    3063: 0.70,  # Seniorenfreizeitstätte                 [DIN277]
    3064: 0.74,  # Obdachlosenheim                        [DIN277]
    3065: 0.72,  # Kita/Kinderkrippe                      [DIN277]
    3070: 0.70,  # Sicherheit/Ordnung allgemein           [DIN277]
    3071: 0.72,  # Polizei                                [DIN277]
    3072: 0.75,  # Feuerwehr                              [DIN277]
    3073: 0.74,  # Kaserne                                [DIN277]
    3074: 0.55,  # Schutzbunker                           [DIN277]
    3075: 0.65,  # Justizvollzugsanstalt                  [DIN277]
    3080: 0.68,  # Friedhofsgebäude                       [DIN277]
    3081: 0.65,  # Trauerhalle                            [DIN277]
    3082: 0.62,  # Krematorium                            [DIN277]
    3090: 0.72,  # Empfangsgebäude allgemein              [DIN277]
    3091: 0.70,  # Bahnhofsgebäude                        [DIN277]
    3092: 0.68,  # Flughafengebäude                       [DIN277]
    3094: 0.74,  # U-Bahnhof                              [DIN277]
    3095: 0.74,  # S-Bahnhof                              [DIN277]
    3097: 0.74,  # Busbahnhof                             [DIN277]
    3100: 0.72,  # Öffentl. Zwecke mit Wohnen             [DIN277]
    3200: 0.68,  # Erholungszwecke allgemein              [DIN277]
    3210: 0.70,  # Sportzwecke allgemein                  [DIN277]
    3211: 0.74,  # Sport-/Turnhalle                       [DIN277]
    3212: 0.70,  # Gebäude zum Sportplatz                 [DIN277]
    3220: 0.62,  # Badegebäude allgemein                  [DIN277]
    3221: 0.62,  # Hallenbad                              [DIN277]
    3222: 0.66,  # Freibad-Gebäude                        [DIN277]
    3240: 0.65,  # Kurbetrieb                             [DIN277]
    3241: 0.60,  # Badegebäude medizinisch                [DIN277]
    3242: 0.65,  # Sanatorium                             [DIN277]
    3260: 0.65,  # Zoo allgemein                          [DIN277]
    3261: 0.72,  # Empfangsgebäude Zoo                    [DIN277]
    3262: 0.68,  # Aquarium/Terrarium/Voliere             [DIN277]
    3263: 0.65,  # Tierschauhaus                          [DIN277]
    3264: 0.80,  # Stall im Zoo                           [DIN277]
    3270: 0.65,  # Botanischer Garten allgemein           [DIN277]
    3271: 0.70,  # Empfangsgebäude Botanik                [DIN277]
    3272: 0.80,  # Gewächshaus (Botanik)                  [DIN277]
    3273: 0.68,  # Pflanzenschauhaus                      [DIN277]
    3281: 0.75,  # Schutzhütte                            [DIN277]
    9998: EF_DEFAULT_UNKNOWN,
}

# ---------------------------------------------------------------------------
# m²/employee table (net usable floor area per employee position)
# ---------------------------------------------------------------------------
# Values represent the net usable floor area (NUF) allocated per employee
# position for each building function.  Multiply by EF to obtain gross
# floor area per employee (not required here — the cap uses NUF directly).
#
# Annotation key:
#   [ArbStättV]   Arbeitsstättenverordnung Anhang 1.2 (minimum 8 m²/person)
#   [gif BR]      gif Büroflächenreport 2022/2023 Berlin
#   [RICS GOCS]   RICS Global Occupancy Costs Survey 2023
#   [StaBu EH]    Statistisches Bundesamt, Strukturerhebung Einzelhandel
#   [IHA]         IHA Hotelmarkt Deutschland 2023
#   [DKG]         Deutsche Krankenhausgesellschaft staffing/area norms
#   [BVL]         BVL Logistikimmobilien report (Bundesvereinigung Logistik)
#   [DEHOGA]      DEHOGA Branchenbericht Gastronomie
#   [KMK]         KMK Lehrerarbeitsstätten / school staffing guidelines
#   [INF]         Effectively infinite — building type has no meaningful
#                 employee capacity; prevents accidental cap enforcement
#
GFK_M2_PER_EMPLOYEE: dict[int, float] = {
    # ── 1000 Wohngebäude — no commercial employees ───────────────────────
    1000: 999.0,  # [INF]
    1010: 999.0,  # [INF]
    1020: 999.0,  # [INF]
    1021: 999.0,  # [INF]
    1022:  35.0,  # Seniorenheim — care staff: ~1 staff/35 m² [DKG analogy]
    1023: 999.0,  # [INF]
    1024: 999.0,  # [INF]
    1025: 999.0,  # [INF]
    1100:  20.0,  # Mixed — split residential/commercial     [gif BR]
    1110:  20.0,  # Wohngebäude mit Gemeinbedarf             [gif BR]
    1120:  15.0,  # Wohngebäude mit Handel                   [StaBu EH]
    1130:  30.0,  # Wohngebäude mit Gewerbe                  [ArbStättV]
    1210: 999.0,  # [INF]
    1220: 999.0,  # [INF]
    1223: 999.0,  # [INF]
    1310: 999.0,  # [INF]
    1311: 999.0,  # [INF]
    1312: 999.0,  # [INF]
    1313: 999.0,  # [INF]
    # ── 2000 Wirtschaft & Gewerbe ────────────────────────────────────────
    # Office types: Berlin average from gif BR 2022/2023 = 13 m² NIA/worker
    # post-COVID ABW trend pushes toward 12–14 m²; 12 m² adopted as
    # conservative cap (tight packing = max headcount scenario)
    2000:  12.0,  # [gif BR]
    2010:  12.0,  # Handel+Dienstleistungen (offices)        [gif BR / RICS GOCS]
    2020:  12.0,  # Bürogebäude                              [gif BR / RICS GOCS]
    2030:  13.0,  # Kreditinstitut — slightly more open      [gif BR]
    2040:  12.0,  # Versicherung                             [gif BR]
    # Retail: selling floor + back-office + stockroom
    # Statistisches Bundesamt EH Strukturerhebung: ~8–10 m² selling/employee
    # including stockroom / loading → 15–20 m² total per employee
    2050:  18.0,  # Geschäftsgebäude                         [StaBu EH]
    2051:  20.0,  # Kaufhaus — large floor, staff spread out [StaBu EH]
    2052:  20.0,  # Einkaufszentrum                          [StaBu EH]
    2053:  15.0,  # Markthalle — dense stall staffing        [StaBu EH]
    2054:  15.0,  # Laden — small shop                       [StaBu EH]
    2055:   8.0,  # Kiosk — 1–2 staff, tiny space            [ArbStättV min]
    2060:  30.0,  # Messehalle — event/exhibition staffing   [ArbStättV]
    # Hotels: housekeeping + reception + F&B + management
    # IHA 2023: average German hotel ~0.4–0.5 FTE/room; ~20–30 m²/FTE
    2070:  25.0,  # Beherbergung allgemein                   [IHA]
    2071:  25.0,  # Hotel/Motel/Pension                      [IHA]
    2072:  20.0,  # Jugendherberge — leaner staffing         [IHA]
    2074:  20.0,  # Campingplatzgebäude                      [ArbStättV]
    # Restaurants: DEHOGA — dense kitchen + front-of-house
    2080:  12.0,  # Bewirtung allgemein                      [DEHOGA]
    2081:  12.0,  # Gaststätte/Restaurant — kitchen+dining   [DEHOGA]
    2083:  12.0,  # Kantine                                  [DEHOGA]
    2090:  20.0,  # Freizeit/Vergnüg.                        [ArbStättV]
    2091:  25.0,  # Festsaal                                 [ArbStättV]
    2092:  25.0,  # Kino                                     [ArbStättV]
    2093:  30.0,  # Kegel-/Bowlinghalle                      [ArbStättV]
    2094:  20.0,  # Spielkasino                              [ArbStättV]
    # Industrial / manufacturing
    # StaBu Verarbeitendes Gewerbe 2022: varies 20–80 m²/employee by sector
    # 35 m² is a reasonable mid-point for mixed Berlin industrial stock
    2100:  35.0,  # Gewerbe/Industrie allgemein              [ArbStättV]
    2111:  35.0,  # Fabrik                                   [ArbStättV]
    2120:  25.0,  # Werkstatt — denser than factory          [ArbStättV]
    2130:  30.0,  # Tankstelle — forecourt + workshop bays   [ArbStättV]
    # Logistics: BVL report — automated warehouses >100 m²/worker;
    # manual operations 40–60 m²/worker; 80 m² adopted
    2140:  80.0,  # Vorratshaltung allgemein                 [BVL]
    2141:  80.0,  # Kühlhaus — automated cold store          [BVL]
    2142:  90.0,  # Speichergebäude                          [BVL]
    2143:  80.0,  # Lagerhalle                               [BVL]
    2150:  40.0,  # Spedition — more admin per m²            [BVL]
    # Research
    2160:  18.0,  # Forschungsgebäude — lab benches          [ArbStättV]
    2180:  15.0,  # Betriebliche Sozialeinrichtung           [ArbStättV]
    2200:  30.0,  # Sonstiges Gewerbe                        [ArbStättV]
    2211: 999.0,  # Windmühle                                [INF]
    2212: 999.0,  # Wassermühle                              [INF]
    2213: 999.0,  # Schöpfwerk                               [INF]
    2310:  15.0,  # Handel+DL mit Wohnen                     [gif BR]
    2320:  30.0,  # Gewerbe+Industrie mit Wohnen             [ArbStättV]
    2400:  15.0,  # Betriebsgeb. Verkehrsanlagen             [ArbStättV]
    2410:  15.0,  # Betriebsgeb. Straßenverkehr              [ArbStättV]
    2411:  15.0,  # Straßenmeisterei                         [ArbStättV]
    2412:  30.0,  # Wartungshalle                            [ArbStättV]
    2420:  15.0,  # Betriebsgeb. Schienenverkehr             [ArbStättV]
    2421: 999.0,  # Bahnwärterhaus                           [INF]
    2422:  50.0,  # Lokschuppen                              [ArbStättV]
    2423:  10.0,  # Stellwerk — dense control room           [ArbStättV]
    2424:  20.0,  # Betriebsgeb. Güterbahnhof                [ArbStättV]
    2430:  20.0,  # Betriebsgeb. Flugverkehr                 [ArbStättV]
    2431: 100.0,  # Flugzeughalle — very few staff/m²        [ArbStättV]
    2440:  20.0,  # Betriebsgeb. Schiffsverkehr              [ArbStättV]
    2441: 100.0,  # Werft                                    [ArbStättV]
    2442: 100.0,  # Dock                                     [ArbStättV]
    2443:  15.0,  # Betriebsgeb. Schleuse                    [ArbStättV]
    2444:  50.0,  # Bootshaus                                [ArbStättV]
    2460: 999.0,  # Parken — [INF/N/A]
    2461: 999.0,  # Parkhaus                                 [INF]
    2462: 999.0,  # Parkdeck                                 [INF]
    2463: 999.0,  # Garage                                   [INF]
    2464:  50.0,  # Fahrzeughalle — maintenance staff        [ArbStättV]
    2465: 999.0,  # Tiefgarage                               [INF]
    2500:  20.0,  # Versorgung allgemein                     [ArbStättV]
    2510:  20.0,  # Wasserversorgung                         [ArbStättV]
    2511:  30.0,  # Wasserwerk                               [ArbStättV]
    2512:  30.0,  # Pumpstation                              [ArbStättV]
    2513: 999.0,  # Wasserbehälter                           [INF]
    2520:  20.0,  # Elektrizitätsversorgung                  [ArbStättV]
    2521:  30.0,  # Elektrizitätswerk                        [ArbStättV]
    2522:  50.0,  # Umspannwerk — few operators              [ArbStättV]
    2523: 999.0,  # Umformer — unmanned                      [INF]
    2540:  15.0,  # Fernmeldewesen                           [ArbStättV]
    2560: 999.0,  # Gebäude an unterird. Leitungen           [INF]
    2571:  30.0,  # Gaswerk                                  [ArbStättV]
    2580:  25.0,  # Heizwerk                                 [ArbStättV]
    2591: 999.0,  # Pumpwerk                                 [INF]
    2600:  25.0,  # Entsorgung allgemein                     [ArbStättV]
    2610:  25.0,  # Abwasserbeseitigung                      [ArbStättV]
    2611:  30.0,  # Kläranlage                               [ArbStättV]
    2612: 999.0,  # Toilette                                 [INF]
    2620:  30.0,  # Abfallbehandlung                         [ArbStättV]
    2621: 999.0,  # Müllbunker                               [INF]
    2622:  40.0,  # Müllverbrennung                          [ArbStättV]
    2623: 999.0,  # Abfalldeponie                            [INF]
    2700: 999.0,  # Landwirtschaft                           [INF]
    2720: 999.0,  # Betriebsgebäude Landwirtschaft           [INF]
    2721: 999.0,  # Scheune                                  [INF]
    2723: 999.0,  # Schuppen                                 [INF]
    2724: 999.0,  # Stall                                    [INF]
    2726: 999.0,  # Scheune und Stall                        [INF]
    2727: 999.0,  # Stall für Tiergroßhaltung                [INF]
    2735: 999.0,  # Jagdhaus                                 [INF]
    2741: 999.0,  # Treibhaus                                [INF]
    2742: 999.0,  # Gewächshaus                              [INF]
    # ── 3000 Öffentliche Zwecke ──────────────────────────────────────────
    3000:  15.0,  # allgemein                                [ArbStättV]
    3010:  13.0,  # Verwaltungsgebäude — public-sector office [gif BR]
    3011:  15.0,  # Parlament                                [ArbStättV]
    3012:  13.0,  # Rathaus                                  [gif BR]
    3013:  12.0,  # Post — counter + sorting + admin         [ArbStättV]
    3014:  13.0,  # Zollamt                                  [ArbStättV]
    3015:  15.0,  # Gericht                                  [ArbStättV]
    3016:  13.0,  # Botschaft/Konsulat                       [ArbStättV]
    # Schools: KMK norms → ~1 teacher per 20–25 m² classroom;
    # full staff incl. admin + janitors → ~20–25 m²/FTE
    3020:  22.0,  # Bildung/Forschung allgemein              [KMK]
    3021:  22.0,  # Allgemeinbildende Schule                 [KMK]
    3022:  20.0,  # Berufsbildende Schule — workshops        [KMK]
    3023:  20.0,  # Hochschule                               [KMK]
    3024:  18.0,  # Forschungsinstitut — lab-heavy           [ArbStättV]
    3030:  25.0,  # Kulturell allgemein                      [ArbStättV]
    3031:  25.0,  # Schloss                                  [ArbStättV]
    3032:  20.0,  # Theater/Oper — tech+artistic crew        [ArbStättV]
    3033:  20.0,  # Konzertgebäude                           [ArbStättV]
    3034:  25.0,  # Museum — exhibition + curatorial staff   [ArbStättV]
    3035:  12.0,  # Rundfunk/Fernsehen — dense studios       [ArbStättV]
    3036:  20.0,  # Veranstaltungsgebäude                    [ArbStättV]
    3037:  20.0,  # Bibliothek                               [ArbStättV]
    3038:  30.0,  # Burg/Festung                             [ArbStättV]
    3040:  50.0,  # Religiös allgemein — few paid staff      [ArbStättV]
    3041:  50.0,  # Kirche                                   [ArbStättV]
    3042:  50.0,  # Synagoge                                 [ArbStättV]
    3043:  50.0,  # Kapelle                                  [ArbStättV]
    3044:  20.0,  # Gemeindehaus                             [ArbStättV]
    3045:  50.0,  # Gotteshaus                               [ArbStättV]
    3046:  50.0,  # Moschee                                  [ArbStättV]
    # Hospitals: DKG area/staff norms — roughly 30–40 m² NUF per FTE
    # (includes patient areas, nursing staff, ancillary, admin)
    3050:  35.0,  # Gesundheitswesen allgemein               [DKG]
    3051:  35.0,  # Krankenhaus                              [DKG]
    3052:  30.0,  # Heilanstalt/Pflegeanstalt — care-heavy   [DKG]
    3060:  20.0,  # Soziale Zwecke allgemein                 [ArbStättV]
    3061:  20.0,  # Jugendfreizeitheim                       [ArbStättV]
    3062:  20.0,  # Freizeit-/Vereinsheim                    [ArbStättV]
    3063:  20.0,  # Seniorenfreizeitstätte                   [ArbStättV]
    3064:  20.0,  # Obdachlosenheim                          [ArbStättV]
    3065:  15.0,  # Kita — dense caregiver ratio             [ArbStättV]
    3070:  15.0,  # Sicherheit/Ordnung allgemein             [ArbStättV]
    3071:  13.0,  # Polizei — office + operations rooms      [ArbStättV]
    3072:  30.0,  # Feuerwehr — garage dominates             [ArbStättV]
    3073:  25.0,  # Kaserne                                  [ArbStättV]
    3074: 999.0,  # Schutzbunker                             [INF]
    3075:  20.0,  # Justizvollzugsanstalt                    [ArbStättV]
    3080:  30.0,  # Friedhofsgebäude                         [ArbStättV]
    3081:  30.0,  # Trauerhalle                              [ArbStättV]
    3082:  30.0,  # Krematorium                              [ArbStättV]
    3090:  15.0,  # Empfangsgebäude allgemein                [ArbStättV]
    3091:  15.0,  # Bahnhofsgebäude                          [ArbStättV]
    3092:  15.0,  # Flughafengebäude                         [ArbStättV]
    3094:  15.0,  # U-Bahnhof                                [ArbStättV]
    3095:  15.0,  # S-Bahnhof                                [ArbStättV]
    3097:  15.0,  # Busbahnhof                               [ArbStättV]
    3100:  15.0,  # Öffentl. Zwecke mit Wohnen               [ArbStättV]
    3200:  25.0,  # Erholungszwecke allgemein                [ArbStättV]
    3210:  25.0,  # Sportzwecke allgemein                    [ArbStättV]
    3211:  30.0,  # Sport-/Turnhalle                         [ArbStättV]
    3212:  25.0,  # Gebäude zum Sportplatz                   [ArbStättV]
    3220:  25.0,  # Badegebäude allgemein                    [ArbStättV]
    3221:  25.0,  # Hallenbad                                [ArbStättV]
    3222:  25.0,  # Freibad-Gebäude                          [ArbStättV]
    3240:  25.0,  # Kurbetrieb                               [ArbStättV]
    3241:  30.0,  # Badegebäude medizinisch                  [DKG analogy]
    3242:  30.0,  # Sanatorium                               [DKG analogy]
    3260:  25.0,  # Zoo allgemein                            [ArbStättV]
    3261:  15.0,  # Empfangsgebäude Zoo                      [ArbStättV]
    3262:  25.0,  # Aquarium/Terrarium/Voliere               [ArbStättV]
    3263:  25.0,  # Tierschauhaus                            [ArbStättV]
    3264:  50.0,  # Stall im Zoo — few keepers               [ArbStättV]
    3270:  30.0,  # Botanischer Garten allgemein             [ArbStättV]
    3271:  15.0,  # Empfangsgebäude Botanik                  [ArbStättV]
    3272:  30.0,  # Gewächshaus (Botanik)                    [ArbStättV]
    3273:  25.0,  # Pflanzenschauhaus                        [ArbStättV]
    3281:  50.0,  # Schutzhütte                              [ArbStättV]
    9998:  15.0,  # Unspecified                              [default]
}

# GFK codes exempt from hochhaus EF penalty (physically meaningless)
_HOCHHAUS_EXEMPT_GFK: frozenset[int] = frozenset({
    2055, 2060, 2130,
    2140, 2141, 2142, 2143,
    2211, 2212, 2213,
    2460, 2461, 2462, 2463, 2464, 2465,
    2500, 2510, 2511, 2512, 2513,
    2520, 2521, 2522, 2523,
    2540, 2560, 2571, 2580, 2591,
    2600, 2610, 2611, 2612,
    2620, 2621, 2622, 2623,
    2700, 2720, 2721, 2723, 2724, 2726, 2727, 2735, 2741, 2742,
    3041, 3043,
    3074, 3081, 3082,
    3281,
    1313,
})


# ===========================================================================
# Public API
# ===========================================================================

def get_efficiency_factor(
    gfk: Optional[int],
    hochhaus: bool = False,
) -> float:
    """Return the floor-space efficiency factor (NUF/BGF) for a building.

    Parameters
    ----------
    gfk:
        ALKIS Gebäudefunktion code (integer).  None / NaN → fallback.
    hochhaus:
        True if the building is classified as Hochhaus (≥22 m / ≥8 storeys).
        Applies HOCHHAUS_PENALTY for types where a structural core matters.

    Returns
    -------
    float
        Efficiency factor in (0, 1).
    """
    gfk = _parse_gfk(gfk)
    if gfk is None:
        return EF_DEFAULT_UNKNOWN

    base_ef = GFK_BASE_EFFICIENCY.get(gfk)
    if base_ef is None:
        group = (gfk // 1000) * 1000
        base_ef = _GROUP_EF_DEFAULTS.get(group, EF_DEFAULT_UNKNOWN)

    if not hochhaus or gfk in _HOCHHAUS_EXEMPT_GFK:
        return base_ef

    return max(EF_MIN, base_ef - HOCHHAUS_PENALTY)


def get_m2_per_employee(gfk: Optional[int]) -> float:
    """Return net usable floor area (m²) per employee position for a GFK code.

    Parameters
    ----------
    gfk:
        ALKIS Gebäudefunktion code.

    Returns
    -------
    float
        m² of net usable floor area per FTE/workstation.
        Returns 999.0 for building types with no meaningful employee capacity
        (residential, unmanned infrastructure, parking), which effectively
        makes the resulting hard cap infinite.
    """
    gfk = _parse_gfk(gfk)
    if gfk is None:
        return M2_DEFAULT_UNKNOWN

    m2 = GFK_M2_PER_EMPLOYEE.get(gfk)
    if m2 is None:
        group = (gfk // 1000) * 1000
        m2 = _GROUP_M2_DEFAULTS.get(group, M2_DEFAULT_UNKNOWN)

    return m2


def compute_employee_hard_cap(
    footprint_m2: float,
    num_floors: Optional[int],
    gfk: Optional[int],
    hochhaus: bool = False,
) -> float:
    """Compute the physical employee hard cap H for a building.

    H = (footprint × floors × EF) / m²_per_employee

    Parameters
    ----------
    footprint_m2:
        Building footprint area in m² (from polygon geometry).
    num_floors:
        Number of above-ground storeys.  None / 0 / NaN → treated as 1.
    gfk:
        ALKIS Gebäudefunktion code.
    hochhaus:
        Whether the building is classified as Hochhaus.

    Returns
    -------
    float
        Maximum number of employees the building can physically accommodate.
        Returns ``np.inf`` when the GFK type has no meaningful cap (e.g.
        residential, unmanned infrastructure).
    """
    floors = _parse_floors(num_floors)
    ef = get_efficiency_factor(gfk, hochhaus)
    m2_per = get_m2_per_employee(gfk)

    if m2_per >= 999.0:
        return np.inf

    usable = footprint_m2 * floors * ef
    return usable / m2_per

def compute_usable_floor_area(
    footprint_m2: float,
    num_floors: Optional[int],
    gfk: Optional[int],
    hochhaus: bool = False,
) -> float:
    """Compute approximate net usable floor area for a building.

    Parameters
    ----------
    footprint_m2:
        Building footprint area in square metres (from polygon geometry).
    num_floors:
        Number of above-ground storeys (anzahl_der_oberirdischen_geschosse).
        None / 0 / NaN is treated as 1.
    gfk:
        ALKIS Gebäudefunktion code.
    hochhaus:
        Whether the building is classified as Hochhaus.

    Returns
    -------
    float
        Approximate net usable floor area in m².
    """
    floors = _parse_floors(num_floors)
    ef = get_efficiency_factor(gfk, hochhaus)
    return footprint_m2 * floors * ef

def apply_hard_cap_single(reported: float, hard_cap: float) -> float:
    """Enforce the employee hard cap for a single-company building.

    Returns ``min(reported, hard_cap)``.

    Parameters
    ----------
    reported:
        IHK-reported employee count (midpoint of Beschäftigtengrößenklasse).
    hard_cap:
        Physical hard cap H from ``compute_employee_hard_cap``.

    Returns
    -------
    float
        Capped employee count.
    """
    if np.isnan(reported):
        return reported
    return float(min(reported, hard_cap))


def apply_hard_cap_multi(
    reported: "pd.Series",
    hard_cap: float,
) -> pd.Series:
    """Enforce the employee hard cap for multiple companies in the same building.

    When the aggregate reported headcount exceeds the hard cap H, a
    **proportional scaling** rule is applied to preserve relative firm sizes:

        X_i_capped = X_i × min(1.0, H / Σ X_j)

    For a single company the rule reduces to ``min(X, H)``.

    The scaling factor is computed on **valid (non-NaN) entries only**; NaN
    entries are passed through unchanged.

    Parameters
    ----------
    reported:
        ``pd.Series`` of IHK-reported employee midpoints for all companies
        sharing the building.  Index is arbitrary.
    hard_cap:
        Physical hard cap H (total for the building).

    Returns
    -------
    pd.Series
        Scaled employee counts with the same index as ``reported``.
    """
    if np.isinf(hard_cap):
        return reported

    valid_mask = reported.notna()
    total = reported[valid_mask].sum()

    if total <= 0:
        return reported

    scale = min(1.0, hard_cap / total)
    result = reported.copy()
    result[valid_mask] = result[valid_mask] * scale
    return result


# ===========================================================================
# Convenience wrapper for GeoDataFrame use
# ===========================================================================

def enrich_gebaeude(
    gdf: "gpd.GeoDataFrame",  # type: ignore[name-defined]
    gfk_col: str = "gfk",
    hochhaus_col: str = "hochhaus",
    floors_col: str = "anzahl_der_oberirdischen_geschosse",
) -> gpd.GeoDataFrame:
    """Add efficiency, usable area, and employee hard cap columns to a building GDF.

    Adds three columns in-place and returns the GDF for chaining:
      - ``efficiency``          NUF/BGF ratio
      - ``usable_area_m2``      footprint × floors × efficiency  [m²]
      - ``employee_hard_cap``   maximum physical employee count

    Parameters
    ----------
    gdf:
        GeoDataFrame with ALKIS building polygons (EPSG:3035 expected so
        that ``.geometry.area`` yields m²).
    gfk_col, hochhaus_col, floors_col:
        Column name overrides.
    """
    gdf = gdf.copy()

    def _ef(row: "pd.Series") -> float:  # type: ignore[name-defined]
        return get_efficiency_factor(row.get(gfk_col), bool(row.get(hochhaus_col, False)))

    def _cap(row: "pd.Series") -> float:  # type: ignore[name-defined]
        return compute_employee_hard_cap(
            row.geometry.area,
            row.get(floors_col),
            row.get(gfk_col),
            bool(row.get(hochhaus_col, False)),
        )

    gdf["efficiency"] = gdf.apply(_ef, axis=1)
    gdf["usable_area_m2"] = (
        gdf.geometry.area
        * gdf[floors_col].clip(lower=1).fillna(1)
        * gdf["efficiency"]
    )
    gdf["employee_hard_cap"] = gdf.apply(_cap, axis=1)
    return gdf


# ===========================================================================
# Internal helpers
# ===========================================================================

def _parse_gfk(gfk: object) -> Optional[int]:
    if gfk is None:
        return None
    try:
        v = int(gfk)
        return v if v > 0 else None
    except (ValueError, TypeError):
        return None


def _parse_floors(num_floors: object) -> int:
    try:
        v = int(num_floors)
        return max(1, v)
    except (ValueError, TypeError):
        return 1
