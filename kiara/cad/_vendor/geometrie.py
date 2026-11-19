#!/usr/bin/env python3
"""B-Rep-Geometrie aus Creo-.pkg lesen.

  geometrie.py datei.pkg   Übersicht je Teil mit Selbstprüfung; Rückgabewert 1 bei Befund

Als Modul: teile(pfad) -> {teil_id: brep}. Koordinaten lokal im Teil, mm, Winkel im Bogenmaß.
Unterstützt: Ebene, Zylinder, Kugel, Kegel, Torus, B-Spline-Fläche; Gerade, Kreis, Ellipse, B-Spline (auch als
Näherung von Schnittkurven; Isolinien exakt). Andere Klassen erscheinen als typ "unbekannt" mit Klassennamen.
"""
import math
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from altformat import oeffnen  # noqa: E402
from pkgleser import ist_zahl, koerperart, kopf_lesen, segmente_lesen, zahl  # noqa: E402

TOL = 1e-6


def kreuz(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def skal(a, b):
    return sum(x * y for x, y in zip(a, b))


def plus(a, b, f=1.0):
    return [x + f * y for x, y in zip(a, b)]


def laenge(a):
    return math.sqrt(skal(a, a))


def euklidisch(q):
    """Homogenen Pol (w·x, …, w) in den Punkt zurückrechnen."""
    return [x / q[-1] for x in q[:-1]]


def pol_punkte(k):
    """Pole eines B-Splines als Punkte (rationale Splines halten sie homogen, siehe bspline)."""
    if k.get("rational"):
        return [euklidisch(q) for q in k["pole"]]
    return k["pole"]


def bspline(o, dim):
    """k2_bspline_cv/cv2: B nbp, nbp Stützstellen, B rational, B ?, B n, n·dim Pole, B nk, nk Knoten, B Ordnung.
    Rational (Kundenmodelle, 31.10.2026): danach B (64 oder 0), B n, n Gewichte; die Pole stehen kartesisch. Intern
    homogen (w·x, …, w) mit "rational": True – de Boor, Knoteneinfügen, Umkehren und Zuschneiden rechnen unverändert,
    Punkte entstehen über euklidisch()."""
    felder = iter([t for t in o.felder if t.art == "B" or ist_zahl(t)])

    def b():
        t = next(felder)
        if t.art != "B":
            raise ValueError(f"{o.klasse} #{o.nr}: B erwartet, {t.art} gefunden")
        return t.wert

    def n_zahlen(k):
        return [zahl(next(felder)) for _ in range(k)]

    nbp = b()
    n_zahlen(nbp)
    rational, _, n = b(), b(), b()
    pole = n_zahlen(n * dim)
    knoten = n_zahlen(b())
    ordnung = b()
    phantomknoten_klemmen(knoten, ordnung - 1)
    P = [pole[i * dim:(i + 1) * dim] for i in range(n)]
    k = {"typ": "bspline", "grad": ordnung - 1, "knoten": knoten, "pole": P}
    if rational:
        b()                                                         # 64 oder 0, Bedeutung offen
        w = n_zahlen(b())
        if len(w) != n or min(w) <= 0:
            raise ValueError(f"{o.klasse} #{o.nr}: {len(w)} Gewichte für {n} Pole")
        k["pole"] = [[x * wi for x in q] + [wi] for q, wi in zip(P, w)]
        k["rational"] = True
    return bspline_pruefen(k, f"{o.klasse} #{o.nr}")


def phantomknoten_klemmen(knoten, grad):
    """SolidDesigner 11 schreibt außen je einen Phantomknoten (Endvielfachheit = Grad statt Grad + 1); auf
    [U[p], U[n]] ist die Kurve dieselbe wie mit geklemmten Enden. Bringt den Vektor in-place in die geklemmte Form."""
    p, U = grad, knoten
    if len(U) > 2 * p + 1:
        if U[0] < U[p] and all(abs(U[i] - U[p]) < 1e-12 for i in range(1, p + 1)):
            U[0] = U[p]
        if U[-1] > U[-p - 1] and all(abs(U[-i] - U[-p - 1]) < 1e-12 for i in range(2, p + 2)):
            U[-1] = U[-p - 1]
    return knoten


def bspline_pruefen(k, wo="B-Spline"):
    """Strukturprüfung: Knotenzahl = Pole + Grad + 1, Knoten nicht fallend, geklemmte Enden,
    innere Vielfachheit höchstens Grad. Wirft ValueError, sonst gibt es k zurück."""
    p, U, n = k["grad"], k["knoten"], len(k["pole"])
    if len(U) != n + p + 1:
        raise ValueError(f"{wo}: {len(U)} Knoten passen nicht zu {n} Polen, Grad {p} (erwartet {n + p + 1})")
    if any(b < a for a, b in zip(U, U[1:])):
        raise ValueError(f"{wo}: Knoten nicht aufsteigend")
    # Ungeklemmte Knotenvektoren sind zulässig (STEP .UNSPECIFIED., Open CASCADE): gültiger Bereich ist dann
    # [U[p], U[n]]. Creo schreibt solche Flächen bei gerollten Blechen (Praxistest 30.10.2026, Ansaugkonus).
    for u in set(U[p + 1:-p - 1]):
        if U.count(u) > p:
            raise ValueError(f"{wo}: innerer Knoten {u:g} mit Vielfachheit {U.count(u)} > Grad {p}")
    return k


def bspline_flaeche(o):
    """k2_bspline_sf: B m, m×(Zahl, B Kennung), B nbu, nbu Stützstellen u, B nbv, nbv Stützstellen v, B rational, B×3, B n,
    n×3 Pole (u läuft schnell), B nku, nku Knoten u, B nkv, nkv Knoten v, B Ordnung u, B Ordnung v, B×2."""
    felder = iter([t for t in o.felder if t.art == "B" or ist_zahl(t)])

    def nimm():
        try:
            return next(felder)
        except StopIteration:
            raise ValueError(f"{o.klasse} #{o.nr}: unvollständige B-Spline-Fläche") from None

    def b():
        t = nimm()
        if t.art != "B":
            raise ValueError(f"{o.klasse} #{o.nr}: B erwartet, {t.art} gefunden")
        return t.wert

    def n_zahlen(k):
        if type(k) is not int or k < 0:
            raise ValueError(f"{o.klasse} #{o.nr}: ungültige Zahlenanzahl {k}")
        out = []
        for _ in range(k):
            t = nimm()
            if not ist_zahl(t):
                raise ValueError(f"{o.klasse} #{o.nr}: Zahl erwartet, {t.art} gefunden")
            out.append(zahl(t))
        return out

    vorspann = b()
    # Belegt: 0/1 im Bestandskorpus, 2 bei nativer VarN-Verrundung (#379).
    # Paarbedeutung offen; die Zahlen und Kennungen sind keine Polkoordinaten.
    if vorspann not in (0, 1, 2):
        raise ValueError(f"{o.klasse} #{o.nr}: unbelegter B-Spline-Vorspann {vorspann}")
    for _ in range(vorspann):
        n_zahlen(1)
        b()
    n_zahlen(b())
    n_zahlen(b())
    rational, _, zu_u, zu_v, n = b(), b(), b(), b(), b()       # zu: 2 = in u bzw. v geschlossen (Lauf 11)
    pole = n_zahlen(3 * n)
    ku = n_zahlen(b())
    kv = n_zahlen(b())
    ou, ov = b(), b()
    nu, nv = len(ku) - ou, len(kv) - ov
    if nu * nv != n:
        raise ValueError(f"{o.klasse} #{o.nr}: {n} Pole passen nicht zu {nu}×{nv}")
    P = [pole[i * 3:(i + 1) * 3] for i in range(n)]
    if rational:                                                    # wie bei Kurven: B 64, B n, n Gewichte; homogen
        b()
        w = n_zahlen(b())
        if len(w) != n or min(w) <= 0:
            raise ValueError(f"{o.klasse} #{o.nr}: {len(w)} Gewichte für {n} Pole")
        P = [[x * wi for x in q] + [wi] for q, wi in zip(P, w)]
    gitter = [[P[j * nu + i] for j in range(nv)] for i in range(nu)]
    bspline_pruefen({"grad": ou - 1, "knoten": ku, "pole": gitter}, f"{o.klasse} #{o.nr} u")
    bspline_pruefen({"grad": ov - 1, "knoten": kv, "pole": gitter[0]}, f"{o.klasse} #{o.nr} v")
    return {"typ": "bspline_flaeche", "grad_u": ou - 1, "grad_v": ov - 1,
            "knoten_u": ku, "knoten_v": kv, "pole": gitter, "flag_zu": (zu_u == 2, zu_v == 2),
            **({"rational": True} if rational else {})}


def sweep_als_bspline(sw, v0, v1):
    """Parallelverschobene Profilkurve als B-Spline-Fläche vom Grad (p, 1) über v ∈ [v0, v1]."""
    k, d = sw["profil"], sw["d"]
    if k.get("rational"):                                           # homogen: w·(x + v·d), w
        gitter = [[plus(q[:-1], d, v0 * q[-1]) + [q[-1]], plus(q[:-1], d, v1 * q[-1]) + [q[-1]]] for q in k["pole"]]
    else:
        gitter = [[plus(q, d, v0), plus(q, d, v1)] for q in k["pole"]]
    return {"typ": "bspline_flaeche", "grad_u": k["grad"], "grad_v": 1, "knoten_u": list(k["knoten"]),
            "knoten_v": [v0, v0, v1, v1], "pole": gitter, **({"rational": True} if k.get("rational") else {})}


def flaeche_punkt(f, u, v):
    """Punkt in nativen Parametern: Ebene/Zylinder oder rationale Tensorfläche.

    Analytische Trägerachsen bleiben von der Parameterbasis getrennt. Bei
    alten handgebauten Dictionaries ohne parameter gilt der Einheitsmaßstab.
    """
    if f["typ"] == "ebene":
        p = f.get("parameter", {"u": f["u"], "v": f["v"], "su": 1., "sv": 1.})
        return plus(plus(f["p"], p["u"], u * p["su"]), p["v"], v * p["sv"])
    if f["typ"] == "zylinder":
        p = f.get("parameter", {"n": f["a"], "sv": 1.})
        y = kreuz(p["n"], f["x"])
        return [m + f["r"] * (math.cos(u)*x + math.sin(u)*yy) + v*p["sv"]*a
                for m, x, yy, a in zip(f["m"], f["x"], y, f["a"])]
    nu, nv = len(f["pole"]), len(f["pole"][0])
    lo_u, hi_u = f["knoten_u"][f["grad_u"]], f["knoten_u"][nu]
    lo_v, hi_v = f["knoten_v"][f["grad_v"]], f["knoten_v"][nv]
    if not (lo_u <= u <= hi_u and lo_v <= v <= hi_v):
        # In geschlossener Richtung setzt sich die Fläche periodisch fort (Flag 2): Parameter außerhalb [U[p], U[n]]
        # modulo Periode zurückfalten statt den Randabschnitt zu extrapolieren (Bogen 3002728, Konverterfälle
        # 7.11.2026: Coedge-Ende u = 0,752 auf [−1; 0], extrapoliert 1.798 mm daneben). geschlossen() wertet nur
        # innerhalb des Abschnitts aus.
        zu = geschlossen(f)
        if zu[0] and not lo_u <= u <= hi_u:
            u = lo_u + (u - lo_u) % (hi_u - lo_u)
        if zu[1] and not lo_v <= v <= hi_v:
            v = lo_v + (v - lo_v) % (hi_v - lo_v)
    zw = [de_boor({"grad": f["grad_u"], "knoten": f["knoten_u"], "pole": [f["pole"][i][j] for i in range(nu)]}, u)
          for j in range(nv)]
    q = de_boor({"grad": f["grad_v"], "knoten": f["knoten_v"], "pole": zw}, v)
    return euklidisch(q) if f.get("rational") else q


def flaeche_uv(f, punkt):
    """Inverse native Parameter für Ebene/Zylinder; Zylinderwinkel als Hauptwert.

    Die Ebenenbasis darf nach einer affinen Abbildung schief sein. Eine
    singuläre Parameterbasis ist nicht invertierbar. pkgbau.uv_von benutzt
    hingegen das eigene, neu erzeugte STEP/PKG-Schema und Einheitsparameter.
    """
    if f["typ"] == "ebene":
        p = f.get("parameter", {"u": f["u"], "v": f["v"], "su": 1., "sv": 1.})
        a, b = [x*p["su"] for x in p["u"]], [x*p["sv"] for x in p["v"]]
        d = plus(punkt, f["p"], -1)
        aa, ab, bb = skal(a, a), skal(a, b), skal(b, b)
        det = aa*bb-ab*ab
        if not math.isfinite(det) or det <= 0:
            raise ValueError("Ebene mit singulärer Parameterbasis")
        da, db = skal(d, a), skal(d, b)
        return (da*bb-db*ab)/det, (db*aa-da*ab)/det
    if f["typ"] == "zylinder":
        p = f.get("parameter", {"n": f["a"], "sv": 1.})
        d, a = plus(punkt, f["m"], -1), f["a"]
        den = skal(a, a)*p["sv"]
        if not math.isfinite(den) or den == 0:
            raise ValueError("Zylinder mit singulärer Parameterachse")
        v = skal(d, a)/den
        radial = plus(d, a, -v*p["sv"])
        x, y = f["x"], kreuz(p["n"], f["x"])
        return math.atan2(skal(radial, y)/skal(y, y), skal(radial, x)/skal(x, x)), v
    raise ValueError(f"{f['typ']}: native inverse Parameter nicht umgesetzt")


def isolinie(f, fest, wert):
    """Isoparametrische Kurve einer B-Spline-Fläche als B-Spline: fest "u" → Kurve in v bei u = wert, fest "v" umgekehrt."""
    nu, nv = len(f["pole"]), len(f["pole"][0])
    if fest == "u":
        pole = [de_boor({"grad": f["grad_u"], "knoten": f["knoten_u"], "pole": [f["pole"][i][j] for i in range(nu)]}, wert)
                for j in range(nv)]
        return {"typ": "bspline", "grad": f["grad_v"], "knoten": list(f["knoten_v"]), "pole": pole,
                **({"rational": True} if f.get("rational") else {})}
    pole = [de_boor({"grad": f["grad_v"], "knoten": f["knoten_v"], "pole": f["pole"][i]}, wert) for i in range(nu)]
    return {"typ": "bspline", "grad": f["grad_u"], "knoten": list(f["knoten_u"]), "pole": pole,
            **({"rational": True} if f.get("rational") else {})}


def umkehren(k):
    """B-Spline rückwärts durchlaufen (Pole umgekehrt, Knoten gespiegelt)."""
    lo, hi = k["knoten"][0], k["knoten"][-1]
    return {**k, "pole": list(reversed(k["pole"])), "knoten": [lo + hi - u for u in reversed(k["knoten"])]}


def de_boor(k, t):
    p, U, P = k["grad"], k["knoten"], k["pole"]
    n = len(P)
    t = max(t, U[p])                                # Rundung knapp unter dem Bereichsanfang (Parameterkurven, 1e-10)
    s = n - 1 if t >= U[n] else max(i for i in range(p, n) if U[i] <= t)
    d = [list(P[j + s - p]) for j in range(p + 1)]
    for r in range(1, p + 1):
        for j in range(p, r - 1, -1):
            i = j + s - p
            nenner = U[i + p + 1 - r] - U[i]
            a = 0.0 if nenner == 0 else (t - U[i]) / nenner
            d[j] = [(1 - a) * x + a * y for x, y in zip(d[j - 1], d[j])]
    return d[p]


def knoten_einfuegen(k, t):
    """Boehm: Knoten t einmal einfügen, Kurve bleibt gleich."""
    p, U, P = k["grad"], k["knoten"], k["pole"]
    s = max(i for i in range(len(U) - 1) if U[i] <= t < U[i + 1])
    s = min(s, len(P) - 1)            # t am Ende des gültigen Bereichs eines ungeklemmten Splines: letzter Abschnitt
    Q = []
    for i in range(len(P) + 1):
        if i <= s - p:
            Q.append(P[i])
        elif i <= s:
            a = (t - U[i]) / (U[i + p] - U[i])
            Q.append([(1 - a) * x + a * y for x, y in zip(P[i - 1], P[i])])
        else:
            Q.append(P[i - 1])
    return {**k, "pole": Q, "knoten": U[:s + 1] + [t] + U[s + 1:]}


def teilkurve(k, t0, t1):
    """Abschnitt [t0, t1] eines B-Splines als eigener geklemmter B-Spline. Auch ungeklemmte Knotenvektoren: der
    Bereich wird auf den gültigen Abschnitt [U[p], U[n]] begrenzt und Werte bis 1e-7 neben einem Knoten auf ihn
    gelegt (Creos Biegekanten reichen um 1e-8 darüber hinaus, Praxistest 30.10.2026)."""
    p = k["grad"]
    U0, n = k["knoten"], len(k["pole"])
    t0, t1 = max(t0, U0[p]), min(t1, U0[n])

    def einrasten(t):
        nah = min(U0, key=lambda u: abs(u - t))
        return nah if abs(nah - t) <= 1e-7 * max(1.0, abs(t)) else t
    t0, t1 = einrasten(t0), einrasten(t1)
    for t in (t0, t1):
        while k["knoten"][0] < t < k["knoten"][-1] and k["knoten"].count(t) < p:
            k = knoten_einfuegen(k, t)
    U = k["knoten"]
    # Pol auf der Kurve: bei t0 über den letzten Knotenindex (gilt auch am geklemmten Anfang),
    # bei t1 über den ersten (gilt auch am geklemmten Ende); innere Knoten liegen strikt dazwischen
    letzter0 = max(i for i, u in enumerate(U) if u == t0)
    erster1 = min(i for i, u in enumerate(U) if u == t1)
    pole = k["pole"][letzter0 - p:erster1]
    knoten = [t0] * (p + 1) + U[letzter0 + 1:erster1] + [t1] * (p + 1)
    return bspline_pruefen({**k, "pole": pole, "knoten": knoten}, f"Teilkurve [{t0:g}, {t1:g}]")


def torus_aus(m, a, kreis, bogen=None):
    """Torus aus Achspunkt m, Achse a und Erzeugerkreis; bogen = (φ0, φ1) des benutzten Kreisbogens im
    Kreisparameter (altes Format). Liegt die Bogenmitte jenseits der Achse, ist der Zitronenteil eines
    Spindeltorus gemeint: R < 0 (hypot(ρ − R, h) − r bleibt die Abstandsformel); BRep.lesen ersetzt solche
    Flächen durch eine Rotationsfläche des Bogens (zitronen_umbauen)."""
    v = plus(kreis["m"], m, -1)
    radial = plus(v, a, -skal(v, a) / skal(a, a))
    R = laenge(radial)
    x = [c / R for c in radial]
    t = {"typ": "torus", "m": m, "a": a, "R": R, "r": kreis["r"], "x": x}
    if bogen:
        t["kreis"], t["bogen"] = kreis, list(bogen)
        w = plus(kurve_punkt(kreis, (bogen[0] + bogen[1]) / 2), m, -1)
        w = plus(w, a, -skal(w, a) / skal(a, a))
        if skal(w, x) < 0:
            t["R"] = -R
    return t


def _knotenspanne(U, n, p, t):
    if t >= U[n]:
        return n - 1
    lo, hi = p, n
    while True:
        mid = (lo + hi) // 2
        if t < U[mid]:
            hi = mid
        elif t >= U[mid + 1]:
            lo = mid
        else:
            return mid


def _basis(U, p, i, t):
    """Die p + 1 an der Spanne i nicht verschwindenden Basisfunktionen (The NURBS Book, A2.2)."""
    N, links, rechts = [1.0], [0.0] * (p + 1), [0.0] * (p + 1)
    for j in range(1, p + 1):
        links[j], rechts[j] = t - U[i + 1 - j], U[i + j] - t
        rest = 0.0
        for r in range(j):
            tmp = N[r] / (rechts[r + 1] + links[j - r])
            N[r] = rest + rechts[r + 1] * tmp
            rest = links[j - r] * tmp
        N.append(rest)
    return N


def _loese(A, b):
    """Gauß mit Spaltenpivot."""
    n = len(A)
    M = [zeile[:] + [b[i]] for i, zeile in enumerate(A)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        for r in range(n):
            if r != c and M[r][c] != 0.0:
                f = M[r][c] / M[c][c]
                M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return [M[i][n] / M[i][i] for i in range(n)]


def bogen_bspline(kreis, phi0, phi1, spans=24):
    """Kreisbogen [φ0, φ1] als nicht rationaler kubischer B-Spline (kleinste Quadrate, Parameter = Winkel);
    liefert (Kurve, größter Fehler) – 24 Abschnitte über 143° ergeben 2e-4 mm bei r = 735."""
    p = 3
    U = [phi0] * (p + 1) + [phi0 + (phi1 - phi0) * i / spans for i in range(1, spans)] + [phi1] * (p + 1)
    n = spans + p
    ts = [phi0 + (phi1 - phi0) * i / (8 * spans) for i in range(8 * spans + 1)]
    zeilen = []
    for t in ts:
        i = _knotenspanne(U, n, p, t)
        N = _basis(U, p, i, t)
        zeile = [0.0] * n
        for r in range(p + 1):
            zeile[i - p + r] = N[r]
        zeilen.append(zeile)
    A = [[sum(z[i] * z[j] for z in zeilen) for j in range(n)] for i in range(n)]
    punkte = [kurve_punkt(kreis, t) for t in ts]
    pole = [[0.0] * 3 for _ in range(n)]
    for c in range(3):
        x = _loese(A, [sum(z[i] * q[c] for z, q in zip(zeilen, punkte)) for i in range(n)])
        for i in range(n):
            pole[i][c] = x[i]
    k = {"typ": "bspline", "grad": p, "knoten": U, "pole": pole}
    fehler = max(laenge(plus(kurve_punkt(k, t), q, -1)) for t, q in zip(ts, punkte))
    return k, fehler


def zitrone_als_spun(g):
    """Zitronenteil eines Spindeltorus (R < 0, altes Format) als Rotationsfläche seines Kreisbogens; liefert
    (spun, sinn_wechsel). SolidDesigners Flächensinn bezieht sich auf die Normale vom Rohrmittelpunkt weg
    (+n_tube, an 12 Reifenflächen über die Schleifenrichtung geprüft); sinn_wechsel sagt, ob die STEP-Normale
    der Rotationsfläche (T × (A × (P − M)), siehe pkg2step) entgegengesetzt zeigt."""
    kreis, (phi0, phi1) = g["kreis"], g["bogen"]
    profil, _ = bogen_bspline(kreis, phi0, phi1)
    m, a = g["m"], g["a"]
    phim = (phi0 + phi1) / 2
    P = kurve_punkt(kreis, phim)
    y = kreuz(kreis["n"], kreis["x"])
    T = [kreis["r"] * (-math.sin(phim) * kreis["x"][i] + math.cos(phim) * y[i]) for i in range(3)]
    n_spun = kreuz(T, kreuz(a, plus(P, m, -1)))
    n_tube = plus(P, kreis["m"], -1)
    return {"typ": "spun", "profil": profil, "m": m, "a": a, "zitrone": True}, skal(n_spun, n_tube) < 0


def zitronen_umbauen(flaechen):
    """Flächen auf Zitronen-Tori durch Rotationsflächen ersetzen (Flächensinn nach zitrone_als_spun)."""
    for f in flaechen:
        g = f["flaeche"]
        if g["typ"] == "torus" and g["R"] < 0:
            spun, wechsel = zitrone_als_spun(g)
            if wechsel:
                f["umgekehrt"] = not f["umgekehrt"]
            f["flaeche"] = spun


def _geklemmt(k):
    """Ist der Knotenvektor an beiden Enden (Vielfachheit Grad + 1) geklemmt?"""
    p, U = k["grad"], k["knoten"]
    return all(u == U[0] for u in U[:p + 1]) and all(u == U[-1] for u in U[-p - 1:])


def _anhaengen(a, b):
    """Zwei geklemmte B-Splines aneinanderhängen, b beginnt bei a's Endknoten im selben Punkt. Fuge mit
    Vielfachheit Grad (C0). Rationale Pole von b werden auf das Gewicht von a's Endpol skaliert (gleiche Kurve),
    damit der gemeinsame Pol für beide Hälften stimmt."""
    p = a["grad"]
    Pb = b["pole"]
    if a.get("rational"):
        s = a["pole"][-1][-1] / Pb[0][-1]
        Pb = [[x * s for x in q] for q in Pb]
    return {**a, "knoten": a["knoten"][:-1] + b["knoten"][p + 1:], "pole": a["pole"] + Pb[1:]}


def abrollen(kanten):
    """Kanten auf geschlossenen (periodischen) B-Spline-Kurven, deren Bereich außerhalb des gültigen Abschnitts
    [U[p], U[n]] liegt. Geschlossen heißt C(U[p]) = C(U[n]) bis 1e-6; die Kurve setzt sich dann periodisch fort,
    C(t + T) = C(t) mit T = U[n] − U[p].

    - Liegt der Bereich um ganze Perioden versetzt im Abschnitt, wird er nur verschoben (Pilotmodell:
      [d1, d1 + T]).
    - Sonst reicht er über die Naht (Bögen 3002728/3002729, Konverterfälle 7.11.2026: [0,2523; 1,2477] bei [0; 1])
      oder knapp über eine ganze Periode ([2π − 1,3e-6; 4π] bei [0; 2π]). Dann wird die Kurve auf ihren Abschnitt
      geklemmt (Knoteneinfügen, dieselbe Geometrie) und so oft an sich selbst angehängt, wie der Bereich braucht.
      Das gilt für geklemmte geschlossene Kurven (P0 = Pn, altes Format, k2_ssi_curve) wie für ungeklemmt
      periodische (Form 2, P0 ≠ Pn). Die Pole außerhalb des Abschnitts zählen dabei nicht: bei 3002729
      wiederholen sie sich nur bis 0,006 mm, die Kurve auf dem Abschnitt schließt sich bis 2e-14.

    Liefert die Zahl der abgerollten Kanten."""
    n_ab = 0
    for k in kanten.values():
        kv = k["kurve"]
        if kv["typ"] != "bspline":
            continue
        p, U, P = kv["grad"], kv["knoten"], kv["pole"]
        d0, d1 = U[p], U[len(P)]
        t0, t1 = k["bereich"]
        if d0 - 1e-9 <= t0 and t1 <= d1 + 1e-9:
            continue
        periode = d1 - d0
        if periode <= 0 or t1 < t0 or laenge(plus(kurve_punkt(kv, d0), kurve_punkt(kv, d1), -1)) >= 1e-6:
            continue
        n_per = math.floor((t0 - d0) / periode + 1e-9)
        s0, s1 = t0 - n_per * periode, t1 - n_per * periode
        if d0 - 1e-9 <= s0 and s1 <= d1 + 1e-9:
            k["bereich"] = [s0, s1]
            n_ab += 1
            continue
        s0 = max(s0, d0)                                     # 1e-9 unter d0 nach dem Verschieben: auf d0
        kopien = max(2, math.ceil((s1 - d0) / periode - 1e-9))
        basis = kv if _geklemmt(kv) else teilkurve(dict(kv), d0, d1)
        neu = basis
        for j in range(1, kopien):
            neu = _anhaengen(neu, {**basis, "knoten": [u + j * periode for u in basis["knoten"]]})
        k["kurve"] = bspline_pruefen({**kv, "knoten": neu["knoten"], "pole": neu["pole"]}, "abgerollte Kurve")
        k["bereich"] = [s0, s1]
        n_ab += 1
    return n_ab


class BRep:
    """Ein B-Rep-Segment (Wurzel part_struct mit body)."""

    def __init__(self, seg):
        self.seg = seg
        self.O = seg.objekte

    def refs(self, obj, praefix=None):
        """Verweise in Feldreihenfolge, optional nur auf Klassen mit diesem Präfix."""
        out = []
        for t in obj.felder:
            if t.art == "v" and isinstance(t.wert, int) and t.wert in self.O:
                ziel = self.O[t.wert]
                if praefix is None or ziel.klasse == praefix or (praefix.endswith("*") and ziel.klasse.startswith(praefix[:-1])):
                    out.append(ziel)
        return out

    @staticmethod
    def b_nach(obj, ziel):
        """Erstes B-Feld nach dem Verweis auf ziel."""
        gefunden = False
        for t in obj.felder:
            if t.art == "v" and t.wert == ziel.nr:
                gefunden = True
            elif gefunden and t.art == "B":
                return t.wert
        return None

    @staticmethod
    def zahlen_nach(obj, ziel, n):
        gefunden, out = False, []
        for t in obj.felder:
            if t.art == "v" and t.wert == ziel.nr:
                gefunden = True
            elif gefunden and ist_zahl(t):
                out.append(zahl(t))
                if len(out) == n:
                    return out
        return out

    def analytische_form(self, obj, anzahl, referenz=None):
        """Belegte analytische Form, mit Rohindex trotz optionalem 8002-Vorspann.

        8002 ist ein Streammarker vor der ID, kein numerisches Geometriefeld.
        Symboldefinitionen beeinflussen weder Form noch Rohindexzuordnung.
        """
        fs = [(i, t) for i, t in enumerate(obj.felder) if t.art != "s"]
        if fs and fs[0][1].art == "8002":
            fs = fs[1:]
        prefix = ["ID", "D", "D", "D"] + (["B", "v"] if referenz else [])
        art = [t.art for _, t in fs]
        if (len(fs) != len(prefix)+anzahl or art[:len(prefix)] != prefix
                or any(t.wert != 0 for _, t in fs[1:4])
                or (referenz and fs[4][1].wert != 0)
                or not all(ist_zahl(t) for _, t in fs[len(prefix):])):
            raise ValueError(f"{obj.klasse} #{obj.nr}: unbekannte analytische Form")
        nums = []
        for index, token in fs[len(prefix):]:
            value = zahl(token)
            if not math.isfinite(value):
                raise ValueError(f"{obj.klasse} #{obj.nr}: nichtendliches analytisches Feld {index}")
            nums.append(value)
        ref = None
        if referenz:
            ref = self.O.get(fs[5][1].wert)
            if ref is None or ref.klasse != referenz:
                raise ValueError(f"{obj.klasse} #{obj.nr}: analytischer Verweis ist kein {referenz}")
        return nums, ref

    # Geometrie -----------------------------------------------------------
    def geometrie(self, g):
        z = g.zahlen()
        if g.klasse == "k2_point":
            return {"typ": "punkt", "p": z[-3:]}
        if g.klasse == "k2_straight":
            # siebte Zahl: Parametermaßstab, P(t) = p + t·s·d (Creos eigene Teile 1.0; Zukaufteile im Kundenmodell 1000,
            # Bereich dann in Metern: Kante [0, 0.008] = 8 mm, Praxistest 31.10.2026); d bleibt Einheitsrichtung
            return {"typ": "gerade", "p": z[0:3], "d": z[3:6], "s": z[6] if len(z) > 6 else 1.0}
        if g.klasse == "k2_circle":
            return {"typ": "kreis", "n": z[0:3], "m": z[3:6], "r": z[6], "x": z[7:10]}
        if g.klasse == "k2_plane":
            z, achse = self.analytische_form(g, 4, "k2_straight")
            az, _ = self.analytische_form(achse, 7)
            u, v = az[3:6], z[:3]
            if (az[6] == 0 or z[3] == 0 or abs(skal(u, u)-1.) > 1e-12
                    or abs(skal(v, v)-1.) > 1e-12 or abs(skal(u, v)) > 1e-12):
                raise ValueError(f"{g.klasse} #{g.nr}: singuläre analytische Parameterbasis")
            return {"typ": "ebene", "p": az[:3], "u": u, "v": v, "n": kreuz(u, v),
                    "parameter": {"u": list(u), "v": list(v), "su": az[6], "sv": z[3]}}
        if g.klasse == "k2_cylinder":
            # Natürliche Normale (n_Kreis × radial) × a: nach außen, wenn die Normale des Bezugskreises mit der Achse
            # läuft (alle Creo-Proben), sonst nach innen (SD 11, Fahrwerk: 5 von 382 Zylindern) → Flächensinn kippen
            z, kreisobj = self.analytische_form(g, 4, "k2_circle")
            kz, _ = self.analytische_form(kreisobj, 10)
            kreis = self.geometrie(kreisobj)
            if (z[3] == 0 or kz[6] <= 0
                    or any(abs(skal(v, v)-1.) > 1e-12 for v in (z[:3], kreis["n"], kreis["x"]))
                    or abs(skal(kreis["n"], kreis["x"])) > 1e-12
                    or abs(abs(skal(kreis["n"], z[:3]))-1.) > 1e-12):
                raise ValueError(f"{g.klasse} #{g.nr}: singuläre analytische Parameterbasis")
            return {"typ": "zylinder", "m": kreis["m"], "a": z[0:3], "r": kreis["r"], "x": kreis["x"],
                    "parameter": {"n": list(kreis["n"]), "sv": z[3]},
                    "normale_innen": skal(kreis["n"], z[0:3]) < 0}
        if g.klasse == "k2_sphere":
            # Zahlen: 2 Winkel, Mittelpunkt, Achse; Radius und Bezugsrichtung aus dem Kreis
            kreis = self.geometrie(self.refs(g, "k2_circle")[0])
            return {"typ": "kugel", "m": z[2:5], "a": z[5:8], "r": kreis["r"], "x": kreis["x"]}
        if g.klasse == "k2_cone":
            # Zahlen: Vorzeichen (offen), Spitze; Grundkreis gibt Radius und Lage
            kreis = self.geometrie(self.refs(g, "k2_circle")[0])
            achse = plus(kreis["m"], z[1:4], -1)
            return {"typ": "kegel", "m": kreis["m"], "r": kreis["r"], "x": kreis["x"], "spitze": z[1:4],
                    "a": [x / laenge(achse) for x in achse], "winkel": math.atan2(kreis["r"], laenge(achse)),
                    "normale_innen": skal(kreis["n"], achse) < 0}       # wie beim Zylinder
        if g.klasse == "k2_torus":
            # Zahlen: Mittelpunkt, Achse; Erzeugerkreis in einer Meridianebene. Altes Format: davor zwei Winkel
            # (benutzter Bogen des Erzeugerkreises); Bogen jenseits der Achse = Zitronenteil eines Spindeltorus
            kreis = self.geometrie(self.refs(g, "k2_circle")[0])
            t = torus_aus(z[-6:-3], z[-3:], kreis, (z[0], z[1]) if len(z) >= 8 else None)
            if t["R"] > 0:
                # Natürliche Normale vom Rohr weg, wenn die Kreisnormale mit a × radial läuft (alle Creo-Proben), sonst
                # zum Rohr hin – in den Kundenmodellen 115 von 327 bzw. 37 von 212 Tori (Gummidichtung, Praxistest
                # 31.10.2026: pkg2step schrieb dort den Flächensinn verkehrt) → Flächensinn kippen wie beim Zylinder.
                # Zitronenteile (R < 0) haben ihre eigene, an Reifenflächen geprüfte Regel (zitrone_als_spun).
                t["normale_innen"] = skal(kreis["n"], kreuz(t["a"], t["x"])) < 0
            return t
        if g.klasse == "k2_transf_bspline_cv":
            # altes Format: transformierte Kurve (Verweis auf k2_ellipse) mit eingebetteter B-Spline-Näherung
            return bspline(g, 3)
        if g.klasse == "k2_ellipse":
            # Normale, Mittelpunkt, große und kleine Halbachse, Richtung der großen Achse
            return {"typ": "ellipse", "n": z[0:3], "m": z[3:6], "a": z[6], "b": z[7], "x": z[8:11]}
        if g.klasse == "k2_straight2":
            return {"typ": "gerade2d", "p": z[0:2], "d": z[2:4]}
        if g.klasse == "k2_bspline_sf":
            return bspline_flaeche(g)
        if g.klasse == "k2_helical_sf":
            # Schraubfläche (Schneckenspirale im Kundenmodell): trägt ihre rationale B-Spline-Darstellung im Aufbau von
            # k2_bspline_sf; dahinter die Schraubparameter (Achse, Steigung, Windungen), für die Geometrie nicht nötig
            return bspline_flaeche(g)
        if g.klasse == "k2_spun_bspline":
            # Profilkurve um die Achse A durch M gedreht (Creos Import 29): u = Profilparameter, v = Winkel um A
            profil = bspline(self.refs(g, "k2_bspline_cv")[0], 3)
            # Mittelpunkt und Achse sind die letzten sechs Zahlen; davor steht bei Creos Import (29) der Parameter-
            # bereich „B2 0 B1 1 B1“, bei konstruierten Teilen nur „B0“ (Praxistest 30.10.2026)
            return {"typ": "spun", "profil": profil, "m": z[-6:-3], "a": z[-3:]}
        if g.klasse == "k2_parallel_swept_bspline":
            # Bekannte Form (ZIP und SD11): ID,D0³,B0,v Profil,d(3),s.
            # Native Skalierung ×2 lässt d unverändert und setzt s von 1 auf 2:
            # S(u,v) = C(u) + v·s·d. sweep_als_bspline erhält die effektive Richtung.
            f = g.felder
            if (len(f) != 10 or [t.art for t in f[:6]] != ["ID", "D", "D", "D", "B", "v"]
                    or [t.wert for t in f[1:5]] != [0, 0, 0, 0]
                    or not all(ist_zahl(t) and math.isfinite(zahl(t)) for t in f[6:])
                    or f[5].wert not in self.O or self.O[f[5].wert].klasse != "k2_bspline_cv"):
                raise ValueError(f"{g.klasse} #{g.nr}: unbekannte Parallel-Sweep-Form")
            profil = bspline(self.O[f[5].wert], 3)
            return {"typ": "sweep", "profil": profil, "d": [x * z[3] for x in z[:3]]}
        if g.klasse in ("k2_bspline_cv", "k2_bspline_cv2"):
            return bspline(g, 3 if g.klasse == "k2_bspline_cv" else 2)
        if g.klasse == "k2_transf_cone_bspline_cv":
            # Kante auf einem gerollten Kegelblech: vorn eine gewöhnliche 3D-B-Spline-Kurve in Teilkoordinaten, dahinter
            # Kegelachse und Abwicklungslage (Praxistest 30.10.2026, Ansaugkonus); für die Geometrie genügt die Kurve
            return bspline(g, 3)
        if g.klasse == "k2_ssi_curve":
            # Schnittkurve zweier Flächen; für die Geometrie zählt die 3D-Näherung
            k = self.geometrie(self.refs(g, "k2_bspline_cv")[0])
            flaechen = [v for v in self.refs(g, "k2_*") if v.klasse not in ("k2_bspline_cv", "k2_bspline_cv2")]
            k["schnitt"] = [v.klasse for v in flaechen]
            k["flaechen_nr"] = [v.nr for v in flaechen]   # die beiden Flächen, auf denen die Kurve liegt
            return k
        return {"typ": "unbekannt", "klasse": g.klasse}

    # Topologie -----------------------------------------------------------
    def koerperanker(self, body):
        """Verweis, hinter dem Lage und Quader im body stehen: die erste Schale, bei Drahtkörpern ohne Schale
        (Pilotmodell: Schlauchmittellinie als eine B-Spline-Kante) der Draht."""
        return (self.refs(body, "shell") or self.refs(body, "wire"))[0]

    def koerperlage(self, body):
        """Matrix und Translation im body (nach dem shell-Verweis: B, 9 Zahlen, 3 B, 3 Zahlen)."""
        z = self.zahlen_nach(body, self.koerperanker(body), 12)
        return [z[0:3], z[3:6], z[6:9]], z[9:12]

    def toleranz(self, body):
        """Teiltoleranz t im body: nach Lage (12), Skalierung (1) und Quader (6) die 20. Zahl."""
        return self.zahlen_nach(body, self.koerperanker(body), 20)[19]

    def lesen(self):
        body = self.seg.von_klasse("body")[0]
        m, t = self.koerperlage(body)      # altes Format: Geometrie in Körperkoordinaten, Lage hier (pkg2step bäckt sie ein)
        kanten, punkte, flaechen = {}, {}, []
        for v in self.seg.von_klasse("vertex"):
            punkte[v.nr] = self.geometrie(self.refs(v, "k2_point")[0])["p"]
        for e in self.seg.von_klasse("edge"):
            kurven = self.refs(e, "k2_*")
            vs = self.refs(e, "vertex")
            if not kurven:                                          # entartete Polkante (Creo 26/29): Kurve NIL
                kanten[e.nr] = {"kurve": {"typ": "punkt", "p": punkte[vs[0].nr]}, "bereich": [0.0, 0.0],
                                "start": vs[0].nr, "ende": vs[1].nr}
                continue
            kurve = kurven[0]
            kanten[e.nr] = {"kurve": self.geometrie(kurve), "bereich": self.zahlen_nach(e, kurve, 2),
                            "start": vs[0].nr, "ende": vs[1].nr}
        schalen = []
        for sh in self.refs(body, "shell"):
            fl = []
            for f in self.schalenflaechen(sh):
                flaeche = self.refs(f, "k2_*")[0]
                geo = self.geometrie(flaeche)
                if geo["typ"] == "sweep":
                    zf = f.zahlen()
                    geo = sweep_als_bspline(geo, zf[-3], zf[-2])  # Parameterquader des face: v_min, v_max
                schleifen, innen = [], []
                for lp in self.refs(f, "loop"):
                    start = self.refs(lp, "coedge")[0]
                    kette, c = [], start
                    while True:
                        e = self.refs(c, "edge")[0]
                        b = [t.wert for t in c.felder if t.art == "B"]
                        pc = [o for o in self.refs(c, "k2_*") if o.klasse in ("k2_straight2", "k2_bspline_cv2")]
                        uv = self.zahlen_nach(c, flaeche, 4)
                        kette.append({"kante": e.nr, "umgekehrt": b[0] == 1,
                                      "uv": uv if len(uv) == 4 else None,
                                      "pcurve": self.geometrie(pc[0]) if pc else None})
                        c = self.refs(c, "coedge")[0]   # next
                        if c.nr == start.nr or len(kette) > 10000:
                            break
                    schleifen.append(kette)
                    lb = [t.wert for t in lp.felder if t.art == "B"]
                    innen.append(bool(lb) and lb[0] == 1)     # erstes Schleifenflag 1 = Innenschleife, 2 = Rand (06/23/25, alt)
                fl.append({"nr": f.nr, "flaeche": geo, "flaeche_nr": flaeche.nr,
                           "umgekehrt": (self.b_nach(f, flaeche) == 1) != geo.pop("normale_innen", False),
                           "schleifen": schleifen, "innen": innen})
                flaechen.append(fl[-1])
            schalen.append(fl)
        abrollen(kanten)
        zitronen_umbauen(flaechen)
        flaechen_geo = {}
        for k in kanten.values():
            for nr in k["kurve"].get("flaechen_nr", []):
                if nr not in flaechen_geo:
                    geo = self.geometrie(self.O[nr])
                    flaechen_geo[nr] = zitrone_als_spun(geo)[0] if geo["typ"] == "torus" and geo["R"] < 0 else geo
        self.isolinien_einsetzen(flaechen, kanten, punkte)
        return {"schalen": schalen, "kanten": kanten, "punkte": punkte, "toleranz": self.toleranz(body),
                "flaechen_geo": flaechen_geo, "lage": {"matrix": m, "verschiebung": t},
                "koerperart": koerperart(self.seg, body)}

    def schalenflaechen(self, sh):
        """Flächen einer Schale: direkt verwiesen (Creo) oder über subshell-Objekte (altes Format: die shell
        verweist nur auf eine Fläche und eine subshell, jede Fläche auf ihre subshell, die auf die shell)."""
        out, gesehen = [], set()
        for f in self.refs(sh, "face"):
            gesehen.add(f.nr)
            out.append(f)
        for f in self.seg.von_klasse("face"):
            sub = self.refs(f, "subshell")
            if sub and f.nr not in gesehen and any(s.nr == sh.nr for s in self.refs(sub[0], "shell")):
                gesehen.add(f.nr)
                out.append(f)
        return out

    @staticmethod
    def isolinien_einsetzen(flaechen, kanten, punkte):
        """Schnittkurven, deren Parameterkurve auf einer B-Spline-Fläche eine Isolinie (u oder v fest) ist,
        durch die exakte Isolinie ersetzen – die gespeicherte 3D-Näherung liegt sonst einige µm neben der Fläche."""
        for f in flaechen:
            if f["flaeche"]["typ"] != "bspline_flaeche":
                continue
            for lp in f["schleifen"]:
                for ce in lp:
                    k = kanten[ce["kante"]]
                    pc = ce["pcurve"]
                    if "schnitt" not in k["kurve"] or not ce["uv"] or not pc or pc["typ"] != "gerade2d":
                        continue
                    u0, v0, u1, v1 = ce["uv"]
                    if abs(u1 - u0) < 1e-9:
                        kurve, lo, hi = isolinie(f["flaeche"], "u", u0), min(v0, v1), max(v0, v1)
                    elif abs(v1 - v0) < 1e-9:
                        kurve, lo, hi = isolinie(f["flaeche"], "v", v0), min(u0, u1), max(u0, u1)
                    else:
                        continue
                    lo, hi = max(lo, kurve["knoten"][0]), min(hi, kurve["knoten"][-1])
                    kurve = teilkurve(kurve, lo, hi)
                    ends = pol_punkte(kurve)
                    if laenge(plus(ends[0], punkte[k["start"]], -1)) > laenge(plus(ends[-1], punkte[k["start"]], -1)):
                        kurve = umkehren(kurve)
                    kurve["flaechen_nr"] = k["kurve"].get("flaechen_nr", [])
                    kurve["isolinie"] = True
                    k["kurve"], k["bereich"] = kurve, [kurve["knoten"][0], kurve["knoten"][-1]]


def kurve_punkt(k, t):
    if k["typ"] == "punkt":
        return list(k["p"])
    if k["typ"] == "gerade":
        return plus(k["p"], k["d"], t * k.get("s", 1.0))
    if k["typ"] == "kreis":
        y = kreuz(k["n"], k["x"])
        return [m + k["r"] * (math.cos(t) * x + math.sin(t) * yy) for m, x, yy in zip(k["m"], k["x"], y)]
    if k["typ"] == "ellipse":
        y = kreuz(k["n"], k["x"])
        return [m + k["a"] * math.cos(t) * x + k["b"] * math.sin(t) * yy for m, x, yy in zip(k["m"], k["x"], y)]
    if k["typ"] == "bspline":
        q = de_boor(k, t)
        return euklidisch(q) if k.get("rational") else q
    return None


def abstand_kurve(k, p, t0, t1, grob=64):
    """Abstand eines Punkts zur Kurve auf [t0, t1]: grobe Abtastung, dann Dreiteilungssuche um die beste Stelle."""
    ts = [t0 + (t1 - t0) * i / grob for i in range(grob + 1)]
    d = [laenge(plus(kurve_punkt(k, t), p, -1)) for t in ts]
    i = min(range(len(ts)), key=d.__getitem__)
    lo, hi = ts[max(i - 1, 0)], ts[min(i + 1, grob)]
    for _ in range(60):
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if laenge(plus(kurve_punkt(k, m1), p, -1)) < laenge(plus(kurve_punkt(k, m2), p, -1)):
            hi = m2
        else:
            lo = m1
    return min(d[i], laenge(plus(kurve_punkt(k, (lo + hi) / 2), p, -1)))


def abstand_flaeche(f, p):
    if f["typ"] == "ebene":
        n = f["n"]
        return abs(skal(plus(p, f["p"], -1), n)) / laenge(n)
    if f["typ"] == "zylinder":
        a = [x / laenge(f["a"]) for x in f["a"]]
        v = plus(p, f["m"], -1)
        radial = plus(v, a, -skal(v, a))
        return abs(laenge(radial) - f["r"])
    if f["typ"] == "kugel":
        return abs(laenge(plus(p, f["m"], -1)) - f["r"])
    if f["typ"] == "kegel":
        v = plus(p, f["spitze"], -1)
        h = skal(v, f["a"])
        radial = laenge(plus(v, f["a"], -h))
        return abs(radial - h * math.tan(f["winkel"])) * math.cos(f["winkel"])
    if f["typ"] == "torus":
        a = [x / laenge(f["a"]) for x in f["a"]]
        v = plus(p, f["m"], -1)
        h = skal(v, a)
        radial = laenge(plus(v, a, -h))
        return abs(math.hypot(radial - f["R"], h) - f["r"])
    if f["typ"] == "spun":                                          # Abstand in der Meridianebene zur Profilkurve
        a = [x / laenge(f["a"]) for x in f["a"]]
        d = plus(p, f["m"], -1)
        h = skal(d, a)
        rho = laenge(plus(d, a, -h))
        pole = []
        for q in pol_punkte(f["profil"]):
            dq = plus(q, f["m"], -1)
            hq = skal(dq, a)
            pole.append([laenge(plus(dq, a, -hq)), hq, 0.0])
        k2 = {"typ": "bspline", "grad": f["profil"]["grad"], "knoten": f["profil"]["knoten"], "pole": pole}
        if f["profil"].get("rational"):
            k2["pole"] = [[x * q[-1] for x in pp] + [q[-1]] for pp, q in zip(pole, f["profil"]["pole"])]
            k2["rational"] = True
        return abstand_kurve(k2, [rho, h, 0.0], k2["knoten"][0], k2["knoten"][-1])
    return None


def schleife_polygon(b, kette, n_kreis=32):
    """Punktfolge einer Schleife in Laufrichtung (Kreise abgetastet)."""
    pts = []
    for ce in kette:
        k = b["kanten"][ce["kante"]]
        t0, t1 = k["bereich"]
        n = 1 if k["kurve"]["typ"] == "gerade" else n_kreis
        ts = [t0 + (t1 - t0) * i / n for i in range(n)]
        seg = [kurve_punkt(k["kurve"], t) for t in ts] + [kurve_punkt(k["kurve"], t1)]
        if any(p is None for p in seg):
            raise ValueError(f"Kante #{ce['kante']}: Kurventyp {k['kurve']['typ']} nicht auswertbar")
        if ce["umgekehrt"]:
            seg.reverse()
        pts += seg[:-1]
    return pts


def orientierung(b):
    """Je ebener Fläche: vorzeichenbehaftete Fläche jeder Schleife bezogen auf die Außennormale."""
    out = []
    for f in (f for s in b["schalen"] for f in s):
        if f["flaeche"]["typ"] != "ebene":
            continue
        n = f["flaeche"]["n"]
        if f["umgekehrt"]:
            n = [-x for x in n]
        flaechen = []
        try:
            for kette in f["schleifen"]:
                p = schleife_polygon(b, kette)
                newell = [0.0, 0.0, 0.0]
                for a, c in zip(p, p[1:] + p[:1]):
                    newell = plus(newell, kreuz(a, c))
                flaechen.append(skal(newell, n) / 2 / laenge(n))
        except ValueError:
            continue   # unbekannte Kurve, wird bei den Kanten gemeldet
        out.append((f["nr"], flaechen))
    return out


def geschlossen(f):
    """Ist die B-Spline-Fläche in u bzw. v geschlossen (Rohr, gerolltes Blech)? Je Richtung (lo, hi) oder None."""
    if "_geschlossen" not in f:
        lo_u, hi_u = f["knoten_u"][f["grad_u"]], f["knoten_u"][len(f["pole"])]
        lo_v, hi_v = f["knoten_v"][f["grad_v"]], f["knoten_v"][len(f["pole"][0])]
        mu, mv = (lo_u + hi_u) / 2, (lo_v + hi_v) / 2
        zu_u = all(laenge(plus(flaeche_punkt(f, lo_u, v), flaeche_punkt(f, hi_u, v), -1)) < 1e-6
                   for v in (lo_v, mv, hi_v))
        zu_v = all(laenge(plus(flaeche_punkt(f, u, lo_v), flaeche_punkt(f, u, hi_v), -1)) < 1e-6
                   for u in (lo_u, mu, hi_u))
        f["_geschlossen"] = ((lo_u, hi_u) if zu_u and hi_u > lo_u else None, (lo_v, hi_v) if zu_v and hi_v > lo_v else None)
    return f["_geschlossen"]


def flag2_fehlt(f):
    """Richtungen ("u", "v"), in denen eine k2_bspline_sf geometrisch geschlossen ist, aber kein Flag 2 trägt. Creo
    misst eine solche Fläche nicht mit und meldet nichts (Lisp-Läufe 10 und 11: s10_24, s11_03); Creos eigene geschlossene
    Flächen tragen es ausnahmslos (Kundenmodelle: 20 von 20)."""
    if f.get("typ") != "bspline_flaeche" or "flag_zu" not in f:
        return []
    zu = geschlossen(f)
    return [r for r, z, flag in zip("uv", zu, f["flag_zu"]) if z is not None and not flag]


def geschlossen_ohne_flag(pfad):
    """Alle geschlossenen k2_bspline_sf ohne Flag 2 eines Pakets als (Teil-ID, Objektnummer, Richtung); liest die
    Teilsegmente einzeln (auch große Kundenmodelle)."""
    import pkgleser as pl
    leser = pl.Leser(mit_punkten=False)
    modell = leser.lesen(pfad, sparsam=True)
    out = []
    for tid in list(modell["teile"]):
        for seg in leser.index.get(tid) or []:
            if seg.klasse != "part_struct":
                continue
            for o in seg.von_klasse("k2_bspline_sf"):
                try:
                    f = bspline_flaeche(o)
                except (ValueError, StopIteration):
                    continue
                out += [(tid, o.nr, r) for r in flag2_fehlt(f)]
    return out


def abstand_bspline_flaeche(f, p, u, v, schritte=12):
    """Abstand eines Punkts zu einer B-Spline-Fläche: Gauß-Newton ab (u, v), numerische Ableitungen; Parameter im
    gültigen Bereich, in geschlossener Richtung modulo Periode zurückgefaltet statt geklemmt."""
    pu, pv = f["grad_u"], f["grad_v"]
    lo_u, hi_u = f["knoten_u"][pu], f["knoten_u"][len(f["pole"])]
    lo_v, hi_v = f["knoten_v"][pv], f["knoten_v"][len(f["pole"][0])]
    zu = geschlossen(f)

    def falten(x, lo, hi, z):
        return lo + (x - lo) % (hi - lo) if z else min(max(x, lo), hi)
    hu, hv = 1e-6 * max(1.0, hi_u - lo_u), 1e-6 * max(1.0, hi_v - lo_v)
    for _ in range(schritte):
        u, v = falten(u, lo_u, hi_u, zu[0]), falten(v, lo_v, hi_v, zu[1])
        S = flaeche_punkt(f, u, v)
        Su = [(a - c) / (2 * hu) for a, c in zip(flaeche_punkt(f, min(u + hu, hi_u), v), flaeche_punkt(f, max(u - hu, lo_u), v))]
        Sv = [(a - c) / (2 * hv) for a, c in zip(flaeche_punkt(f, u, min(v + hv, hi_v)), flaeche_punkt(f, u, max(v - hv, lo_v)))]
        r = plus(p, S, -1)
        a11, a12, a22 = skal(Su, Su), skal(Su, Sv), skal(Sv, Sv)
        b1, b2 = skal(Su, r), skal(Sv, r)
        det = a11 * a22 - a12 * a12
        if abs(det) < 1e-30:
            break
        du, dv = (a22 * b1 - a12 * b2) / det, (a11 * b2 - a12 * b1) / det
        u, v = u + du, v + dv
        if abs(du) < 1e-12 * max(1.0, hi_u - lo_u) and abs(dv) < 1e-12 * max(1.0, hi_v - lo_v):
            break
    u, v = falten(u, lo_u, hi_u, zu[0]), falten(v, lo_v, hi_v, zu[1])
    return laenge(plus(flaeche_punkt(f, u, v), p, -1))


def abstand_bspline_flaeche_global(f, p):
    """Abstand eines Punkts zur ganzen B-Spline-Fläche: bester Rasterpunkt als Startwert (je Richtung doppelt so
    viele Rasterpunkte wie Pole – die Schraubfläche hat 162 Pole in v), dann Gauß-Newton."""
    nu, nv = 2 * len(f["pole"]), 2 * len(f["pole"][0])
    lo_u, hi_u = f["knoten_u"][f["grad_u"]], f["knoten_u"][len(f["pole"])]
    lo_v, hi_v = f["knoten_v"][f["grad_v"]], f["knoten_v"][len(f["pole"][0])]
    if "_raster" not in f:
        f["_raster"] = [(lo_u + (hi_u - lo_u) * i / nu, lo_v + (hi_v - lo_v) * j / nv,
                         flaeche_punkt(f, lo_u + (hi_u - lo_u) * i / nu, lo_v + (hi_v - lo_v) * j / nv))
                        for i in range(nu + 1) for j in range(nv + 1)]
    u, v, _ = min(f["_raster"], key=lambda e: laenge(plus(e[2], p, -1)))
    return abstand_bspline_flaeche(f, p, u, v, schritte=20)


def bspline_flaechen_pruefen(b, flaechen, tol, n, befund):
    """B-Spline-Flächen: Eckpunkte über die uv-Enden der coedges, Kanten entlang der Parameterkurve."""
    for f in flaechen:
        if f["flaeche"]["typ"] != "bspline_flaeche":
            continue
        for lp in f["schleifen"]:
            for ce in lp:
                k = b["kanten"][ce["kante"]]
                if not ce.get("uv"):
                    befund.append(f"Fläche #{f['nr']}: Kante #{ce['kante']} ohne uv-Bereich")
                    continue
                u0, v0, u1, v1 = ce["uv"]   # in Kantenrichtung, unabhängig von umgekehrt
                for (u, v), p in (((u0, v0), b["punkte"][k["start"]]), ((u1, v1), b["punkte"][k["ende"]])):
                    d = laenge(plus(flaeche_punkt(f["flaeche"], u, v), p, -1))
                    if d > tol:
                        befund.append(f"Fläche #{f['nr']}: Eckpunkt {p} liegt {d:g} neben S({u:g}, {v:g})")
                    else:
                        n["punkte_auf_flaeche"] += 1
                if k["kurve"]["typ"] == "unbekannt":
                    continue
                t0, t1 = k["bereich"]
                pc = ce.get("pcurve")
                if pc and pc["typ"] == "bspline":
                    U = pc["knoten"]
                    uvs = [kurve_punkt(pc, U[0] + (U[-1] - U[0]) * i / 16) for i in range(17)]
                    d = max(abstand_kurve(k["kurve"], flaeche_punkt(f["flaeche"], u, v), t0, t1) for u, v in uvs)
                else:
                    # ohne Parameterkurve ist der uv-Weg unbekannt (nicht unbedingt gerade): Kurvenpunkte auf die
                    # Fläche projizieren, Startwert auf der Geraden zwischen den uv-Enden (Pilotmodell 7f:9b46ee).
                    # In geschlossener Richtung kann die Kante auch andersherum über die Naht laufen (Rohre in
                    # Kundenmodell): beide Wege versuchen, der bessere gilt
                    fl = f["flaeche"]
                    wege = [(u1, v1)]
                    for richtung, zu in enumerate(geschlossen(fl)):
                        if zu:
                            T = zu[1] - zu[0]
                            for w in list(wege):
                                ende = list(w)
                                ende[richtung] += -T if ende[richtung] > (u0, v0)[richtung] else T
                                wege.append(tuple(ende))
                    d = min(max(abstand_bspline_flaeche(fl, kurve_punkt(k["kurve"], t0 + (t1 - t0) * i / 16),
                                                        u0 + (ue - u0) * i / 16, v0 + (ve - v0) * i / 16) for i in range(17))
                            for ue, ve in wege)
                    if d > tol:
                        # Rückfall: Startwert je Punkt aus einem groben Raster über die ganze Fläche (Kanten über
                        # mehrere Windungen einer Schraubfläche, Kundenmodell)
                        d = max(abstand_bspline_flaeche_global(fl, kurve_punkt(k["kurve"], t0 + (t1 - t0) * i / 16))
                                for i in range(17))
                n["abweichung_max"] = max(n.get("abweichung_max", 0.0), d)
                if d > tol:
                    befund.append(f"Kante #{ce['kante']} liegt bis {d:g} neben B-Spline-Fläche #{f['nr']} (Toleranz {tol:g})")
                else:
                    n["kanten_auf_flaeche"] += 1


def analytische_parameter_pruefen(b, flaechen, tol, n, befund):
    """Native analytische UV-Endpunkte gegen unabhängige räumliche Vertices."""
    n["analytische_uv_punkte"] = 0
    for face in flaechen:
        geo = face["flaeche"]
        if geo["typ"] not in ("ebene", "zylinder") or "parameter" not in geo:
            continue
        for loop in face["schleifen"]:
            for ce in loop:
                uv = ce.get("uv")
                if uv is None or len(uv) != 4 or not all(math.isfinite(x) for x in uv):
                    befund.append(f"Fläche #{face['nr']}: Kante #{ce['kante']} ohne endliche native UV-Enden")
                    continue
                edge = b["kanten"][ce["kante"]]
                for par, vertex in ((uv[:2], edge["start"]), (uv[2:], edge["ende"])):
                    d = laenge(plus(flaeche_punkt(geo, *par), b["punkte"][vertex], -1))
                    n["abweichung_max"] = max(n["abweichung_max"], d)
                    if not math.isfinite(d) or d > tol:
                        befund.append(f"Fläche #{face['nr']}: native UV-Enden passen nicht zu Vertex #{vertex} (Abstand {d:g})")
                    else:
                        n["analytische_uv_punkte"] += 1


def pruefen(b, nachsicht=0.0):
    """Selbstprüfung: Kurvenenden = Eckpunkte, Eckpunkte auf ihren Flächen, Euler-Poincaré, Unbekanntes.
    Toleranz: die Teiltoleranz, mindestens `nachsicht` – Creos Schnittkurven liegen in Kundenmodellen bis etwa 12 µm
    neben ihren Eckpunkten, auch bei Teiltoleranz 1e-6 (Praxistest 31.10.2026). n["abweichung_max"] ist die größte
    gemessene Abweichung dieser Prüfungen."""
    befund, n = [], {"kantenenden": 0, "punkte_auf_flaeche": 0, "abweichung_max": 0.0}
    tol = max(TOL * 100, b.get("toleranz", 0.0), nachsicht)
    for nr, k in b["kanten"].items():
        if k["kurve"]["typ"] == "unbekannt":
            befund.append(f"Kante #{nr}: Kurve {k['kurve']['klasse']} unbekannt")
            continue
        for t, v in zip(k["bereich"], (k["start"], k["ende"])):
            p = kurve_punkt(k["kurve"], t)
            n["abweichung_max"] = max(n["abweichung_max"], laenge(plus(p, b["punkte"][v], -1)))
            if laenge(plus(p, b["punkte"][v], -1)) > tol:
                befund.append(f"Kante #{nr}: Kurve bei t={t:g} ist {p}, Eckpunkt {b['punkte'][v]}")
            else:
                n["kantenenden"] += 1
    flaechen = [f for s in b["schalen"] for f in s]
    for f in flaechen:
        if f["flaeche"]["typ"] == "unbekannt":
            befund.append(f"Fläche #{f['nr']}: {f['flaeche']['klasse']} unbekannt")
            continue
        for lp in f["schleifen"]:
            for ce in lp:
                k = b["kanten"][ce["kante"]]
                for v in (k["start"], k["ende"]):
                    d = abstand_flaeche(f["flaeche"], b["punkte"][v])
                    if d is not None:
                        n["abweichung_max"] = max(n["abweichung_max"], d)
                    if d is not None and d > tol:
                        befund.append(f"Fläche #{f['nr']}: Eckpunkt {b['punkte'][v]} Abstand {d:g}")
                    else:
                        n["punkte_auf_flaeche"] += 1
    # jede Kante liegt auf allen angrenzenden Flächen, innerhalb der Teiltoleranz
    n["kanten_auf_flaeche"] = 0
    for f in flaechen:
        if f["flaeche"]["typ"] in ("unbekannt", "bspline_flaeche"):
            continue
        for lp in f["schleifen"]:
            for ce in lp:
                k = b["kanten"][ce["kante"]]
                if k["kurve"]["typ"] == "unbekannt":
                    continue
                t0, t1 = k["bereich"]
                d = max(abstand_flaeche(f["flaeche"], kurve_punkt(k["kurve"], t0 + (t1 - t0) * i / 16)) for i in range(17))
                n["abweichung_max"] = max(n["abweichung_max"], d)
                if d > tol:
                    befund.append(f"Kante #{ce['kante']} liegt bis {d:g} neben Fläche #{f['nr']} (Toleranz {tol:g})")
                else:
                    n["kanten_auf_flaeche"] += 1
    analytische_parameter_pruefen(b, flaechen, tol, n, befund)
    bspline_flaechen_pruefen(b, flaechen, tol, n, befund)
    for f in flaechen:
        for r in flag2_fehlt(f["flaeche"]):
            befund.append(f"Fläche #{f['nr']}: in {r} geschlossen, aber ohne Flag 2 – Creo lässt sie beim Volumen weg")
    if not flaechen:                                                # Drahtkörper: keine Flächentopologie
        n["draht"] = True
        return n, befund
    nutzung = {}
    for f in flaechen:
        for lp in f["schleifen"]:
            for ce in lp:
                nutzung.setdefault(ce["kante"], []).append(ce["umgekehrt"])
    for nr in b["kanten"]:
        if b["kanten"][nr]["kurve"]["typ"] == "punkt":
            continue                                                # Polkante: eine Coedge, Partner sie selbst
        if sorted(nutzung.get(nr, [])) != [False, True]:
            befund.append(f"Kante #{nr}: Nutzung {nutzung.get(nr)} statt einmal je Richtung")
    n["kanten_mannigfaltig"] = sum(sorted(u) == [False, True] for u in nutzung.values())
    n["entartet"] = 0
    for nr, fl in orientierung(b):
        if fl and max(abs(x) for x in fl) <= tol * tol:
            n["entartet"] += 1                                      # Fläche ohne Inhalt (Creo-Daten): keine Richtung
            continue
        if fl and (max(fl) <= 0 or sum(x > 0 for x in fl) != 1):
            befund.append(f"Fläche #{nr}: Schleifenorientierung {fl}")
    # Polkanten (Creo 26/29: entartete Kante mit einem Eckpunkt, eigene Schleife) zählen nicht
    pol = {nr for nr, k in b["kanten"].items() if k["kurve"]["typ"] == "punkt"}
    pol_v = {b["kanten"][nr]["start"] for nr in pol}
    V, E, F = len(b["punkte"]) - len(pol_v), len(b["kanten"]) - len(pol), len(flaechen)
    L = sum(1 for f in flaechen for lp in f["schleifen"] if not any(ce["kante"] in pol for ce in lp))
    S = len(b["schalen"])
    # Euler-Poincaré: V − E + F − (L − F) = 2 (S − G)  →  Geschlecht G
    rest = V - E + F - (L - F)
    g = S - rest / 2
    n["euler"] = {"V": V, "E": E, "F": F, "L": L, "S": S, "geschlecht": g}
    if g != int(g) or g < 0:
        befund.append(f"Euler-Poincaré verletzt: {n['euler']}")
    return n, befund


def _einheit(v):
    lv = laenge(v)
    return [c / lv for c in v] if lv > 1e-300 else v


def normale_aussen(f, p):
    """Außennormale der Fläche f im Punkt p (natürliche Normale wie in pkg2step geschrieben, bei umgekehrter Fläche
    negiert); None für Typen ohne geschlossene Formel (B-Spline-Fläche, Zitronen-Torus)."""
    g = f["flaeche"]
    t = g["typ"]
    n = None
    if t == "ebene":
        n = _einheit(g["n"])
    elif t in ("zylinder", "kegel", "kugel", "torus", "spun"):
        a = _einheit(g["a"])
        d = plus(p, g["spitze"] if t == "kegel" else g["m"], -1)
        h = skal(d, a)
        rad = plus(d, a, -h)
        er = _einheit(rad)
        if t == "zylinder":
            n = er
        elif t == "kegel":
            n = _einheit(plus(er, a, -math.tan(g["winkel"])))
        elif t == "kugel":
            n = _einheit(d)
        elif t == "torus":
            n = _einheit(plus(p, plus(g["m"], er, g["R"]), -1)) if g["R"] > 0 else None
        else:
            # Rotationsfläche (STEP-Achse −a): N ∝ ρ'·a − h'·e_r am Profilpunkt mit gleichem (ρ, h)
            k, rho = g["profil"], laenge(rad)
            u0, u1 = k["knoten"][0], k["knoten"][-1]

            def rh(u):
                q = plus(kurve_punkt(k, u), g["m"], -1)
                hq = skal(q, a)
                return laenge(plus(q, a, -hq)), hq
            us = [u0 + (u1 - u0) * i / 200 for i in range(201)]
            bu = min(us, key=lambda u: (rh(u)[0] - rho) ** 2 + (rh(u)[1] - h) ** 2)
            e = (u1 - u0) * 1e-5
            (r1, h1), (r0, h0) = rh(min(bu + e, u1)), rh(max(bu - e, u0))
            n = _einheit(plus([(r1 - r0) * c for c in a], er, -(h1 - h0)))
    if n is None:
        return None
    return [-c for c in n] if f["umgekehrt"] else n


def flaechensinn_tangential(b, schwelle=0.9999):
    """Flächensinn an tangentialen Kanten: liegen zwei Flächen G1-stetig aneinander, zeigen ihre Außennormalen
    gleich. Gegenläufige Normalen bedeuten einen verkehrten Flächensinn – oder eine echte Schneide (Flächen
    berühren sich tangential von derselben Seite, z. B. Rundstab auf Blech im Fahrwerk-Stützfuß). Liefert Zähler
    und die gegenläufigen Kanten (Kante, Fläche 1, Fläche 2)."""
    nachbarn = {}
    for s in b["schalen"]:
        for f in s:
            for kette in f["schleifen"]:
                for ce in kette:
                    nachbarn.setdefault(ce["kante"], []).append(f)
    zaehler = {"tangential passend": 0, "tangential gegenläufig": 0, "nicht tangential": 0, "ohne Normale": 0}
    gegen = []
    for nr, fl in nachbarn.items():
        k = b["kanten"][nr]
        if len(fl) != 2 or fl[0] is fl[1] or k["kurve"]["typ"] == "punkt":
            continue
        t0, t1 = k["bereich"]
        p = kurve_punkt(k["kurve"], (t0 + t1) / 2)
        ns = [normale_aussen(f, p) for f in fl]
        if None in ns:
            zaehler["ohne Normale"] += 1
            continue
        c = skal(ns[0], ns[1])
        if abs(c) < schwelle:
            zaehler["nicht tangential"] += 1
        elif c > 0:
            zaehler["tangential passend"] += 1
        else:
            zaehler["tangential gegenläufig"] += 1
            gegen.append((nr, fl[0]["nr"], fl[1]["nr"]))
    return zaehler, gegen


def polyeder_volumen(b):
    """Orientiertes Volumen eines Teils, dessen Flächen alle eben und dessen Kanten alle gerade sind (Satz von Gauß
    über die Schleifen: je Schleife Fächer aus Dreiecken, Summe p0 · (pi × pi+1) / 6); sonst None. Hohlräume mit
    richtiger Orientierung ziehen ab; ein falsch orientierter Hohlraum zählt dazu (Prüflauf 30.10.2026, Probe 30)."""
    flaechen = [f for s in b["schalen"] for f in s]
    if not flaechen or any(f["flaeche"]["typ"] != "ebene" for f in flaechen):
        return None
    if any(k["kurve"]["typ"] != "gerade" for k in b["kanten"].values()):
        return None
    v = 0.0
    for f in flaechen:
        for lp in f["schleifen"]:
            ecken = []
            for ce in lp:
                k = b["kanten"][ce["kante"]]
                ecken.append(b["punkte"][k["ende"] if ce["umgekehrt"] else k["start"]])
            p0 = ecken[0]
            for a, c in zip(ecken[1:], ecken[2:]):
                v += skal(p0, kreuz(a, c)) / 6.0
    return v


def _zeile(p, m, t=None):
    """p·M (+ t): Zeilenvektor wie bei den Lagen (pkgleser.anwenden)."""
    q = [sum(p[k] * m[k][c] for k in range(3)) for c in range(3)]
    return [x + y for x, y in zip(q, t)] if t is not None else q


def _norm(v):
    lv = laenge(v)
    return [x / lv for x in v]


def _pole_abbilden(pole, abb, rational):
    """Pole (homogen bei rationalen: w·x, w) mit der Punktabbildung abb."""
    if not rational:
        return [abb(q) for q in pole]
    return [[x * q[-1] for x in abb(euklidisch(q))] + [q[-1]] for q in pole]


AFFIN_TYPEN = {"ebene", "gerade", "bspline", "bspline_flaeche", "punkt", "gerade2d", "unbekannt"}


def geo_abbilden(g, m, t, s=None):
    """Kurve oder Fläche mit p' = p·m + t abbilden. s: Maßstab, wenn m = s·R eine Ähnlichkeit mit det > 0 ist; dann
    gehen alle Typen (Richtungen über R, Längen ·s). Sonst nur Ebenen, Geraden und B-Splines (exakt affin)."""
    typ = g["typ"]
    punkt = lambda p: _zeile(p, m, t)  # noqa: E731
    richtung = lambda d: _zeile(d, m)  # noqa: E731
    if s is None and typ not in AFFIN_TYPEN:
        raise ValueError(f"{typ}: nur mit Drehen, Verschieben und gleichmäßigem Skalieren abbildbar")
    g = dict(g)
    if typ == "punkt":
        g["p"] = punkt(g["p"])
    elif typ == "gerade":                                   # P = p + τ·s·d: Bereich bleibt, Maßstab wächst mit |d·m|
        d = richtung(g["d"])
        g.update(p=punkt(g["p"]), d=_norm(d), s=g.get("s", 1.0) * laenge(d))
    elif typ in ("kreis", "ellipse"):
        x, y = richtung(g["x"]), richtung(kreuz(g["n"], g["x"]))
        g.update(m=punkt(g["m"]), x=_norm(x), n=_norm(kreuz(x, y)))
        for schl in ("r", "a", "b"):
            if schl in g:
                g[schl] = g[schl] * s
    elif typ == "bspline":
        g["pole"] = _pole_abbilden(g["pole"], punkt, g.get("rational"))
    elif typ == "bspline_flaeche":
        g["pole"] = [_pole_abbilden(zeile, punkt, g.get("rational")) for zeile in g["pole"]]
        g.pop("_geschlossen", None)
    elif typ == "ebene":                                    # Normale u × v behält ihre Seite (det > 0)
        # Native Parameter können nach einer Scherung schief sein; die
        # orthonormalisierte Trägerbasis dient nur STEP/Geometrieprojektion.
        par = g.get("parameter", {"u": g["u"], "v": g["v"], "su": 1., "sv": 1.})
        pu, pv = richtung(par["u"]), richtung(par["v"])
        g["parameter"] = {"u": _norm(pu), "v": _norm(pv),
                          "su": par["su"]*laenge(pu), "sv": par["sv"]*laenge(pv)}
        u, v = richtung(g["u"]), richtung(g["v"])
        u = _norm(u)
        v = _norm(plus(v, u, -skal(v, u)))
        g.update(p=punkt(g["p"]), u=u, v=v, n=kreuz(u, v))
    elif typ in ("zylinder", "kugel", "kegel", "torus"):
        if typ == "zylinder":
            par = g.get("parameter", {"n": g["a"], "sv": 1.})
            g["parameter"] = {"n": _norm(richtung(par["n"])), "sv": par["sv"]*s}
        for schl in ("m", "spitze"):
            if schl in g:
                g[schl] = punkt(g[schl])
        for schl in ("a", "x"):
            g[schl] = _norm(richtung(g[schl]))
        for schl in ("r", "R"):
            if schl in g:
                g[schl] = g[schl] * s
    elif typ == "spun":
        g.update(profil=geo_abbilden(g["profil"], m, t, s), m=punkt(g["m"]), a=_norm(richtung(g["a"])))
    elif typ not in ("gerade2d", "unbekannt"):
        raise ValueError(f"{typ}: Abbildung nicht umgesetzt")
    return g


def brep_abbilden(b, matrix, verschiebung):
    """Geometrie eines B-Rep (Aufbau wie BRep.lesen) mit p' = p·M + t abbilden, nach einer etwaigen Körperlage.
    Ähnlichkeit (M = s·R, R orthogonal): jede Fläche bleibt ihr Typ, Radien ·s; eine Spiegelung (det R < 0) geht als
    Körperlage an pkg2step, das sie samt Flächensinn einbäckt. Sonst (Strecken, Scheren): nur Ebenen, Geraden und
    B-Splines, det M > 0. Topologie, Kantenbereiche und Flächensinn bleiben; liefert ein neues B-Rep."""
    import copy
    m = [list(map(float, z)) for z in matrix]
    t = list(map(float, verschiebung))
    kl = b.get("lage") or {"matrix": [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]], "verschiebung": [0.0, 0.0, 0.0]}
    mmt = [[sum(m[r][k] * m[c][k] for k in range(3)) for c in range(3)] for r in range(3)]
    s2 = sum(mmt[i][i] for i in range(3)) / 3
    aehnlich = s2 > 0 and all(abs(mmt[r][c] - (s2 if r == c else 0.0)) <= 1e-9 * s2 for r in range(3) for c in range(3))
    det_m = (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
             + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
    neu = copy.deepcopy(b)
    if aehnlich:
        sk = math.sqrt(s2)
        rot = [[x / sk for x in z] for z in m]
        einheit = [[sk, 0.0, 0.0], [0.0, sk, 0.0], [0.0, 0.0, sk]]
        abb = (einheit, [0.0, 0.0, 0.0], sk)                         # Geometrie nur skalieren (um den Ursprung)
        km, kt = kl["matrix"], kl["verschiebung"]                    # Lage: Körperlage, dann R, dann t
        neu["lage"] = {"matrix": [[sum(km[r][k] * rot[k][c] for k in range(3)) for c in range(3)] for r in range(3)],
                       "verschiebung": [x + y for x, y in zip(_zeile([sk * v for v in kt], rot), t)]}
    else:
        if det_m <= 0:
            raise ValueError("allgemeine affine Abbildung mit det ≤ 0 nicht umgesetzt")
        km, kt = kl["matrix"], kl["verschiebung"]                    # Körperlage einbacken
        gesamt = [[sum(km[r][k] * m[k][c] for k in range(3)) for c in range(3)] for r in range(3)]
        abb = (gesamt, _zeile(kt, m, t), None)
        neu["lage"] = {"matrix": [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]], "verschiebung": [0.0, 0.0, 0.0]}
        if det_m < 0:
            raise ValueError("Spiegelung nur als Ähnlichkeit")
    gm, gt, sk = abb
    neu["punkte"] = {nr: _zeile(p, gm, gt) for nr, p in b["punkte"].items()}
    for k in neu["kanten"].values():
        k["kurve"] = geo_abbilden(k["kurve"], gm, gt, sk)
    for sch in neu["schalen"]:
        for f in sch:
            f["flaeche"] = geo_abbilden(f["flaeche"], gm, gt, sk)
            for lp in f["schleifen"]:
                for ce in lp:
                    ce["pcurve"] = None                               # Parameterkurven gelten nicht mehr
    neu["flaechen_geo"] = {nr: geo_abbilden(g, gm, gt, sk) for nr, g in b.get("flaechen_geo", {}).items()}
    if sk:
        neu["toleranz"] = b.get("toleranz", 0.0) * sk
    return neu


def teile(pfad):
    warn, erg = [], {}
    for name, b in oeffnen(pfad):                           # ZIP (20.3) oder altes compress-Paket (altformat)
        if b[:2] != b"Rz":
            continue
        # SolidDesigner 11: Teiltoleranz 1e-6, die Näherungen der Schnittkurven weichen aber bis 3e-3 mm ab
        alt = "SolidDesigner" in kopf_lesen(b).get("version", "")
        for seg in segmente_lesen(b, name, warn):
            if seg.klasse == "part_struct" and seg.von_klasse("body"):
                brep = BRep(seg).lesen()
                if alt:
                    brep["toleranz"] = max(brep["toleranz"], 3e-3)
                erg[seg.id] = brep
    return erg, warn


def main():
    rc = 0
    for pfad in sys.argv[1:]:
        erg, warn = teile(pfad)
        print(f"== {os.path.basename(pfad)}")
        for w in warn:
            print("   WARNUNG", w)
            rc = 1
        for tid, b in erg.items():
            n, befund = pruefen(b)
            typen = {}
            for f in (f for s in b["schalen"] for f in s):
                typen[f["flaeche"]["typ"]] = typen.get(f["flaeche"]["typ"], 0) + 1
            ktypen = {}
            for k in b["kanten"].values():
                ktypen[k["kurve"]["typ"]] = ktypen.get(k["kurve"]["typ"], 0) + 1
            print(f"   Teil {tid}: Flächen {typen}, Kanten {ktypen}, Euler {n['euler']}, "
                  f"Kantenenden ok {n['kantenenden']}, Punkte auf Fläche ok {n['punkte_auf_flaeche']}, "
                  f"Befunde {len(befund)}")
            for x in befund[:8]:
                print("     !", x)
            rc |= bool(befund)
    return rc


if __name__ == "__main__":
    sys.exit(main())
