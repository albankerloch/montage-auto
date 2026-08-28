"""Métriques image mesurées en local, en pleine résolution.

Pourquoi elles ne pouvaient pas venir du modèle. La vignette envoyée à l'API
fait 640 px de large en JPEG q:v 3 : à cette échelle, un rush 4K légèrement
flou et un rush net sont visuellement identiques, et un écrêtage d'un demi-stop
a disparu dans la recompression. On demandait donc au modèle de juger la mise
au point et l'exposition à partir d'une image où l'information n'existe plus.
Ces trois grandeurs se mesurent, et se mesurent sur les pixels d'origine.

Le modèle garde ce que la vignette permet réellement de juger : le cadre, le
sujet, l'intérêt narratif.

### La nuance que le discours habituel escamote

« Netteté = variance du Laplacien » est faux tel quel : la variance du Laplacien
dépend du *contenu* autant que de la mise au point. Un mur de briques net score
bien plus haut qu'un ciel dégagé net. Utilisée pour comparer des plans entre
eux — exactement ce que fait le solveur — elle classe la texture, pas le point.

On mesure donc un rapport auto-référencé : on floute délibérément l'image et on
regarde ce que le flou détruit. Une image nette perd énormément de hautes
fréquences, une image déjà molle n'en perd presque pas. Le rapport est sans
dimension et le contenu se simplifie largement.

L'exposition et la stabilité n'ont pas ce défaut : le taux de pixels écrêtés et
l'amplitude du mouvement inter-image sont directement comparables d'un plan à
l'autre.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

# Seuils : au-delà, la note tombe à 0.
# Les hautes lumières écrêtées sont irrécupérables ; les noirs bouchés sont
# souvent intentionnels et se rattrapent à l'étalonnage. On les pondère donc
# différemment. Le seuil est volontairement tolérant : une fenêtre cramée
# derrière des mariés est banale et ne disqualifie pas un plan.
_CLIP_TOLERANCE = 0.08
_SHADOW_WEIGHT = 0.3
_JITTER_TOLERANCE = 6.0  # px de déplacement résiduel par image, à 320 px de large
_FLOW_WIDTH = 320        # le mouvement est global : le sous-échantillonnage est licite
# Le rapport de flou couvre deux ordres de grandeur (≈3 pour une image molle,
# ≈150 pour une image piquée) : il se normalise en log, pas linéairement.
# Ces bornes sont un calage sur mires de synthèse ; elles demandent à être
# revues sur de vrais rushes, optique et grain compris.
_SHARP_FLOOR = 2.0
_SHARP_CEIL = 40.0


class SegmentMetrics(BaseModel):
    """Mesures d'un plan. Toutes les notes sont dans [0, 1], 1 = sans défaut."""

    segment_index: int
    sharpness: float = Field(ge=0.0, le=1.0, description="Rapport de flou normalisé")
    exposure: float = Field(ge=0.0, le=1.0, description="1 = aucun écrêtage")
    stability: float | None = Field(
        default=None,
        description="1 = aucun tremblement résiduel. None = non mesurable, et surtout "
                    "pas 1.0 : un plan trop secoué fait échouer le suivi de points, "
                    "donc la valeur neutre récompenserait exactement les pires plans.",
    )
    motion: float | None = Field(
        default=None, description="Déplacement de la caméra, px/image à 320 px"
    )
    frames_measured: int = 0
    failed: bool = False
    notes: str = ""

    @property
    def technical(self) -> float:
        """Un plan vaut son pire défaut technique, pas sa moyenne.

        Une moyenne laisse passer un plan cramé mais net et stable ; un monteur
        le jette. Le minimum reproduit ce comportement. Une mesure absente est
        ignorée plutôt que remplacée par une valeur neutre.
        """
        known = [v for v in (self.sharpness, self.exposure, self.stability) if v is not None]
        return min(known) if known else 0.0


def _failed(index: int, why: str) -> SegmentMetrics:
    return SegmentMetrics(
        segment_index=index, sharpness=0.0, exposure=0.0, stability=None,
        motion=None, frames_measured=0, failed=True, notes=why,
    )


def _sharpness_ratio(gray) -> float:
    """Rapport entre l'énergie haute fréquence de l'image et celle de sa version floutée.

    Sans dimension, donc comparable entre plans — contrairement à la variance
    du Laplacien brute.
    """
    import cv2

    lap = cv2.Laplacian(gray, cv2.CV_64F).var()
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.6)
    lap_blur = cv2.Laplacian(blurred, cv2.CV_64F).var()
    if lap_blur < 1e-6:
        return 1.0
    return float(lap / lap_blur)


def _exposure_clipping(gray) -> float:
    """Fraction de pixels collés aux deux extrémités de l'histogramme."""
    import numpy as np

    total = gray.size
    low = float((gray <= 2).sum()) / total
    high = float((gray >= 253).sum()) / total
    return _SHADOW_WEIGHT * low + high


def _camera_shift(prev_small, next_small) -> tuple[float, float] | None:
    """Déplacement de la CAMÉRA entre deux images. `None` si non estimable.

    Deux approches ont été écartées avant celle-ci, et la raison compte :

    1. Flux dense de Farneback, amplitude médiane. Sur un plan comportant de
       larges aplats, la médiane est écrasée par les zones sans texture : un
       panoramique franc ressortait à 0 px.

    2. Corrélation de phase. Rapide, mais elle décroche sur les motifs
       périodiques — briques, grillages, rideaux — et surtout elle mesure le
       mouvement *apparent* sans distinguer la caméra du sujet. Une caméra
       posée devant une piste de danse en ressortait « instable ».

    Ici on estime la transformation dominante entre les deux images par RANSAC
    sur des points suivis. Les inliers décrivent l'arrière-plan, donc la
    caméra ; les sujets qui bougent deviennent des outliers et sont ignorés.
    C'est le procédé des stabilisateurs vidéo, et c'est la seule des trois qui
    réponde à la question posée.
    """
    import cv2
    import numpy as np

    pts = cv2.goodFeaturesToTrack(
        prev_small, maxCorners=200, qualityLevel=0.01, minDistance=8, blockSize=7
    )
    if pts is None or len(pts) < 12:
        return None  # scène trop pauvre en texture : on ne devine pas

    # Fenêtre large et pyramide profonde : un plan très secoué déplace les
    # points de plusieurs dizaines de pixels d'une image à l'autre, et un
    # suivi trop étroit échoue précisément sur les plans qu'on veut détecter.
    nxt, status, _err = cv2.calcOpticalFlowPyrLK(
        prev_small, next_small, pts, None, winSize=(31, 31), maxLevel=4
    )
    if nxt is None:
        return None
    ok = status.ravel() == 1
    if ok.sum() < 12:
        return None

    matrix, inliers = cv2.estimateAffinePartial2D(
        pts[ok], nxt[ok], method=cv2.RANSAC, ransacReprojThreshold=1.5
    )
    if matrix is None or inliers is None or inliers.sum() < 8:
        return None
    return float(matrix[0, 2]), float(matrix[1, 2])


def _normalize(value: float, floor: float, ceil: float) -> float:
    if ceil <= floor:
        return 0.0
    return max(0.0, min(1.0, (value - floor) / (ceil - floor)))


def measure_segment(
    path: str,
    start: float,
    end: float,
    index: int = 0,
    n_samples: int = 3,
) -> SegmentMetrics:
    """Mesure un plan sur `n_samples` points, en pleine résolution.

    À chaque point on lit deux images consécutives : la première sert à la
    netteté et à l'exposition, la paire au mouvement. Un plan mesuré sur une
    seule image serait aussi peu fiable que la vignette qu'on veut remplacer.
    """
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return _failed(index, f"ouverture impossible: {path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        duration = max(0.0, end - start)
        if duration <= 0:
            return _failed(index, "durée nulle")

        # Points d'échantillonnage à l'intérieur du plan, bords évités : les
        # premières images d'un plan portent souvent le stabilisateur qui se
        # cale, et ne représentent pas le reste.
        offsets = [0.25, 0.5, 0.75][:n_samples] if n_samples <= 3 else [
            (i + 1) / (n_samples + 1) for i in range(n_samples)
        ]

        sharps: list[float] = []
        clips: list[float] = []
        shifts: list[tuple[float, float]] = []
        measured = 0

        for frac in offsets:
            cap.set(cv2.CAP_PROP_POS_MSEC, (start + duration * frac) * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharps.append(_sharpness_ratio(gray))
            clips.append(_exposure_clipping(gray))
            measured += 1

            ok2, frame2 = cap.read()  # image immédiatement suivante
            if ok2 and frame2 is not None:
                scale = _FLOW_WIDTH / max(1, gray.shape[1])
                small_a = cv2.resize(gray, None, fx=scale, fy=scale)
                small_b = cv2.resize(
                    cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY), None, fx=scale, fy=scale
                )
                shift = _camera_shift(small_a, small_b)
                if shift is not None:
                    shifts.append(shift)

        if measured == 0:
            return _failed(index, "aucune image lisible")

        sharpness = _normalize(
            float(np.log10(max(1.0, np.median(sharps)))),
            float(np.log10(_SHARP_FLOOR)),
            float(np.log10(_SHARP_CEIL)),
        )
        exposure = 1.0 - min(1.0, float(np.max(clips)) / _CLIP_TOLERANCE)
        # Le mouvement n'est pas un défaut : un panoramique est intentionnel et
        # régulier. Ce qui pénalise, c'est le mouvement *incohérent* d'un point
        # d'échantillonnage à l'autre — signature d'une caméra tenue à la main.
        if len(shifts) >= 2:
            dxs = np.array([d[0] for d in shifts])
            dys = np.array([d[1] for d in shifts])
            motion = float(np.median(np.hypot(dxs, dys)))
            jitter = float(np.hypot(dxs.std(), dys.std()))
            stability = 1.0 - min(1.0, jitter / _JITTER_TOLERANCE)
        else:
            # Moins de deux estimations fiables. On ne met surtout PAS 1.0 : le
            # suivi de points échoue d'abord sur les plans très secoués, donc
            # une valeur neutre donnerait la meilleure note aux pires plans.
            motion, stability = None, None

        return SegmentMetrics(
            segment_index=index,
            sharpness=sharpness,
            exposure=exposure,
            stability=stability,
            motion=motion,
            frames_measured=measured,
        )
    except Exception as e:  # noqa: BLE001
        return _failed(index, f"mesure impossible: {e}")
    finally:
        cap.release()


def measure_scenes(
    path: str, scenes: list[tuple[float, float]], n_samples: int = 3
) -> list[SegmentMetrics]:
    return [
        measure_segment(path, s, e, index=i, n_samples=n_samples)
        for i, (s, e) in enumerate(scenes)
    ]
