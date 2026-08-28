"""Métriques locales et retour depuis le NLE.

Les métriques sont validées sur des clips ffmpeg générés à la volée : on
dégrade une source connue d'une seule manière à la fois, et on vérifie que
c'est bien la métrique correspondante qui chute. C'est le seul contrôle
possible tant qu'on n'a pas de rushes réels annotés — et c'est aussi pourquoi
les seuils de `src/video/metrics.py` sont un calage provisoire.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.conform import ConformedClip, diff, parse_fcpxml, parse_time  # noqa: E402
from src.models import EditPlan, TimelineEdit, VideoSegment, segment_key  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg absent")

def make_clip(path: Path, extra_vf: str = "") -> str:
    """Un clip de 2 s à partir d'une image FIXE bouclée.

    Important : `testsrc2` est une mire *animée*. L'utiliser comme référence
    « caméra fixe » injecte du mouvement de contenu et rend tout test de
    stabilité ininterprétable — c'est ce qui avait masqué le fait que la
    mesure confondait mouvement de caméra et mouvement de sujet.
    """
    still = path.parent / "still.png"
    if not still.exists():
        subprocess.run(
            f"ffmpeg -y -loglevel error -f lavfi -i testsrc2=size=1600x900:rate=1:duration=1 "
            f"-frames:v 1 {still}",
            shell=True, check=True,
        )
    vf = "crop=900:520:300:180" + extra_vf
    subprocess.run(
        f'ffmpeg -y -loglevel error -loop 1 -i {still} -t 2 -r 25 -vf "{vf}" '
        f"-c:v libx264 -pix_fmt yuv420p -crf 12 {path}",
        shell=True, check=True,
    )
    return str(path)


def _moving(tmp_path: Path, name: str, x_expr: str, y_expr: str) -> str:
    """Même image fixe, fenêtre de recadrage animée : seule la caméra bouge."""
    still = tmp_path / "still.png"
    if not still.exists():
        make_clip(tmp_path / "_seed.mp4")
    out = tmp_path / name
    subprocess.run(
        f'ffmpeg -y -loglevel error -loop 1 -i {still} -t 2 -r 25 '
        f'-vf "crop=900:520:{x_expr}:{y_expr}" -c:v libx264 -pix_fmt yuv420p -crf 12 {out}',
        shell=True, check=True,
    )
    return str(out)


def _static_camera_moving_subject(tmp_path: Path) -> str:
    """Le cas qui a fait abandonner la corrélation de phase : caméra posée,
    sujet qui traverse. Doit ressortir parfaitement stable."""
    still = tmp_path / "still.png"
    if not still.exists():
        make_clip(tmp_path / "_seed.mp4")
    out = tmp_path / "sujet.mp4"
    subprocess.run(
        f"ffmpeg -y -loglevel error -loop 1 -i {still} "
        f"-f lavfi -i color=c=red:size=160x300:rate=25 "
        f'-filter_complex "[0:v]crop=900:520:300:180[bg];[bg][1:v]overlay=x=\'60+240*t\':y=120" '
        f"-t 2 -r 25 -c:v libx264 -pix_fmt yuv420p -crf 12 {out}",
        shell=True, check=True,
    )
    return str(out)


def measure(path: str):
    from src.video.metrics import measure_segment

    return measure_segment(path, 0.0, 2.0, n_samples=3)


# ── Métriques ───────────────────────────────────────────────────────────────


def test_blur_lowers_sharpness_only(tmp_path):
    sharp = measure(make_clip(tmp_path / "net.mp4"))
    soft = measure(make_clip(tmp_path / "flou.mp4", ",gblur=sigma=4"))
    assert soft.sharpness < sharp.sharpness - 0.3
    assert soft.exposure >= sharp.exposure - 0.05, "le flou ne doit pas toucher l'exposition"


def test_sharpness_is_content_normalised(tmp_path):
    """Le piège de la variance du Laplacien brute : une image très texturée
    score haut même floue. Le rapport auto-référencé doit résister."""
    from src.video.metrics import _sharpness_ratio
    import cv2

    plain = make_clip(tmp_path / "plain.mp4")
    textured = make_clip(tmp_path / "tex.mp4", ",noise=alls=22:allf=t+u")
    blurred_tex = make_clip(tmp_path / "tex_flou.mp4", ",noise=alls=22:allf=t+u,gblur=sigma=4")

    ratios = {}
    for name, path in (("plain", plain), ("tex", textured), ("tex_flou", blurred_tex)):
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_MSEC, 1000)
        ok, frame = cap.read()
        cap.release()
        assert ok
        ratios[name] = _sharpness_ratio(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    assert ratios["tex_flou"] < ratios["tex"], "le flou doit faire chuter le rapport"
    assert ratios["tex_flou"] < ratios["plain"], (
        "une image texturée mais floue ne doit pas battre une image nette peu texturée — "
        "c'est exactement ce que la variance brute du Laplacien se trompe à faire"
    )


def _blown(tmp_path: Path, name: str = "crame.mp4") -> str:
    """Écrêtage franc : un aplat blanc pur couvrant ~18 % du cadre.

    Pousser la luminance d'une mire ne suffit pas — même à +1.6 elle n'écrête
    que 0,9 % des pixels, sous le seuil de tolérance. Le test doit produire le
    défaut qu'il prétend mesurer, pas s'en approcher.
    """
    still = tmp_path / "still.png"
    if not still.exists():
        make_clip(tmp_path / "_seed.mp4")
    out = tmp_path / name
    subprocess.run(
        f"ffmpeg -y -loglevel error -loop 1 -i {still} "
        f"-f lavfi -i color=c=white:size=900x95:rate=25 "
        f'-filter_complex "[0:v]crop=900:520:300:180[bg];[bg][1:v]overlay=x=0:y=0" '
        f"-t 2 -r 25 -c:v libx264 -pix_fmt yuv420p -crf 12 {out}",
        shell=True, check=True,
    )
    return str(out)


def test_clipping_lowers_exposure_only(tmp_path):
    ok = measure(make_clip(tmp_path / "ok.mp4"))
    blown = measure(_blown(tmp_path))
    assert ok.exposure > 0.9
    assert blown.exposure < 0.5
    assert blown.sharpness > 0.5, "l'écrêtage ne doit pas faire chuter la netteté"


def test_pan_is_not_penalised_but_shake_is(tmp_path):
    """Un panoramique est intentionnel et régulier ; un tremblement ne l'est pas.
    C'est l'incohérence du déplacement qui pénalise, pas son amplitude."""
    pan = measure(_moving(tmp_path, "pano.mp4", "'200+180*t'", "180"))
    shake = measure(_moving(tmp_path, "shake.mp4", "'300+90*sin(41*t)'", "'180+70*sin(53*t)'"))
    assert pan.stability > 0.75, f"panoramique pénalisé à tort ({pan.stability:.2f})"
    assert shake.stability is not None, (
        "un plan très secoué fait échouer le suivi de points : renvoyer None "
        "est acceptable, renvoyer 1.0 récompenserait le pire plan du lot"
    )
    assert shake.stability < 0.4, f"tremblement non détecté ({shake.stability:.2f})"
    assert shake.motion > pan.motion


def test_moving_subject_is_not_camera_shake(tmp_path):
    """Une caméra posée devant une piste de danse doit rester « stable ».
    RANSAC traite le sujet comme un outlier ; l'arrière-plan donne la caméra."""
    m = measure(_static_camera_moving_subject(tmp_path))
    assert m.stability > 0.9, f"sujet mobile pris pour un tremblement ({m.stability:.2f})"
    assert m.motion < 1.0, f"mouvement caméra fantôme : {m.motion:.2f} px/img"


def test_technical_is_the_worst_defect_not_the_mean(tmp_path):
    """Un plan net et stable mais cramé doit être disqualifié : un monteur le
    jette, une moyenne le sauverait."""
    blown = measure(_blown(tmp_path, "crame2.mp4"))
    assert blown.sharpness > 0.5 and (blown.stability or 1.0) > 0.8
    assert blown.technical == blown.exposure < 0.5, (
        "une moyenne sauverait ce plan ; le minimum le disqualifie, comme un monteur"
    )


def test_unmeasurable_stability_is_none_not_perfect(tmp_path):
    """Régression : le suivi de points échoue d'abord sur les plans très
    secoués. Une valeur neutre par défaut leur donnait la meilleure note."""
    from src.video.metrics import SegmentMetrics

    m = SegmentMetrics(segment_index=0, sharpness=0.9, exposure=0.9, stability=None)
    assert m.technical == 0.9, "une mesure absente est ignorée, pas comptée comme parfaite"
    assert m.stability is None


def test_unreadable_file_fails_explicitly(tmp_path):
    from src.video.metrics import measure_segment

    bad = tmp_path / "pas_une_video.mp4"
    bad.write_bytes(b"nope")
    m = measure_segment(str(bad), 0.0, 2.0)
    assert m.failed and m.notes, "l'échec doit être porté par le contrat, pas maquillé"


# ── Retour depuis le NLE ────────────────────────────────────────────────────


def fake_plan(n: int = 6) -> EditPlan:
    edits = []
    for i in range(n):
        start = 3.0 * i
        edits.append(
            TimelineEdit(
                order=i,
                segment=VideoSegment(
                    source_file=f"/rushes/rush_{i % 2}.mp4",
                    start_time=start, end_time=start + 2.0, duration=2.0,
                    quality_score=0.8, semantic_tags=[], emotion="neutral",
                    suggested_role="b_roll",
                    source_key=segment_key(f"/rushes/rush_{i % 2}.mp4", start),
                ),
                transition_in="cut", transition_duration=0.0,
                audio_level=1.0, speed_factor=1.0,
            )
        )
    return EditPlan(title="t", total_duration=2.0 * n, narrative_arc="", edits=edits)


def as_conformed(plan: EditPlan, keep: list[int], trim: dict[int, float] | None = None):
    trim = trim or {}
    out = []
    offset = 0.0
    for rank, i in enumerate(keep):
        seg = plan.edits[i].segment
        dur = trim.get(i, seg.duration)
        out.append(
            ConformedClip(
                source_file=seg.source_file, start=seg.start_time,
                end=seg.start_time + dur, record_offset=offset, order=rank,
            )
        )
        offset += dur
    return out


def test_parse_time_handles_rationals():
    assert parse_time("150/25s") == 6.0
    assert parse_time("6s") == 6.0
    assert parse_time("0s") == 0.0
    assert parse_time(None) == 0.0
    assert parse_time("garbage") == 0.0


def test_roundtrip_through_our_own_fcpxml(tmp_path):
    from src.export import export_fcpxml

    plan = fake_plan(5)
    out = export_fcpxml(plan, str(tmp_path / "t.fcpxml"), fps=25.0)
    clips = parse_fcpxml(out)
    assert len(clips) == 5
    for c, e in zip(clips, sorted(plan.edits, key=lambda x: x.order)):
        assert Path(c.source_file).name == Path(e.segment.source_file).name
        assert abs(c.duration - e.segment.duration) < 0.05


def test_dropped_shots_become_ban_keys():
    plan = fake_plan(6)
    report = diff(plan, as_conformed(plan, keep=[0, 2, 3, 5]))
    assert len(report.kept) == 4
    assert report.dropped == [
        plan.edits[1].segment.source_key,
        plan.edits[4].segment.source_key,
    ]
    assert report.agreement == pytest.approx(4 / 6)


def test_trimmed_shot_counts_as_kept_not_dropped():
    """Un monteur rogne presque toujours. Compter un plan raccourci comme
    « écarté puis ajouté » fausserait le taux d'accord dans les deux sens."""
    plan = fake_plan(4)
    report = diff(plan, as_conformed(plan, keep=[0, 1, 2, 3], trim={2: 1.0}))
    assert report.dropped == [] and report.added == []
    assert len(report.kept) == 4
    assert len(report.trimmed) == 1 and plan.edits[2].segment.source_key in report.trimmed[0]


def test_shots_added_by_the_editor_are_reported():
    plan = fake_plan(3)
    conformed = as_conformed(plan, keep=[0, 1, 2])
    conformed.append(
        ConformedClip(
            source_file="/rushes/rush_1.mp4", start=90.0, end=92.0,
            record_offset=99.0, order=3,
        )
    )
    report = diff(plan, conformed)
    assert report.added == ["rush_1@90.000"]
    assert len(report.kept) == 3


def test_full_agreement_yields_no_veto():
    plan = fake_plan(5)
    report = diff(plan, as_conformed(plan, keep=list(range(5))))
    assert report.agreement == 1.0
    assert not report.dropped and not report.added
    assert "Aucun veto" in report.render()


def test_empty_timeline_drops_everything():
    plan = fake_plan(3)
    report = diff(plan, [])
    assert report.agreement == 0.0
    assert len(report.dropped) == 3
