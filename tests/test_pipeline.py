"""Unit tests for the nuclei segmentation and morphometry project.

Run with:  pytest -q
"""

import numpy as np
import pytest
from skimage import draw

from src import classical, metrics, ml_pixel, morphometry, synth


# --------------------------------------------------------------------------
# Synthetic data generator
# --------------------------------------------------------------------------
def test_field_shapes_and_label_consistency():
    f = synth.synth_field(shape=(256, 256), n_nuclei=25, seed=1)
    assert f.image.shape == f.labels.shape == (256, 256)
    present = set(np.unique(f.labels)) - {0}
    assert present == set(f.classes)
    assert f.n_nuclei == len(present)


def test_every_nucleus_has_a_known_class():
    f = synth.synth_field(shape=(256, 256), n_nuclei=30, seed=2)
    assert all(v in ("benign", "malignant") for v in f.classes.values())


def test_malignant_nuclei_are_larger_and_less_round():
    """The generator must actually encode the biology it claims to."""
    f = synth.synth_field(shape=(512, 512), n_nuclei=120,
                          malignant_fraction=0.5, seed=3)
    df = morphometry.measure_nuclei(f.labels, f.image)
    df["cls"] = df["label"].map(f.classes)
    ben = df[df.cls == "benign"]
    mal = df[df.cls == "malignant"]
    assert mal.area.median() > 1.4 * ben.area.median()
    assert mal.circularity.median() < ben.circularity.median()
    assert mal.eccentricity.median() > ben.eccentricity.median()


def test_images_contain_touching_nuclei():
    """If nothing touches, watershed is pointless and the benchmark is trivial."""
    from scipy import ndimage as ndi
    f = synth.synth_field(shape=(512, 512), n_nuclei=110, crowding=0.8, seed=4)
    touching = 0
    for i in np.unique(f.labels)[1:]:
        m = f.labels == i
        ring = ndi.binary_dilation(m, np.ones((3, 3))) & ~m
        if np.any(f.labels[ring] > 0):
            touching += 1
    assert touching / f.n_nuclei > 0.15


def test_illumination_field_is_uneven():
    """A global threshold should be genuinely challenged by the background."""
    f = synth.synth_field(shape=(256, 256), n_nuclei=20, seed=5)
    bg = f.image[f.labels == 0]
    assert bg.std() > 0.01


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def _two_disks(shape=(120, 120)):
    lab = np.zeros(shape, dtype=np.int32)
    rr, cc = draw.disk((40, 40), 15, shape=shape)
    lab[rr, cc] = 1
    rr, cc = draw.disk((80, 80), 15, shape=shape)
    lab[rr, cc] = 2
    return lab


def test_identical_masks_score_perfectly():
    lab = _two_disks()
    assert metrics.dice(lab > 0, lab > 0) == pytest.approx(1.0)
    assert metrics.iou_binary(lab > 0, lab > 0) == pytest.approx(1.0)
    assert metrics.aggregated_jaccard_index(lab, lab) == pytest.approx(1.0)
    assert metrics.object_scores(lab, lab)["f1_at_50"] == pytest.approx(1.0)


def test_disjoint_masks_score_zero():
    a = np.zeros((60, 60), np.int32)
    a[5:15, 5:15] = 1
    b = np.zeros((60, 60), np.int32)
    b[40:50, 40:50] = 1
    assert metrics.dice(a > 0, b > 0) == pytest.approx(0.0)
    assert metrics.aggregated_jaccard_index(a, b) == pytest.approx(0.0)


def test_aji_punishes_merging_two_nuclei_into_one():
    """Dice is blind to under-segmentation; AJI and object F1 are not."""
    gt = _two_disks()
    merged = (gt > 0).astype(np.int32)          # both nuclei labelled as one
    assert metrics.dice(gt > 0, merged > 0) == pytest.approx(1.0)
    assert metrics.aggregated_jaccard_index(gt, merged) < 0.6
    assert metrics.object_scores(gt, merged)["f1_at_50"] < 0.7


def test_object_scores_count_split_objects_as_errors():
    gt = np.zeros((80, 80), np.int32)
    rr, cc = draw.disk((40, 40), 18, shape=(80, 80))
    gt[rr, cc] = 1
    split = gt.copy()
    split[:40, :] = np.where(split[:40, :] > 0, 2, 0)   # cut it in half
    s = metrics.object_scores(gt, split)
    assert s["n_pred"] == 2 and s["n_gt"] == 1
    assert s["by_threshold"]["0.5"]["fp"] >= 1


def test_evaluate_segmentation_returns_all_expected_keys():
    lab = _two_disks()
    out = metrics.evaluate_segmentation(lab, lab)
    for k in ("dice", "iou", "aji", "f1_at_50", "mean_f1", "n_gt", "n_pred",
              "count_error_pct"):
        assert k in out


# --------------------------------------------------------------------------
# Classical segmentation
# --------------------------------------------------------------------------
def test_classical_segmentation_finds_roughly_the_right_count():
    f = synth.synth_field(shape=(384, 384), n_nuclei=40, snr=18, seed=6)
    seg = classical.segment(f.image)
    assert 0.6 * f.n_nuclei < seg.max() < 1.5 * f.n_nuclei


def test_illumination_correction_flattens_the_background():
    f = synth.synth_field(shape=(256, 256), n_nuclei=20, seed=7)
    corrected = classical.correct_illumination(f.image)
    bg_before = f.image[f.labels == 0]
    bg_after = corrected[f.labels == 0]
    assert bg_after.std() < bg_before.std()


def test_watershed_splits_two_touching_disks():
    shape = (100, 100)
    mask = np.zeros(shape, bool)
    for centre in ((50, 40), (50, 60)):
        rr, cc = draw.disk(centre, 16, shape=shape)
        mask[rr, cc] = True
    assert classical.watershed_split(mask, min_distance=5).max() == 2


# --------------------------------------------------------------------------
# Learned pixel classifier
# --------------------------------------------------------------------------
def test_feature_stack_matches_declared_feature_names():
    img = np.random.default_rng(0).random((64, 64)).astype(np.float32)
    stack = ml_pixel.feature_stack(img)
    assert stack.shape[:2] == img.shape
    assert stack.shape[2] == len(ml_pixel.feature_names())
    assert np.isfinite(stack).all()


def test_pixel_labels_produce_three_classes_with_boundaries_inside_nuclei():
    f = synth.synth_field(shape=(256, 256), n_nuclei=25, seed=8)
    pl = ml_pixel.pixel_labels(f.labels)
    assert set(np.unique(pl)) <= {ml_pixel.BACKGROUND, ml_pixel.INTERIOR,
                                  ml_pixel.BOUNDARY}
    assert (pl == ml_pixel.BOUNDARY).sum() > 0
    # boundary pixels must lie on foreground, never in the background
    assert np.all(f.labels[pl == ml_pixel.BOUNDARY] > 0)


def test_training_sample_is_class_balanced():
    f = synth.synth_field(shape=(256, 256), n_nuclei=25, seed=9)
    X, y = ml_pixel.sample_training_pixels(f.image, f.labels, n_per_class=500)
    counts = np.bincount(y, minlength=3)
    assert counts.min() > 0
    assert counts.max() / counts.min() < 1.5


def test_probability_maps_are_valid_distributions():
    f = synth.synth_field(shape=(128, 128), n_nuclei=12, seed=10)
    X, y = ml_pixel.sample_training_pixels(f.image, f.labels, n_per_class=400)
    model = ml_pixel.build_model()
    model.fit(X, y)
    proba = ml_pixel.predict_probabilities(model, f.image)
    assert proba.shape == (128, 128, 3)
    assert np.allclose(proba.sum(axis=-1), 1.0, atol=1e-5)


# --------------------------------------------------------------------------
# Morphometry
# --------------------------------------------------------------------------
def test_circularity_of_a_disk_is_close_to_one():
    lab = np.zeros((120, 120), np.int32)
    rr, cc = draw.disk((60, 60), 30, shape=(120, 120))
    lab[rr, cc] = 1
    df = morphometry.measure_nuclei(lab, lab.astype(float))
    assert 0.85 < df.circularity.iloc[0] < 1.15
    assert df.eccentricity.iloc[0] < 0.3
    assert df.aspect_ratio.iloc[0] == pytest.approx(1.0, abs=0.15)


def test_elongated_ellipse_has_high_aspect_ratio_and_eccentricity():
    lab = np.zeros((160, 160), np.int32)
    rr, cc = draw.ellipse(80, 80, 12, 45, shape=(160, 160))
    lab[rr, cc] = 1
    df = morphometry.measure_nuclei(lab, lab.astype(float))
    assert df.aspect_ratio.iloc[0] > 2.5
    assert df.eccentricity.iloc[0] > 0.8
    assert df.circularity.iloc[0] < 0.9


def test_one_row_per_segmented_object():
    f = synth.synth_field(shape=(256, 256), n_nuclei=20, seed=11)
    df = morphometry.measure_nuclei(f.labels, f.image, "fieldX")
    assert len(df) == f.n_nuclei
    assert set(morphometry.FEATURE_COLUMNS).issubset(df.columns)
    assert df[morphometry.FEATURE_COLUMNS].notna().all().all()


def test_ground_truth_assignment_matches_a_perfect_segmentation():
    f = synth.synth_field(shape=(256, 256), n_nuclei=20, seed=12)
    df = morphometry.measure_nuclei(f.labels, f.image)
    df = morphometry.assign_ground_truth_class(df, f.labels, f.labels, f.classes)
    assert (df.true_class == "unmatched").sum() == 0
    assert set(df.true_class) <= {"benign", "malignant"}


def test_pleomorphism_score_reports_variability():
    f = synth.synth_field(shape=(256, 256), n_nuclei=25, seed=13)
    df = morphometry.measure_nuclei(f.labels, f.image)
    df["pred_class"] = "benign"
    score = morphometry.pleomorphism_score(df)
    assert score["n_nuclei"] == len(df)
    assert score["area_cv_pct"] > 0
    assert score["malignant_fraction"] == pytest.approx(0.0)
