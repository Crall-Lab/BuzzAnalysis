import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).with_name("analyze_clean_data_0.2.py")
SPEC = importlib.util.spec_from_file_location("analyze_clean_data_0_2", MODULE_PATH)
analyze_clean_data_0_2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyze_clean_data_0_2)


def test_find_brood_map_matches_colony_and_date_with_zero_padding(tmp_path):
    brood_dir = tmp_path / "brood"
    brood_dir.mkdir()

    target = brood_dir / "col_01-2021-06-12-nest_image.csv"
    target.write_text("x,y\n", encoding="utf-8")
    (brood_dir / "col_01-2021-06-13-nest_image.csv").write_text("x,y\n", encoding="utf-8")
    (brood_dir / "col_02-2021-06-12-nest_image.csv").write_text("x,y\n", encoding="utf-8")

    result = analyze_clean_data_0_2._find_brood_map("col_1-2021-06-12", str(brood_dir), "_nest_image.csv")

    assert result == str(target)


def test_find_brood_map_finds_nested_match(tmp_path):
    brood_dir = tmp_path / "brood"
    nested_dir = brood_dir / "2021-06-12"
    nested_dir.mkdir(parents=True)

    target = nested_dir / "col_15-2021-06-12-nest_image.csv"
    target.write_text("x,y\n", encoding="utf-8")

    result = analyze_clean_data_0_2._find_brood_map("col_15-2021-06-12", str(brood_dir), "_nest_image.csv")

    assert result == str(target)


def test_find_brood_map_raises_for_ambiguous_matches(tmp_path):
    brood_dir = tmp_path / "brood"
    brood_dir.mkdir()

    (brood_dir / "col_1-2021-06-12-nest_image.csv").write_text("x,y\n", encoding="utf-8")
    (brood_dir / "col_01-2021-06-12-nest_image.csv").write_text("x,y\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Multiple brood maps matched"):
        analyze_clean_data_0_2._find_brood_map("col_1-2021-06-12", str(brood_dir), "_nest_image.csv")
