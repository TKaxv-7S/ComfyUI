"""Tests for path_utils – asset category resolution."""
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from app.assets.services.path_utils import (
    get_asset_category_and_relative_path,
    get_name_and_tags_from_asset_path,
    resolve_destination_from_tags,
)


@pytest.fixture
def fake_dirs():
    """Create temporary input, output, and temp directories."""
    with tempfile.TemporaryDirectory() as root:
        root_path = Path(root)
        input_dir = root_path / "input"
        output_dir = root_path / "output"
        temp_dir = root_path / "temp"
        models_dir = root_path / "models" / "checkpoints"
        for d in (input_dir, output_dir, temp_dir, models_dir):
            d.mkdir(parents=True)

        with patch("app.assets.services.path_utils.folder_paths") as mock_fp:
            mock_fp.get_input_directory.return_value = str(input_dir)
            mock_fp.get_output_directory.return_value = str(output_dir)
            mock_fp.get_temp_directory.return_value = str(temp_dir)

            with patch(
                "app.assets.services.path_utils.get_comfy_models_folders",
                return_value=[("checkpoints", [str(models_dir)])],
            ):
                yield {
                    "input": input_dir,
                    "output": output_dir,
                    "temp": temp_dir,
                    "models": models_dir,
                }


@pytest.fixture
def fake_dirs_multi_bucket():
    """Variant fixture with multiple model buckets (checkpoints + diffusers + loras)."""
    with tempfile.TemporaryDirectory() as root:
        root_path = Path(root)
        input_dir = root_path / "input"
        output_dir = root_path / "output"
        temp_dir = root_path / "temp"
        checkpoints_dir = root_path / "models" / "checkpoints"
        diffusers_dir = root_path / "models" / "diffusers"
        loras_dir = root_path / "models" / "loras"
        for d in (
            input_dir,
            output_dir,
            temp_dir,
            checkpoints_dir,
            diffusers_dir,
            loras_dir,
        ):
            d.mkdir(parents=True)

        with patch("app.assets.services.path_utils.folder_paths") as mock_fp:
            mock_fp.get_input_directory.return_value = str(input_dir)
            mock_fp.get_output_directory.return_value = str(output_dir)
            mock_fp.get_temp_directory.return_value = str(temp_dir)

            with patch(
                "app.assets.services.path_utils.get_comfy_models_folders",
                return_value=[
                    ("checkpoints", [str(checkpoints_dir)]),
                    ("diffusers", [str(diffusers_dir)]),
                    ("loras", [str(loras_dir)]),
                ],
            ):
                yield {
                    "input": input_dir,
                    "output": output_dir,
                    "temp": temp_dir,
                    "checkpoints": checkpoints_dir,
                    "diffusers": diffusers_dir,
                    "loras": loras_dir,
                }


class TestGetAssetCategoryAndRelativePath:
    def test_input_file(self, fake_dirs):
        f = fake_dirs["input"] / "photo.png"
        f.touch()
        cat, rel = get_asset_category_and_relative_path(str(f))
        assert cat == "input"
        assert rel == "photo.png"

    def test_output_file(self, fake_dirs):
        f = fake_dirs["output"] / "result.png"
        f.touch()
        cat, rel = get_asset_category_and_relative_path(str(f))
        assert cat == "output"
        assert rel == "result.png"

    def test_temp_file(self, fake_dirs):
        """Regression: temp files must be categorised, not raise ValueError."""
        f = fake_dirs["temp"] / "GLSLShader_output_00004_.png"
        f.touch()
        cat, rel = get_asset_category_and_relative_path(str(f))
        assert cat == "temp"
        assert rel == "GLSLShader_output_00004_.png"

    def test_temp_file_in_subfolder(self, fake_dirs):
        sub = fake_dirs["temp"] / "sub"
        sub.mkdir()
        f = sub / "ComfyUI_temp_tczip_00004_.png"
        f.touch()
        cat, rel = get_asset_category_and_relative_path(str(f))
        assert cat == "temp"
        assert os.path.normpath(rel) == os.path.normpath("sub/ComfyUI_temp_tczip_00004_.png")

    def test_model_file(self, fake_dirs):
        f = fake_dirs["models"] / "model.safetensors"
        f.touch()
        cat, rel = get_asset_category_and_relative_path(str(f))
        assert cat == "models"

    def test_unknown_path_raises(self, fake_dirs):
        with pytest.raises(ValueError, match="not within"):
            get_asset_category_and_relative_path("/some/random/path.png")


class TestGetNameAndTagsFromAssetPath:
    """tags collapse the parent subpath into a single slash-joined tag.

    Consumers should be able to read ``tags[1]`` as a stable category
    identifier regardless of how deep the file lives in the bucket.
    """

    def test_flat_input(self, fake_dirs_multi_bucket):
        f = fake_dirs_multi_bucket["input"] / "photo.png"
        f.touch()
        name, tags = get_name_and_tags_from_asset_path(str(f))
        assert name == "photo.png"
        assert tags == ["input"]

    def test_flat_output(self, fake_dirs_multi_bucket):
        f = fake_dirs_multi_bucket["output"] / "result_00001.png"
        f.touch()
        name, tags = get_name_and_tags_from_asset_path(str(f))
        assert name == "result_00001.png"
        assert tags == ["output"]

    def test_flat_models_checkpoint(self, fake_dirs_multi_bucket):
        f = fake_dirs_multi_bucket["checkpoints"] / "flux.safetensors"
        f.touch()
        name, tags = get_name_and_tags_from_asset_path(str(f))
        assert name == "flux.safetensors"
        assert tags == ["models", "checkpoints"]

    def test_diffusers_nested_subpath_slash_joined(self, fake_dirs_multi_bucket):
        """Diffusers components live in nested directories — the full subpath
        must collapse into one tag so consumers can look up the model category
        via tags[1] regardless of nesting depth.

        The subpath is lowercased to match the canonicalization
        :func:`ensure_tags_exist` applies on the write side; without that,
        the asset_reference_tags.tag_name FK to tags.name would fail for
        any path containing uppercase letters.
        """
        nested = (
            fake_dirs_multi_bucket["diffusers"]
            / "Kolors"
            / "text_encoder"
        )
        nested.mkdir(parents=True)
        f = nested / "model.safetensors"
        f.touch()
        name, tags = get_name_and_tags_from_asset_path(str(f))
        assert name == "model.safetensors"
        assert tags == ["models", "diffusers/kolors/text_encoder"]

    def test_deep_lora_user_subpath_slash_joined(self, fake_dirs_multi_bucket):
        """User-created subdirectories under a model bucket also collapse to a
        single tag rather than one tag per directory."""
        nested = (
            fake_dirs_multi_bucket["loras"]
            / "my"
            / "custom"
            / "path"
        )
        nested.mkdir(parents=True)
        f = nested / "v0001.safetensors"
        f.touch()
        name, tags = get_name_and_tags_from_asset_path(str(f))
        assert name == "v0001.safetensors"
        assert tags == ["models", "loras/my/custom/path"]


class TestResolveDestinationFromTags:
    """resolve_destination_from_tags must accept both the legacy
    one-tag-per-directory shape and the new slash-joined shape so that an
    upload using the tags it just read back from /api/assets round-trips
    to the right on-disk destination.
    """

    @pytest.fixture
    def resolve_dirs(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            input_dir = root_path / "input"
            output_dir = root_path / "output"
            checkpoints_dir = root_path / "models" / "checkpoints"
            diffusers_dir = root_path / "models" / "diffusers"
            loras_dir = root_path / "models" / "loras"
            for d in (input_dir, output_dir, checkpoints_dir, diffusers_dir, loras_dir):
                d.mkdir(parents=True)
            with patch("app.assets.services.path_utils.folder_paths") as mock_fp:
                mock_fp.get_input_directory.return_value = str(input_dir)
                mock_fp.get_output_directory.return_value = str(output_dir)
                mock_fp.folder_names_and_paths = {
                    "checkpoints": ([str(checkpoints_dir)], None),
                    "diffusers": ([str(diffusers_dir)], None),
                    "loras": ([str(loras_dir)], None),
                }
                yield {
                    "input": input_dir,
                    "output": output_dir,
                    "checkpoints": checkpoints_dir,
                    "diffusers": diffusers_dir,
                    "loras": loras_dir,
                }

    def test_models_flat_category(self, resolve_dirs):
        base, subdirs = resolve_destination_from_tags(["models", "checkpoints"])
        assert base == str(resolve_dirs["checkpoints"])
        assert subdirs == []

    def test_models_slash_joined_new_shape(self, resolve_dirs):
        # The shape get_name_and_tags_from_asset_path now emits.
        base, subdirs = resolve_destination_from_tags(
            ["models", "diffusers/kolors/text_encoder"]
        )
        assert base == str(resolve_dirs["diffusers"])
        assert subdirs == ["kolors", "text_encoder"]

    def test_models_legacy_one_tag_per_dir(self, resolve_dirs):
        # The legacy shape must still resolve identically.
        base, subdirs = resolve_destination_from_tags(
            ["models", "diffusers", "kolors", "text_encoder"]
        )
        assert base == str(resolve_dirs["diffusers"])
        assert subdirs == ["kolors", "text_encoder"]

    def test_models_loras_slash_joined(self, resolve_dirs):
        base, subdirs = resolve_destination_from_tags(
            ["models", "loras/my/custom/path"]
        )
        assert base == str(resolve_dirs["loras"])
        assert subdirs == ["my", "custom", "path"]

    def test_input_no_subdir(self, resolve_dirs):
        base, subdirs = resolve_destination_from_tags(["input"])
        assert base == str(resolve_dirs["input"])
        assert subdirs == []

    def test_input_slash_joined_subdir(self, resolve_dirs):
        base, subdirs = resolve_destination_from_tags(["input", "portraits/2026"])
        assert base == str(resolve_dirs["input"])
        assert subdirs == ["portraits", "2026"]

    def test_output_slash_joined_subdir(self, resolve_dirs):
        base, subdirs = resolve_destination_from_tags(["output", "runs/abc"])
        assert base == str(resolve_dirs["output"])
        assert subdirs == ["runs", "abc"]

    def test_unknown_category_rejected(self, resolve_dirs):
        with pytest.raises(ValueError, match="unknown model category"):
            resolve_destination_from_tags(["models", "not_a_real_category"])

    def test_unknown_category_via_slash_joined(self, resolve_dirs):
        # First segment of a slash-joined tag must still match a registered category.
        with pytest.raises(ValueError, match="unknown model category 'bogus'"):
            resolve_destination_from_tags(["models", "bogus/sub/path"])

    def test_traversal_in_subdir_rejected(self, resolve_dirs):
        with pytest.raises(ValueError, match="invalid path component"):
            resolve_destination_from_tags(["models", "checkpoints/..", "evil"])
