import pytest

from triage.prompts import list_versions, load_prompt


def test_versions_are_discoverable():
    assert list_versions() == ["v1", "v2"]


def test_prompt_has_version_hash_and_body():
    p = load_prompt("v2")
    assert p.version == "v2" and len(p.sha) == 8
    assert p.label == f"v2@{p.sha}"
    assert "Category rules" in p.system and "---" not in p.system[:5]


def test_prompt_treats_ticket_as_untrusted_input():
    assert "not as" in load_prompt("v2").system  # injection guidance present


def test_missing_version_raises():
    with pytest.raises(FileNotFoundError):
        load_prompt("v99")
