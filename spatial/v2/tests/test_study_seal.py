import json

import pytest

from spatial.v2.study.build import build_release
from spatial.v2.study.seal import audit_release
from spatial.v2.tests.test_context_budget import FixtureTokenizer
from spatial.v2.tests.test_study_release import config


def test_audit_rejects_engineering_and_missing_benchmark_roles(tmp_path):
    p = config(tmp_path)
    release = tmp_path / "release"
    build_release(p, release, tokenizer=FixtureTokenizer())
    with pytest.raises(ValueError, match="production"):
        audit_release(release)
    manifest = json.loads((release / "manifest.json").read_text())
    manifest["status"] = "candidate-awaiting-quality-review"
    (release / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="evaluation roles"):
        audit_release(release)
