import hashlib
import json

from spatial.v2.study.training_token_audit import audit_training_candidate


class _Tokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        rendered = " ".join(message["content"] for message in messages)
        return rendered + (" assistant" if add_generation_prompt else "")

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return text.split()


def test_training_token_audit_verifies_files_and_recommends_envelope(tmp_path) -> None:
    row = {
        "id": "row-1",
        "messages": [
            {"role": "system", "content": "system text"},
            {"role": "user", "content": "user text"},
            {"role": "assistant", "content": "Answer: A"},
        ],
        "metadata": {"base_id": "base-1"},
    }
    content = json.dumps(row) + "\n"
    relative = "single/train-natural.jsonl"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_text(content)
    manifest = {
        "files": {
            relative: {
                "bytes": len(content.encode()),
                "rows": 1,
                "sha256": hashlib.sha256(content.encode()).hexdigest(),
            }
        }
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    audit = audit_training_candidate(
        tmp_path,
        tokenizer=_Tokenizer(),
        tokenizer_name="fixture",
        tokenizer_revision="fixed",
        thresholds=(4, 8, 16),
    )

    assert audit["status"] == "passed"
    assert audit["recommended_train_max_tokens"] == 8
    assert audit["files"][relative]["training_tokens"]["max"] == 6
    assert audit["files"][relative]["rows"] == 1
