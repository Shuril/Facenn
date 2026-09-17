import numpy as np
import pytest
from facenn.database import FaceDB


def test_facedb_basic():
    db = FaceDB(db_path=":memory:")
    assert db.count() == 0
    assert db.list_identities() == []

    # Insert Alice
    alice_emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    id1 = db.add_face(alice_emb, "Alice")
    assert id1 is not None
    assert db.count() == 1
    assert "Alice" in db.list_identities()

    # Insert Bob
    bob_emb = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    db.add_face(bob_emb, "Bob")
    assert db.count() == 2

    # Query for Alice (exact)
    query_alice = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    results = db.search(query_alice, k=2, threshold=0.1)
    assert len(results) == 1
    assert results.iloc[0]["identity"] == "Alice"
    assert pytest.approx(results.iloc[0]["distance"], abs=1e-5) == 0.0

    # Query with wider threshold should find Alice first, then Bob
    results_all = db.search(query_alice, k=2, threshold=1.5)
    assert len(results_all) == 2
    assert results_all.iloc[0]["identity"] == "Alice"
    assert results_all.iloc[1]["identity"] == "Bob"

    # Delete Alice
    deleted = db.delete(identity="Alice")
    assert deleted == 1
    assert db.count() == 1
    assert "Alice" not in db.list_identities()
    assert "Bob" in db.list_identities()

    # Clear
    db.clear()
    assert db.count() == 0
