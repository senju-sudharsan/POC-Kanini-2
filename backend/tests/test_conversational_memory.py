"""Deterministic tests for conversational memory, name parser, and personal fact routing."""

import pytest
from fastapi.testclient import TestClient
from poc_kanini.main import app
from poc_kanini.graphs.supervisor import SupervisorRouter
from poc_kanini.graphs.specialists import _clean_name, _deterministic_general_response


def test_clean_name_bounded_clause_parser():
    """Verify name parsing stops at natural clause and conjunction boundaries."""
    # 1. Plain name
    assert _clean_name("sudharsan") == "Sudharsan"
    # 2. Conjunction 'and'
    assert _clean_name("sudharsan and my dog's name is dogesh") == "Sudharsan"
    # 3. Comma followed by conjunction
    assert _clean_name("sudharsan, and my dog's name is dogesh") == "Sudharsan"
    # 4. Conjunction 'and' with project
    assert _clean_name("sudharsan and my project is apollo") == "Sudharsan"
    # 5. Call me
    assert _clean_name("sudharsan") == "Sudharsan"
    # 6. Stopwords are filtered
    assert _clean_name("a") is None
    assert _clean_name("looking for data") is None


def test_supervisor_routing_personal_facts_with_attached_documents():
    """Verify supervisor routes dog, project, and personal name questions to 'general' even with documents attached."""
    router = SupervisorRouter()

    # Dog name queries
    d1 = router._deterministic_route("what is my dogs name?", has_documents=True)
    assert d1 is not None and d1.route == "general"

    d2 = router._deterministic_route("what is my dog's name?", has_documents=True)
    assert d2 is not None and d2.route == "general"

    d3 = router._deterministic_route("do you know my dog's name?", has_documents=True)
    assert d3 is not None and d3.route == "general"

    d4 = router._deterministic_route("my dog's name is dogesh", has_documents=True)
    assert d4 is not None and d4.route == "general"

    # Project name queries
    p1 = router._deterministic_route("what is my project name?", has_documents=True)
    assert p1 is not None and p1.route == "general"

    p2 = router._deterministic_route("my project is Apollo", has_documents=True)
    assert p2 is not None and p2.route == "general"

    # Personal name queries
    u1 = router._deterministic_route("what is my name?", has_documents=True)
    assert u1 is not None and u1.route == "general"

    u2 = router._deterministic_route("my name is sudharsan and my dog's name is dogesh", has_documents=True)
    assert u2 is not None and u2.route == "general"


def test_supervisor_routing_dataset_and_document_not_hijacked():
    """Verify dataset name and document name queries route to data/rag, not personal memory."""
    router = SupervisorRouter()

    d_data = router._deterministic_route("what columns are in this csv?", has_documents=True)
    assert d_data is not None and d_data.route == "data"

    d_rag = router._deterministic_route("what does the document say about vector databases?", has_documents=True)
    assert d_rag is not None and d_rag.route == "rag"


def test_conversational_memory_combined_name_and_dog_flow():
    """End-to-end multi-turn session with combined name and dog facts."""
    with TestClient(app) as client:
        # Turn 1: User declares name and dog name in a single sentence
        r1 = client.post("/api/chat", json={
            "messages": [{"role": "user", "content": "my name is sudharsan and my dogs name is dogesh"}]
        })
        assert r1.status_code == 200
        b1 = r1.json()
        thread_id = b1["thread_id"]
        c1 = b1["message"]["content"]
        assert "Sudharsan" in c1
        assert "Sudharsan And" not in c1

        # Turn 2: User asks for their own name
        r2 = client.post("/api/chat", json={
            "thread_id": thread_id,
            "messages": [{"role": "user", "content": "what is my name?"}]
        })
        assert r2.status_code == 200
        c2 = r2.json()["message"]["content"]
        assert "Sudharsan" in c2
        assert "Dogesh" not in c2

        # Turn 3: User asks for their dog's name
        r3 = client.post("/api/chat", json={
            "thread_id": thread_id,
            "messages": [{"role": "user", "content": "what is my dogs name?"}]
        })
        assert r3.status_code == 200
        c3 = r3.json()["message"]["content"]
        assert "Dogesh" in c3
        assert "Sudharsan" not in c3


def test_conversational_memory_project_and_name_flow():
    """Multi-turn session with project and user name."""
    with TestClient(app) as client:
        # Turn 1: Project statement
        r1 = client.post("/api/chat", json={
            "messages": [{"role": "user", "content": "my project is named Apollo"}]
        })
        assert r1.status_code == 200
        thread_id = r1.json()["thread_id"]

        # Turn 2: Ask for project name
        r2 = client.post("/api/chat", json={
            "thread_id": thread_id,
            "messages": [{"role": "user", "content": "what is my project name?"}]
        })
        assert r2.status_code == 200
        assert "Apollo" in r2.json()["message"]["content"]

        # Turn 3: Name not yet provided
        r3 = client.post("/api/chat", json={
            "thread_id": thread_id,
            "messages": [{"role": "user", "content": "do you know my name?"}]
        })
        assert r3.status_code == 200
        assert "haven't told me your name" in r3.json()["message"]["content"] or "don't know your name" in r3.json()["message"]["content"]
