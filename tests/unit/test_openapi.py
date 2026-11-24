from fastapi.testclient import TestClient

from app.main import create_app


def spec():
    return TestClient(create_app()).get("/openapi.json").json()


def test_tags_documentadas():
    doc = spec()
    declared = {t["name"] for t in doc["tags"]}
    used = {tag for ops in doc["paths"].values() for op in ops.values() for tag in op["tags"]}

    assert used <= declared
    assert all(t["description"] for t in doc["tags"])
    assert "Fluxo típico" in doc["info"]["description"]


def test_toda_rota_tem_resumo_descricao_e_erros():
    for path, ops in spec()["paths"].items():
        for method, op in ops.items():
            assert op.get("summary"), f"{method} {path} sem summary"
            assert op.get("description"), f"{method} {path} sem descrição"
            if "{" in path or method in {"post", "put", "patch"}:
                assert "422" in op["responses"], f"{method} {path} sem 422"


def test_erros_usam_o_envelope_padrao():
    doc = spec()
    response = doc["paths"]["/users/{user_id}"]["get"]["responses"]["404"]

    ref = response["content"]["application/json"]["schema"]["$ref"]
    assert ref.endswith("/ErrorResponse")
    assert "error" in doc["components"]["schemas"]["ErrorResponse"]["properties"]
