"""Textos da documentação OpenAPI (Swagger UI em ``/docs``)."""

DESCRIPTION = """
Recomenda jogos da Steam com base no que cada usuário comprou e em quanto tempo jogou.

**Fluxo típico**

1. Cadastre um usuário em `POST /users` (opcionalmente já com jogos).
2. Busque jogos em `GET /games?search=...` para descobrir os IDs.
3. Atualize as preferências em `PUT` ou `PATCH /users/{user_id}/preferences`
   (compra, horas jogadas e "não gostei").
4. Peça recomendações em `GET /users/{user_id}/recommendations`.

**Erros** seguem sempre o mesmo formato:
`{"error": {"code": "not_found", "message": "...", "details": null}}`.
Em erros de validação (422), `details` lista o campo e o problema.

Dados: *Steam Video Games* (Kaggle), na versão processada do repositório caserec.
"""

TAGS = [
    {"name": "health", "description": "Status da API e do modelo carregado."},
    {
        "name": "users",
        "description": "Cadastro e consulta de usuários, com o histórico de jogos.",
    },
    {
        "name": "preferences",
        "description": (
            "Atualização do histórico do usuário: jogos comprados, horas jogadas e jogos "
            'marcados como "não gostei". `PUT` substitui tudo; `PATCH` altera só o que for '
            "enviado."
        ),
    },
    {
        "name": "recommendations",
        "description": (
            "Recomendações personalizadas. Jogos que o usuário já tem ou marcou como "
            '"não gostei" nunca são recomendados.'
        ),
    },
    {
        "name": "games",
        "description": "Catálogo de jogos: cadastro, busca, detalhe e jogos parecidos.",
    },
    {"name": "model", "description": "Versão do modelo em uso e métricas do último treino."},
]
