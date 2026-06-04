# Local Neo4j With Docker

This project can run Neo4j locally through Docker instead of Neo4j Aura.

## Start Neo4j

```powershell
Copy-Item .env.neo4j.example .env.neo4j
# Edit .env.neo4j and set a local password.
docker compose -f docker-compose.neo4j.yml up -d --build
```

Neo4j will be available only on localhost:

```text
Browser: http://127.0.0.1:7474
Bolt:    bolt://127.0.0.1:7687
User:    neo4j
```

Then configure the application `.env`:

```env
NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<the password from .env.neo4j after neo4j/>
NEO4J_DATABASE=neo4j
```

## Load Criteria

After criteria JSON exists in `outputs/extracted_json`, load it:

```powershell
python -m src.main_load_graph --json_dir outputs/extracted_json --clear
```

## Stop Neo4j

```powershell
docker compose -f docker-compose.neo4j.yml down
```

To delete the local graph data too:

```powershell
docker compose -f docker-compose.neo4j.yml down -v
```
