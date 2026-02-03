# Neo4j Browser Queries for MIST

## Visualization Queries

### 1. Show ONLY the Knowledge Graph (Entities & Relationships)
**Use this to see the actual extracted knowledge without provenance clutter**

```cypher
MATCH (n:__Entity__)
OPTIONAL MATCH (n)-[r]->(m:__Entity__)
RETURN n, r, m
```

This shows:
- All entities (User, Python, MIST, Sarah, etc.)
- Relationships between entities (USES, WORKS_ON, etc.)
- Excludes: Utterance and ConversationEvent nodes

---

### 2. Show User's Knowledge Profile
**What does the User know/use/prefer?**

```cypher
MATCH (user:__Entity__ {id: "User"})-[r]->(entity:__Entity__)
RETURN user, r, entity
```

Shows all relationships FROM User to other entities.

---

### 3. Show User's Skills with Proficiency Levels
**Technologies/skills User knows, organized by relationship type**

```cypher
MATCH (user:__Entity__ {id: "User"})-[r]->(tech:__Entity__)
WHERE r.type IN ['EXPERT_IN', 'PROFICIENT_IN', 'BEGINNER_IN', 'LEARNING']
RETURN user, r, tech
```

---

### 4. Show User's Current Projects
**What is User working on?**

```cypher
MATCH (user:__Entity__ {id: "User"})-[r:WORKS_ON]->(project:__Entity__)
RETURN user, r, project
```

---

### 5. Show People and Relationships
**Who does User know and how?**

```cypher
MATCH (user:__Entity__ {id: "User"})-[r]->(person:__Entity__)
WHERE person.entity_type = 'Person'
RETURN user, r, person
```

---

### 6. Show Technologies and Tools
**Filter to only see tech stack**

```cypher
MATCH (n:__Entity__)
WHERE n.entity_type IN ['Technology', 'Tool']
OPTIONAL MATCH (n)-[r]->(m:__Entity__)
WHERE m.entity_type IN ['Technology', 'Tool']
RETURN n, r, m
```

---

### 7. Show Full Graph with Provenance
**If you DO want to see where knowledge came from**

```cypher
MATCH (c:ConversationEvent)<-[:PART_OF]-(u:Utterance)-[:HAS_ENTITY]->(e:__Entity__)
RETURN c, u, e
LIMIT 50
```

---

### 8. Show Properties on Relationships
**See temporal/contextual properties**

```cypher
MATCH (user:__Entity__ {id: "User"})-[r]->(entity:__Entity__)
WHERE r.duration IS NOT NULL OR r.when IS NOT NULL OR r.context IS NOT NULL
RETURN user.id AS user,
       type(r) AS relationship,
       entity.id AS target,
       r.duration AS duration,
       r.when AS when,
       r.context AS context
```

---

## Analysis Queries

### 9. Count Entities by Type
**What types of knowledge do we have?**

```cypher
MATCH (e:__Entity__)
RETURN e.entity_type AS type, count(*) AS count
ORDER BY count DESC
```

---

### 10. Most Connected Entities
**What are the key nodes in the graph?**

```cypher
MATCH (n:__Entity__)
RETURN n.id AS entity,
       n.entity_type AS type,
       size((n)--()) AS connections
ORDER BY connections DESC
LIMIT 10
```

---

### 11. User's Tech Stack Summary
**List all technologies User uses**

```cypher
MATCH (user:__Entity__ {id: "User"})-[r]->(tech:__Entity__)
WHERE tech.entity_type IN ['Technology', 'Tool']
  AND type(r) IN ['USES', 'EXPERT_IN', 'PROFICIENT_IN', 'LEARNING']
RETURN tech.id AS technology,
       type(r) AS relationship,
       r.proficiency AS proficiency
ORDER BY technology
```

---

### 12. Timeline of Learning
**What has User learned/used over time?**

```cypher
MATCH (user:__Entity__ {id: "User"})-[r]->(tech:__Entity__)
WHERE r.duration IS NOT NULL OR r.when IS NOT NULL
RETURN tech.id AS technology,
       type(r) AS relationship,
       r.duration AS duration,
       r.when AS when,
       r.ontology_version AS version
ORDER BY r.created_at
```

---

### 13. Knowledge Gaps (Negations)
**What does User NOT use or avoid?**

```cypher
MATCH (user:__Entity__ {id: "User"})-[r]->(entity:__Entity__)
WHERE type(r) IN ['DOES_NOT_USE', 'AVOIDS', 'HAS_NOT_WORKED_WITH']
RETURN entity.id AS entity,
       entity.entity_type AS type,
       type(r) AS relationship
```

---

### 14. Entity Provenance
**Where did we learn about a specific entity?**

```cypher
MATCH (u:Utterance)-[:HAS_ENTITY]->(e:__Entity__ {id: "Python"})
RETURN u.text AS utterance,
       u.timestamp AS when,
       e.id AS entity
ORDER BY u.timestamp
```

---

## Neo4j Browser Configuration

### Recommended Style Settings

In Neo4j Browser, go to bottom-left gear icon → Configure

**Node Captions:**
```
{id}
```

**Node Color by Label:**
- `__Entity__` with `entity_type='Person'` → Blue
- `__Entity__` with `entity_type='Technology'` → Green
- `__Entity__` with `entity_type='Tool'` → Orange
- `__Entity__` with `entity_type='Project'` → Purple
- `__Entity__` with `entity_type='Topic'` → Yellow
- `Utterance` → Gray (if showing provenance)
- `ConversationEvent` → Light Gray (if showing provenance)

**Relationship Captions:**
Show relationship type by default.

---

## Quick Start

**For best visualization, start with Query #1:**

```cypher
MATCH (n:__Entity__)
OPTIONAL MATCH (n)-[r]->(m:__Entity__)
RETURN n, r, m
```

This gives you a clean view of the knowledge graph without the Utterance/ConversationEvent clutter.
