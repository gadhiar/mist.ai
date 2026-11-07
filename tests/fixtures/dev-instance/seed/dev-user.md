---
type: mist-seed
seed_version: dev-fixture-v1
nodes:
  - id: user
    type: User
    display_name: Dev User
  - id: tech-python
    type: Technology
    display_name: Python
  - id: tech-sqlite
    type: Technology
    display_name: SQLite
  - id: skill-api-design
    type: Skill
    display_name: API Design
  - id: project-seedling-planner
    type: Project
    display_name: Seedling Planner
    description: "An invented garden-planning app, the Dev User's practice project."
  - id: concept-knowledge-graphs
    type: Concept
    display_name: Knowledge Graphs
  - id: goal-ship-planner-beta
    type: Goal
    display_name: Ship the Seedling Planner beta
facts:
  - {subject: user, predicate: USES, object: tech-python}
  - {subject: user, predicate: USES, object: tech-sqlite}
  - {subject: user, predicate: EXPERT_IN, object: skill-api-design}
  - {subject: user, predicate: WORKS_ON, object: project-seedling-planner}
  - {subject: user, predicate: INTERESTED_IN, object: concept-knowledge-graphs}
  - {subject: user, predicate: HAS_GOAL, object: goal-ship-planner-beta}
---

# Dev User

The Dev User is invented for the development seed. No fact here belongs to a real person.

The Dev User writes Python and keeps data in SQLite, and is strongest at API Design. Their
practice project is the Seedling Planner, an invented garden-planning app. They are interested
in Knowledge Graphs, and their current goal is to Ship the Seedling Planner beta.
