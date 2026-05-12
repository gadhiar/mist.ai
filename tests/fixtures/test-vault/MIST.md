# MIST Vault Conventions

This file is auto-loaded at the start of every turn as runtime context.
It establishes vault structure, note types, and write conventions.

## Folder Structure
mist-memory/
├── MIST.md                  # this file
├── sessions/                # one file per session
├── decisions/               # vault-scoped DEC-NNN
├── identity/mist.md         # MIST self-model (Bucket 1)
├── users/<user-id>.md       # per-user fact sheet (Bucket 1)
└── meta/
    ├── schema.md
    └── changelog.md

## Note Types (Frontmatter)
- mist-session   -- sessions/<YYYY-MM-DD>-<slug>.md
- mist-identity  -- identity/mist.md
- mist-user      -- users/<user-id>.md
- mist-decision  -- decisions/DEC-NNN.md

## Three-Bucket Write Patterns
Bucket 1 -- Mechanical state mirror (C-pattern auto-render):
  identity/mist.md, users/<user-id>.md
  Re-rendered from graph 1-hop snapshot when relevant edges change.

Bucket 2 -- Rebuild substrate (hybrid):
  sessions/<date>-<slug>.md
  Conditional per-turn append for DERIVED_FROM contract +
  end-of-session synthesis.

Bucket 3 -- Curated knowledge (trigger + approval):
  Future ADRs/research/troubleshooting. NOT YET IMPLEMENTED.

## Authoring Invariants
- Vault NEVER stores inferred beliefs. Only events and user-approved content.
- Session notes contain verbatim user utterances and MIST responses --
  NOT synthetic graph-retrieval prose presented as fact.
- User edits to vault are authoritative; on conflict, vault wins.
