# MIST Vault Conventions

This file is loaded at the start of every turn as runtime context. It sets out the vault's
structure, its note types and the rules for writing to it.

This vault belongs to a development instance on a synthetic seed. Its user is the Dev User, an
invented person, and every note in it is fixture content.

## Folder Structure

```
MIST.md                  this file
identity/mist.md         MIST's self-model, rendered from the graph
users/<user-id>.md       one fact sheet per user, rendered from the graph
sessions/                one synthesis note per conversation, sessions/<YYYY-MM-DD>-<slug>.md
decisions/               decision notes, decisions/DEC-NNN.md
```

## Note Types (Frontmatter `type`)

- `mist-session`: sessions/<YYYY-MM-DD>-<slug>.md
- `mist-identity`: identity/mist.md
- `mist-user`: users/<user-id>.md
- `mist-decision`: decisions/DEC-NNN.md

## Write Patterns

- Mechanical mirror: identity/mist.md and users/<user-id>.md are re-rendered from the graph when
  their edges change.
- Session synthesis: each session note is written whole at the end of its conversation, from the
  session's synthesis. Turns are not appended one by one.

## Authoring Rules

- The vault stores events and approved content, never inferred beliefs.
- A user's edit to a note is authoritative: a note marked `user` or `user-edit` is not
  overwritten.
