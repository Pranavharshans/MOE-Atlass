# Security policy

## Supported versions

MoEAtlas is pre-1.0; only the latest commit on `feat/prd-completion` (and,
after release, the latest tagged release) receives security fixes.

## Reporting a vulnerability

Report suspected vulnerabilities privately to the maintainers via GitHub
Security Advisories ("Report a vulnerability" on the repository's Security
tab). Do not open a public issue for an unreported vulnerability.

Include: affected paths, a minimal reproduction, and any local workspace
files involved. You will get an acknowledgment, a assessment of severity,
and a fix or mitigation timeline.

## Scope notes

- MoEAtlas is a **local-first** tool: the CLI binds servers to loopback by
  default and remote binding requires an explicit opt-in flag. Reports that
  require a user to deliberately disable local-only protections count as
  hardening suggestions unless they bypass the explicit consent flow.
- The package never downloads models, tokenizers, datasets, or telemetry.
  Any behavior that does so silently is in scope and treated as severe.
- Plugin code loaded through entry points runs with the installing user's
  privileges; trust decisions belong to the `adapters list` policy surface
  and the installer. Vulnerabilities in third-party plugins belong to
  those plugins.

## Telemetry

There is none. No usage data leaves the machine through this package.
