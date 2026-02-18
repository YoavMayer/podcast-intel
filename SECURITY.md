# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in podcast-intel, please report it
responsibly.

**Do not open a public GitHub issue for security vulnerabilities.**

Instead, please email the maintainers directly or use GitHub's
[private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability).

We will acknowledge your report within 48 hours and provide a timeline for a fix.

## Scope

podcast-intel processes audio files and podcast metadata. Key areas of concern:

- **File handling**: Malicious audio files or RSS feeds
- **SQL injection**: Database operations via SQLite
- **Path traversal**: File download and storage paths
- **Dependency vulnerabilities**: Third-party package issues

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.2.x   | Yes       |
| 0.1.x   | No        |
