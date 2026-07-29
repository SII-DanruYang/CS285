# Security Policy

## Scope

This repository contains educational reinforcement-learning code and is not
intended for production deployment or use as a public network service.

## Reporting a Security Issue

Please report suspected credential exposure, unsafe scripts, dependency
vulnerabilities, or unintended disclosure of private paths through a private
message to the repository owner rather than opening a public issue containing
sensitive information.

## Pickle Files

Several original CS285 workflows use Python pickle files for expert policies
and datasets. Only load pickle files obtained from trusted sources. Loading an
untrusted pickle file may execute arbitrary code.

## Dependencies

Some dependencies are pinned to older versions for reproducibility. Security
updates should be reviewed together with compatibility requirements before
changing course environments.

## Sensitive Information

The repository should not contain:

- API keys or access tokens
- Passwords or private keys
- `.env` files
- Private datasets
- Internal server addresses or absolute user paths
- Training logs containing personal or infrastructure information
