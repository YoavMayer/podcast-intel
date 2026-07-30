# Releasing podcast-intel

Releases are published to PyPI by `.github/workflows/release.yml` using
**PyPI Trusted Publishing (OIDC)**. GitHub mints a short-lived identity for the
publish job and PyPI exchanges it for a one-shot upload token. There is no API
token on any machine, none in repository secrets, and nothing to rotate.

You do the pypi.org form **once**. After that, a release is a tag push.

---

## Step 1 (one time, ~2 minutes) -- register the trusted publisher on PyPI

The project does not exist on PyPI yet, so you register a **pending** publisher.
PyPI creates the project on the first successful upload and the pending
publisher becomes the project's normal publisher automatically.

1. Sign in at <https://pypi.org> (2FA required for publishing).
2. Go to <https://pypi.org/manage/account/publishing/>.
3. Under **Add a new pending publisher**, pick the **GitHub** tab.
4. Fill in exactly these values -- they are case-insensitive but otherwise
   matched literally:

   | Field | Value |
   |-------|-------|
   | PyPI Project Name | `podcast-intel` |
   | Owner | `YoavMayer` |
   | Repository name | `podcast-intel` |
   | Workflow name | `release.yml` |
   | Environment name | `pypi` |

5. Press **Add**.

`Workflow name` is the **filename**, not the `name:` line inside it. It is
`release.yml`, not `Release`.

`Environment name` is the `environment: name:` of the `publish-pypi` job in
`.github/workflows/release.yml`. GitHub creates the environment itself the
first time the job runs; you do not need to pre-create it. If you later want a
manual approval before every upload, add yourself as a required reviewer on the
`pypi` environment in **Settings -> Environments** -- nothing in the workflow
changes.

### Optional, but do it: the same form on TestPyPI

TestPyPI is a completely separate service with its own accounts. To use the dry
run in Step 2, repeat Step 1 at
<https://test.pypi.org/manage/account/publishing/> with identical values except:

| Field | Value |
|-------|-------|
| Environment name | `testpypi` |

### If a field is wrong

Nothing warns you at form-submission time. The mismatch only surfaces during
the publish job, and the message is not obvious:

- **Any of project name / owner / repository / workflow filename / environment
  is wrong** -> the upload fails with HTTP `403` and a
  `invalid-publisher: valid token, but no corresponding publisher` body. The
  action prints the claims it actually presented (`repository`,
  `workflow_ref`, `environment`). Compare those printed claims to the form,
  field by field; the wrong one will be visibly different. Fix it by deleting
  the publisher on PyPI and adding it again -- publishers are not editable.
- **Environment left blank on the form while the job has one (or the reverse)**
  -> the same `403 invalid-publisher`. Blank on the form does not mean "any
  environment"; it means "no environment claim", which does not match a job
  that has one.
- **`permissions: id-token: write` removed from the publish job** -> it fails
  earlier, before contacting PyPI, complaining that no OIDC token could be
  retrieved. That one is a workflow bug, not a form problem.
- **Registered on pypi.org but the run targeted TestPyPI** (or the reverse) ->
  also `403 invalid-publisher`, because the other service has never heard of
  this repository. Check which URL the failing step used.

---

## Step 2 (one time, recommended) -- dry run against TestPyPI

Do this before the first real tag, so the first production publish is not the
first time the workflow has ever executed.

1. GitHub -> **Actions** -> **Release** -> **Run workflow**.
2. Branch: the branch you are releasing from. Target: **`testpypi`** (default).
3. Run it. The job builds the distributions, runs the full 3.10/3.11/3.12 CI
   matrix first, verifies the preset YAMLs are inside the wheel, installs the
   wheel into a clean virtualenv, and uploads to TestPyPI.

A green run proves everything except the pypi.org half of Step 1.
Re-running the dry run on an unchanged version is fine -- `skip-existing: true`
makes a duplicate upload a no-op rather than an error.

---

## Step 3 -- cut a release

1. Make sure `main` has what you want to ship and its CI is green.
2. Bump the version in **both** places -- they must match, the workflow checks:
   - `pyproject.toml` -> `[project] version`
   - `src/podcast_intel/__init__.py` -> `__version__`
3. Add the release section to `CHANGELOG.md` (`## [X.Y.Z] - YYYY-MM-DD`).
4. Commit and push those changes.
5. Tag and push the tag:

   ```bash
   git checkout main
   git pull
   git tag -a v0.4.0 -m "Release v0.4.0"
   git push origin v0.4.0
   ```

The tag push starts the workflow: CI matrix -> build -> publish to PyPI.

The tag must be `v` + the exact version in `pyproject.toml`. If they disagree
the build job stops before anything is uploaded and tells you both values.
Delete the bad tag (`git tag -d v0.4.0 && git push origin :refs/tags/v0.4.0`),
fix, tag again.

**A version can be uploaded to PyPI exactly once.** If a release is broken you
cannot overwrite it -- yank it on PyPI and ship the fix as a new version.

---

## Step 4 -- after the first successful release

Flip the README install instructions. `README.md` currently tells people to
install from git, and carries the replacement wording in an HTML comment marked
`PENDING FIRST PYPI RELEASE`. Once <https://pypi.org/project/podcast-intel/>
actually resolves, swap the git-install block for that wording and delete the
comment.

Do not do this before the release lands. The name is unclaimed until the first
upload succeeds, and `pip install podcast-intel` would 404.

---

## Maintenance

`pypa/gh-action-pypi-publish` is pinned by commit SHA rather than by tag,
because it is the only step that can mint an upload token. To bump it, take the
commit SHA of the new release from
<https://github.com/pypa/gh-action-pypi-publish/releases> and update both
occurrences in `.github/workflows/release.yml`, keeping the `# v<version>`
comment accurate.
