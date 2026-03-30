import datetime
import json
import logging
import os
import re
import subprocess

from ocs_ci.framework import config as ocsci_config
from ocs_ci.ocs.ui.llm_tools.locator_fallback import get_session_cache_path

logger = logging.getLogger(__name__)

_SQUAD_MARK_RE = re.compile(r"@pytest\.mark\.(\w+_squad)")


def _detect_base_branch():
    """
    Detect the base branch to open PRs against.

    Checks ``config.UI_SELENIUM.pr_base_branch`` first; falls back to the
    git upstream of the current branch, and finally to ``master`` when no
    upstream is configured.

    Returns:
        str: Branch name (e.g. ``"master"`` or ``"release-4.17"``)

    """
    override = ocsci_config.UI_SELENIUM.get("pr_base_branch", "")
    if override:
        return override
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip().removeprefix("origin/")
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or "master"


def _is_gh_available():
    """
    Check whether the ``gh`` CLI is installed and authenticated.

    Returns:
        bool: True if ``gh auth status`` exits successfully, False otherwise

    """
    result = subprocess.run(["gh", "auth", "status"], capture_output=True)
    return result.returncode == 0


def _find_locator_source(old_selector, worktree_path):
    """
    Find the source file and line where a locator selector string is defined.

    Searches ``ocs_ci/ocs/ui/`` recursively for an exact match of
    ``old_selector``.  Returns ``None`` when the selector is absent or
    appears more than once (ambiguous match).

    Args:
        old_selector (str): Selector string to locate (e.g. a CSS selector or XPath)
        worktree_path (str): Absolute path to the git worktree root

    Returns:
        tuple: ``(relative_file_path, line_number)`` on a unique match, or
        ``None`` if not found or ambiguous

    """
    ui_dir = os.path.join(worktree_path, "ocs_ci", "ocs", "ui")
    result = subprocess.run(
        ["grep", "-rn", "--include=*.py", old_selector, ui_dir],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        logger.warning("[AI_FALLBACK] Locator not found in source: %s", old_selector)
        return None
    matches = [line for line in result.stdout.strip().split("\n") if line]
    if len(matches) != 1:
        logger.warning(
            "[AI_FALLBACK] %d source matches for locator '%s', skipping",
            len(matches),
            old_selector,
        )
        return None
    parts = matches[0].split(":", 2)
    rel_path = os.path.relpath(parts[0], worktree_path)
    return (rel_path, int(parts[1]))


def _patch_locator_in_file(
    file_path, old_selector, new_selector, new_by_type, old_by_type
):
    """
    Replace the first occurrence of a selector string inside a Python source file.

    Performs an in-place string substitution of ``old_selector`` with
    ``new_selector``.  Logs a warning when the ``by_type`` changes so that
    reviewers know a manual check may be needed.

    Args:
        file_path (str): Absolute path to the file to patch
        old_selector (str): Selector value to replace
        new_selector (str): Replacement selector value
        new_by_type (str): Selenium ``By`` strategy for the new selector (e.g. ``"xpath"``)
        old_by_type (str): Selenium ``By`` strategy for the old selector

    Returns:
        bool: True if the file was modified, False if the selector was not found

    """
    with open(file_path, "r") as f:
        content = f.read()
    new_content = content.replace(old_selector, new_selector, 1)
    if new_content == content:
        logger.warning("[AI_FALLBACK] Could not replace selector in %s", file_path)
        return False
    if new_by_type != old_by_type:
        logger.warning(
            "[AI_FALLBACK] by_type changed %s → %s in %s — manual review recommended",
            old_by_type,
            new_by_type,
            file_path,
        )
    with open(file_path, "w") as f:
        f.write(new_content)
    return True


def _get_squad_labels_for_test(test_name, worktree_path):
    """
    Extract GitHub squad labels from ``@pytest.mark.<squad>_squad`` decorators.

    Locates the test function definition inside ``tests/``, then walks
    upward through its decorator lines (up to 25 lines) to collect every
    squad mark.

    Args:
        test_name (str): Exact function name of the test (e.g. ``"test_pvc_create"``)
        worktree_path (str): Absolute path to the git worktree root

    Returns:
        list: Label strings in ``"Squad/<Name>"`` format (e.g. ``["Squad/Black"]``),
        or an empty list when the test is not found or has no squad marks

    """
    tests_dir = os.path.join(worktree_path, "tests")
    result = subprocess.run(
        ["grep", "-rn", f"def {test_name}", tests_dir],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return []
    first_match = result.stdout.strip().split("\n")[0]
    parts = first_match.split(":", 2)
    file_path = parts[0]
    def_line = int(parts[1])
    try:
        with open(file_path) as f:
            lines = f.readlines()
    except OSError:
        return []
    labels = []
    for i in range(def_line - 2, max(0, def_line - 25), -1):
        line = lines[i].strip()
        if not line.startswith("@"):
            break
        m = _SQUAD_MARK_RE.search(line)
        if m:
            squad = m.group(1).replace("_squad", "").capitalize()
            labels.append(f"Squad/{squad}")
    return labels


def _build_pr_body(test_name, entries):
    """
    Build the Markdown body for an AI locator update pull request.

    Renders a table with old/new selector pairs and the page URL for each
    updated locator, followed by a footer crediting the auto-generation.

    Args:
        test_name (str): Name of the test whose locators were updated
        entries (list): List of locator-update dicts, each containing
            ``old_selector``, ``new_selector``, ``old_by_type``,
            ``new_by_type``, and optionally ``page_url``

    Returns:
        str: Formatted Markdown string ready to write to a ``.md`` file

    """
    body = "## AI Locator Update\n\n"
    body += f"### `{test_name}`\n\n"
    body += "| | Selector | Type |\n"
    body += "|---|---|---|\n"
    for entry in entries:
        body += f"| **Old** | `{entry['old_selector']}` | {entry['old_by_type']} |\n"
        body += f"| **New** | `{entry['new_selector']}` | {entry['new_by_type']} |\n"
        body += f"\n**Page:** {entry.get('page_url', 'unknown')}\n\n"
    body += "---\n"
    body += "🤖 Auto-generated by ocs-ci AI locator fallback mechanism\n"
    body += "Config: `config.UI_SELENIUM.push_locator_upd_pr`"
    return body


def _create_pr_via_claude(
    worktree_path,
    branch_name,
    base_branch,
    test_name,
    patched_files,
    squad_labels,
    pr_title,
):
    """
    Delegate the git commit, push, and PR creation to a headless ``claude`` subprocess.

    Constructs a natural-language prompt instructing Claude to stage only the
    patched locator files, create a DCO-signed commit, push the branch, and
    open a GitHub PR via ``gh pr create``.  Parses the last
    ``https://github.com/`` URL from Claude's output as the PR URL.

    Args:
        worktree_path (str): Absolute path to the temporary git worktree
        branch_name (str): Name of the already-checked-out feature branch
        base_branch (str): Target base branch for the PR (e.g. ``"master"``)
        test_name (str): Test name used in the commit message and PR title
        patched_files (list): Relative paths of files modified by the locator patch
        squad_labels (list): GitHub label strings to apply (e.g. ``["Squad/Black"]``)
        pr_title (str): Title string for the pull request

    Returns:
        str: PR URL on success (e.g. ``"https://github.com/org/repo/pull/123"``),
        or ``None`` if the subprocess timed out, the CLI was not found, or
        the URL could not be parsed from the output

    """
    files_str = " ".join(patched_files)
    label_flags = "".join(f' --label "{lbl}"' for lbl in squad_labels)
    n = len(patched_files)
    commit_msg = f"fix(locator): {test_name} – update {n} locator(s)"

    prompt = f"""You are a git automation assistant. Your only job is to commit, push, and open a GitHub PR.

Repository directory (all git/gh commands run here): {worktree_path}
Branch already created: {branch_name}

Execute these steps IN ORDER:

Step 1 — Stage ONLY the patched locator files:
  git add {files_str}

Step 2 — Create a signed commit (DCO sign-off, -s flag is mandatory):
  git commit -s -m "{commit_msg}"

  Pre-commit hooks may automatically modify .secrets.baseline or reformat code.
  If the commit exits non-zero because hooks changed files:
    a. Re-stage: git add {files_str} .secrets.baseline
    b. Retry once: git commit -s -m "{commit_msg}"

Step 3 — Push the branch:
  git push -u origin {branch_name}

Step 4 — Create the GitHub PR:
  gh pr create --base {base_branch} --title "{pr_title}" --body-file .ai_pr_body.md{label_flags}

Step 5 — Print the PR URL returned by step 4 as the very last line of your output.

HARD RULES (never violate these):
- Run ALL commands inside: {worktree_path}
- Stage ONLY: {files_str}  (plus .secrets.baseline if and only if hooks modified it)
- Do NOT stage or modify any other file
- The -s flag on git commit is required — do not omit it
"""

    try:
        proc = subprocess.run(
            [
                "claude",
                "-p",
                prompt,
                "--output-format",
                "text",
                "--allowedTools",
                "Bash",
            ],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=worktree_path,
        )
    except subprocess.TimeoutExpired:
        logger.error("[AI_FALLBACK] Claude PR subprocess timed out for %s", test_name)
        return None
    except FileNotFoundError:
        logger.error("[AI_FALLBACK] claude CLI not found, skipping PR creation")
        return None

    if proc.returncode != 0:
        logger.error(
            "[AI_FALLBACK] Claude PR subprocess failed for %s: %s",
            test_name,
            proc.stderr[:500],
        )
        return None

    output = proc.stdout.strip()
    for line in reversed(output.split("\n")):
        line = line.strip()
        if line.startswith("https://github.com/"):
            return line

    logger.warning(
        "[AI_FALLBACK] Could not extract PR URL from Claude response for %s: %s",
        test_name,
        output[:300],
    )
    return None


def _create_test_pr(test_name, entries, base_branch, run_id):
    """
    Create one GitHub PR containing all locator fixes for a single test.

    Spins up a temporary git worktree, applies every locator patch from
    ``entries``, writes the PR body file, then delegates the commit/push/PR
    creation to :func:`_create_pr_via_claude`.  The worktree is removed in
    the ``finally`` block regardless of success or failure.

    Args:
        test_name (str): Name of the test whose locators are being fixed
        entries (list): List of locator-update dicts (same schema as
            :func:`_build_pr_body`)
        base_branch (str): Branch to open the PR against
        run_id (str): Unique run identifier used to name the worktree directory

    Returns:
        str: PR URL on success, or ``None`` if no files were patched or any
        git/GitHub operation failed

    """
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    test_slug = re.sub(r"[^a-z0-9]+", "-", test_name.lower())[:40].strip("-")
    branch_name = f"fix/ai-locator-{test_slug}-{date_str}"
    worktree_path = f"/tmp/locator-pr-{run_id}-{test_slug}"

    try:
        subprocess.run(
            ["git", "worktree", "add", worktree_path, f"origin/{base_branch}"],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "-C", worktree_path, "checkout", "-b", branch_name],
            check=True,
            capture_output=True,
        )

        patched_files = []
        for entry in entries:
            source = _find_locator_source(entry["old_selector"], worktree_path)
            if not source:
                continue
            rel_path, _ = source
            abs_path = os.path.join(worktree_path, rel_path)
            if _patch_locator_in_file(
                abs_path,
                entry["old_selector"],
                entry["new_selector"],
                entry["new_by_type"],
                entry["old_by_type"],
            ):
                patched_files.append(rel_path)

        if not patched_files:
            logger.info(
                "[AI_FALLBACK] No patchable source locators for %s, skipping PR",
                test_name,
            )
            return None

        squad_labels = _get_squad_labels_for_test(test_name, worktree_path)
        pr_title = (
            f"fix(locator): {test_name} "
            f"[{datetime.datetime.now().strftime('%Y-%m-%d')}]"
        )
        pr_body_file = os.path.join(worktree_path, ".ai_pr_body.md")
        with open(pr_body_file, "w") as f:
            f.write(_build_pr_body(test_name, entries))

        return _create_pr_via_claude(
            worktree_path,
            branch_name,
            base_branch,
            test_name,
            patched_files,
            squad_labels,
            pr_title,
        )

    except subprocess.CalledProcessError as e:
        logger.error("[AI_FALLBACK] Git operation failed for %s: %s", test_name, e)
        return None
    finally:
        subprocess.run(
            ["git", "worktree", "remove", worktree_path, "--force"],
            capture_output=True,
        )


def create_locator_pr():
    """
    Creates one GitHub PR per test that had new locators discovered during the session.

    Each PR:
    - Contains the locator fix(es) for a single test
    - Has a signed commit (git commit -s -m)
    - Is labeled with the test's squad (e.g. Squad/Black)
    - Is opened against the branch ocs-ci is currently running on

    Returns a list of created PR URLs (may be empty).
    """
    session_cache_path = get_session_cache_path()
    if not os.path.isfile(session_cache_path):
        logger.info("[AI_FALLBACK] No session cache found, skipping PR creation")
        return []

    try:
        with open(session_cache_path) as f:
            session_data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error("[AI_FALLBACK] Cannot read session cache: %s", e)
        return []

    new_entries = {
        k: v
        for k, v in session_data.items()
        if v.get("new_selector") and v["new_selector"] != v.get("old_selector")
    }
    if not new_entries:
        logger.info("[AI_FALLBACK] No new locators to PR, skipping")
        return []

    if not _is_gh_available():
        logger.warning("[AI_FALLBACK] gh CLI not authenticated, skipping PR creation")
        return []

    base_branch = _detect_base_branch()
    run_id = ocsci_config.RUN.get("run_id", "unknown")

    by_test = {}
    for entry in new_entries.values():
        test = entry.get("test_name", "unknown_test")
        by_test.setdefault(test, []).append(entry)

    pr_urls = []
    for test_name, entries in by_test.items():
        logger.info("[AI_FALLBACK] Creating PR for test: %s", test_name)
        pr_url = _create_test_pr(test_name, entries, base_branch, run_id)
        if pr_url:
            pr_urls.append(pr_url)
            logger.info("[AI_FALLBACK] PR created: %s", pr_url)
    return pr_urls
