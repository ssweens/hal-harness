"""
HAL harness wrapper for pi (AI coding assistant).

Wraps pi CLI for use in hal-eval benchmarks.
Supports:
  - GAIA (Q&A with code/tool capabilities)
  - swebench_verified_mini (code fixing)
  - Other code-based benchmarks
"""

import subprocess
import os
import sys
import re
import tempfile
import shutil
from typing import Dict, Any, Optional


def run(input: dict[str, dict], **kwargs) -> dict[str, str]:
    """
    Run pi on a single benchmark task.

    Args:
        input: {task_id: {task_data}} from benchmark
        **kwargs: Agent configuration
            - model_name: (required) model pattern (e.g., "gpt-4o", "claude-opus-4-5")
            - provider: (optional) provider override (anthropic, google, openai, etc.)
            - thinking: (optional) thinking level (off, minimal, low, medium, high, xhigh)
            - tools: (optional) comma-separated tools (read,bash,edit,write,grep,find,ls)
            - system_prompt: (optional) custom system prompt

    Returns:
        {task_id: output_string} - pi's response or error message
    """
    assert "model_name" in kwargs, "model_name is required"
    assert len(input) == 1, "input must contain only one task"

    task_id, task = list(input.items())[0]
    results = {}

    try:
        # Check if this is a swebench task (has repo info)
        if "repo" in task and "base_commit" in task:
            return _run_swebench_task(task_id, task, kwargs)
        
        # Standard task (GAIA, etc.)
        prompt = _extract_prompt(task)
        if not prompt:
            results[task_id] = "ERROR: Could not extract task prompt from input"
            return results

        cmd = _build_pi_command(prompt, kwargs)
        pi_output = _run_pi(cmd)
        pi_output = _extract_diff(pi_output)
        results[task_id] = pi_output

    except Exception as e:
        results[task_id] = f"ERROR: {str(e)}"
        print(f"[pi_agent] Error on task {task_id}: {e}", file=sys.stderr)

    return results


def _run_swebench_task(task_id: str, task: Dict[str, Any], kwargs: Dict[str, Any]) -> Dict[str, str]:
    """
    Run a swebench task with repo setup.
    
    1. Clone the repo
    2. Check out base commit
    3. Run pi in the repo
    4. Extract git diff
    """
    repo = task.get("repo", "")
    base_commit = task.get("base_commit", "")
    problem_statement = task.get("problem_statement", "")
    
    if not repo or not base_commit:
        return {task_id: "ERROR: Missing repo or base_commit in task data"}
    
    # Create temp directory for the repo
    repo_dir = tempfile.mkdtemp(prefix="pi_swebench_")
    
    try:
        # Clone the repo
        clone_url = f"https://github.com/{repo}.git"
        print(f"[pi_agent] Cloning {repo}...", file=sys.stderr)
        
        clone_result = subprocess.run(
            ["git", "clone", "--depth", "1", clone_url, repo_dir],
            capture_output=True,
            text=True,
            timeout=300,  # 5 min timeout for clone
        )
        
        if clone_result.returncode != 0:
            # Try shallow clone with specific commit
            clone_result = subprocess.run(
                ["git", "clone", "--depth", "100", clone_url, repo_dir],
                capture_output=True,
                text=True,
                timeout=300,
            )
        
        if clone_result.returncode != 0:
            return {task_id: f"ERROR: Failed to clone repo: {clone_result.stderr}"}
        
        # Fetch the specific commit if needed
        subprocess.run(
            ["git", "fetch", "origin", base_commit],
            cwd=repo_dir,
            capture_output=True,
            timeout=60,
        )
        
        # Check out the base commit
        checkout_result = subprocess.run(
            ["git", "checkout", base_commit],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=60,
        )
        
        if checkout_result.returncode != 0:
            return {task_id: f"ERROR: Failed to checkout base commit: {checkout_result.stderr}"}
        
        # Build prompt with repo context
        prompt = f"""You are working in a git repository at {repo_dir}.

The repository is: {repo}
Base commit: {base_commit}

Problem to solve:
{problem_statement}

Instructions:
1. Explore the repository to understand the codebase
2. Identify the files that need to be modified
3. Make the minimal necessary changes to fix the issue
4. If tests need to be updated to match your fix, update them too
5. Test your changes if possible
6. When done, output ONLY the git diff for your changes

Do NOT commit your changes. Just make the edits and output the diff.

Your final message must be the git diff output starting with 'diff --git'."""

        # Build pi command
        cmd = _build_pi_command(prompt, kwargs)
        
        # Run pi in the repo directory
        print(f"[pi_agent] Running pi in {repo_dir}...", file=sys.stderr)
        pi_output = _run_pi(cmd, cwd=repo_dir)
        
        # If pi didn't output a diff, try to get it from git
        if not pi_output.startswith("diff --git"):
            print(f"[pi_agent] No diff in output, extracting from git...", file=sys.stderr)
            diff_result = subprocess.run(
                ["git", "diff"],
                cwd=repo_dir,
                capture_output=True,
                text=True,
            )
            if diff_result.returncode == 0 and diff_result.stdout.strip():
                pi_output = diff_result.stdout.strip()
        
        # Extract diff if embedded in markdown
        pi_output = _extract_diff(pi_output)
        
        return {task_id: pi_output}
        
    finally:
        # Cleanup repo directory
        try:
            shutil.rmtree(repo_dir, ignore_errors=True)
        except Exception as e:
            print(f"[pi_agent] Warning: Failed to cleanup {repo_dir}: {e}", file=sys.stderr)


def _extract_prompt(task: Dict[str, Any]) -> Optional[str]:
    """Extract the prompt/problem statement from task data (benchmark-agnostic)."""
    
    # GAIA: task["Question"]
    if "Question" in task:
        return task["Question"]
    
    # swebench_verified_mini: task["problem_statement"]
    if "problem_statement" in task:
        return task["problem_statement"]
    
    # Generic: description, prompt, instruction, text
    for key in ["description", "prompt", "instruction", "text", "content"]:
        if key in task and task[key]:
            return task[key]
    
    return None


def _build_pi_command(prompt: str, kwargs: Dict[str, Any]) -> list:
    """Build the pi CLI command."""
    
    # Use full path to pi (subprocess may not have ~/.bun/bin in PATH)
    pi_path = os.path.expanduser("~/.bun/bin/pi")
    if not os.path.exists(pi_path):
        # Fallback to finding pi in PATH
        pi_path = "pi"
    
    cmd = [pi_path, "-p"]  # -p = non-interactive print mode
    
    # Model (required)
    model_name = kwargs["model_name"]
    
    # If provider specified separately, prepend it
    provider = kwargs.get("provider", None)
    if provider and "/" not in model_name:
        model_name = f"{provider}/{model_name}"
    
    cmd.extend(["--model", model_name])
    
    # Thinking (for reasoning models)
    if "thinking" in kwargs:
        cmd.extend(["--thinking", str(kwargs["thinking"])])
    
    # Tools to enable
    if "tools" in kwargs:
        cmd.extend(["--tools", str(kwargs["tools"])])
    
    # Custom system prompt
    if "system_prompt" in kwargs:
        cmd.extend(["--system-prompt", str(kwargs["system_prompt"])])
    
    # Session handling: don't persist sessions during evaluation
    cmd.append("--no-session")
    
    # Output as text (default)
    cmd.extend(["--mode", "text"])
    
    # Append the prompt at the end
    cmd.append(prompt)
    
    return cmd


def _run_pi(cmd: list, timeout: int = 1800, cwd: Optional[str] = None) -> str:
    """
    Execute pi command and capture output.
    
    Args:
        cmd: Command list to execute
        timeout: Max seconds to wait for pi to complete (default 30 min)
        cwd: Working directory for pi
        
    Returns:
        stdout from pi (or error message)
        
    Raises:
        RuntimeError if pi fails
    """
    try:
        print(f"[pi_agent] Running: {' '.join(cmd[:5])}...", file=sys.stderr)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        
        # Log both stdout and stderr for debugging
        if result.stderr:
            print(f"[pi_agent] stderr: {result.stderr[:500]}...", file=sys.stderr)
        
        if result.returncode == 0:
            output = result.stdout.strip()
            if not output and result.stderr:
                # Some versions might output to stderr
                print(f"[pi_agent] Empty stdout, checking stderr", file=sys.stderr)
                output = result.stderr.strip()
            return output
        else:
            # pi exited with error
            error_msg = result.stderr.strip() or result.stdout.strip() or "Unknown error"
            raise RuntimeError(f"pi exited with code {result.returncode}: {error_msg}")
            
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"pi timed out after {timeout} seconds")
    except FileNotFoundError:
        raise RuntimeError(
            "pi CLI not found. Install with: npm install -g @mariozechner/pi-coding-agent"
        )


def _extract_diff(output: str) -> str:
    """Extract git diff from output, handling various formats."""
    if not output:
        return output
    
    # If output already starts with diff --git, return as-is
    if output.startswith("diff --git"):
        return output
    
    # Try to extract from ```diff ... ``` code block
    diff_match = re.search(r'```diff\n(.*?)\n```', output, re.DOTALL)
    if diff_match:
        return diff_match.group(1)
    
    # Try to extract from ``` ... ``` code block that contains diff --git
    code_match = re.search(r'```\n?(diff --git.*?)\n?```', output, re.DOTALL)
    if code_match:
        return code_match.group(1)
    
    # No diff found, return original output
    return output
