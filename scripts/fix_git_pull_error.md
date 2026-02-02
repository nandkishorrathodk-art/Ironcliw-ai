# Fix: "bad object refs/remotes/origin/main" / "did not send all necessary objects"

These errors usually mean your local remote-tracking refs are out of sync with GitHub (e.g. after a force-push or partial fetch).

## Option 1: Refresh remote refs (safest)

Run in your repo root:

```bash
cd /Users/djrussell23/Documents/repos/JARVIS-AI-Agent

# Remove stale remote-tracking ref for main (git will recreate it on fetch)
rm -f .git/refs/remotes/origin/main

# Fetch from origin; use refspec so we only update origin/main
git fetch origin main:refs/remotes/origin/main

# Now pull (or merge) as needed
git pull origin main
```

## Option 2: Full fetch with prune

```bash
cd /Users/djrussell23/Documents/repos/JARVIS-AI-Agent

git fetch origin --prune
git pull origin main
```

## Option 3: If you want your local main to match remote exactly

(Only if you're OK discarding local commits on main that aren’t on the remote.)

```bash
cd /Users/djrussell23/Documents/repos/JARVIS-AI-Agent

git fetch origin
git reset --hard origin/main
```

## Option 4: Re-add the remote (if refs are badly corrupted)

```bash
cd /Users/djrussell23/Documents/repos/JARVIS-AI-Agent

git remote remove origin
git remote add origin https://github.com/drussell23/JARVIS.git
git fetch origin
git branch --set-upstream-to=origin/main main
git pull
```

Start with **Option 1**; use 3 or 4 only if you intend to reset to the remote or re-establish the remote.
