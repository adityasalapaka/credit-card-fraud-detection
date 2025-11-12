import os
import re
import unicodedata
import matplotlib.pyplot as plt
from datetime import datetime


def sanitize_filename(title: str, max_length: int = 100) -> str:
    filename = unicodedata.normalize("NFKD", title)
    filename = re.sub(r"[^\w\s-]", "_", filename)
    filename = re.sub(r"[-\s]+", "_", filename).strip("_")
    filename = filename[:max_length] or "untitled"
    return filename.lower()


def enable_autosave(
    folder: str,
    fmt: str = "png",
    dpi: int = 300,
    env_var: str = "AILAB_OBSIDIAN_RESULTS",
    base_results_dir: str = "../results",
    quiet: bool = False,
) -> None:
    """
    Auto-saves matplotlib figures to:
      - repo:        {base_results_dir}/{folder}/
      - obsidian:    {$AILAB_OBSIDIAN_RESULTS}/{folder}/  (if env var is set)

    Nothing Obsidian-related is committed to the repo.
    """
    # Resolve repo dir as ../results/<folder>/ by default
    repo_dir = os.path.abspath(os.path.join(base_results_dir, folder))
    save_dirs = [repo_dir]

    # If an Obsidian vault is provided, mirror the same subfolder there
    vault_root = os.environ.get(env_var)
    vault_dir = None
    if vault_root:
        vault_dir = os.path.abspath(os.path.join(vault_root, folder))
        save_dirs.append(vault_dir)

    for d in save_dirs:
        os.makedirs(d, exist_ok=True)

    _old_show = plt.show
    _saved_ids = set()

    def autosave_show(*args: tuple, **kwargs: dict) -> None:
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            if id(fig) in _saved_ids:
                continue

            # title -> filename
            if getattr(fig, "_suptitle", None) and fig._suptitle.get_text():  # type: ignore[attr-defined]
                title = fig._suptitle.get_text().strip()  # type: ignore[attr-defined]
            elif fig.axes and fig.axes[0].get_title():
                title = fig.axes[0].get_title().strip()
            else:
                title = f"figure_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            fname = f"{sanitize_filename(title)}.{fmt}"

            for d in save_dirs:
                path = os.path.join(d, fname)
                fig.savefig(path, bbox_inches="tight", dpi=dpi)
                if not quiet:
                    if vault_dir and d == vault_dir:
                        print(f"✅ Auto-saved (vault): {path}")
                    else:
                        print(f"✅ Auto-saved (repo):  {path}")

            _saved_ids.add(id(fig))

        _old_show(*args, **kwargs)

    plt.show = autosave_show
    if not quiet:
        print("✅ Auto-save enabled — destinations:")
        print("   📁 repo:", repo_dir)
        if vault_dir:
            print("   📁 vault: (env)", vault_dir)
        else:
            print(f"   ⚠️ vault: not set (set {env_var} to enable)")
