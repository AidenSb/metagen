import os, sys, shutil, tempfile
from pathlib import Path
from huggingface_hub import snapshot_download
root=Path("/home/aidens/metagen/tabasco/src/data"); root.mkdir(parents=True, exist_ok=True)
for repo_id in ["carlosinator/tabasco-geom-drugs","carlosinator/tabasco-qm9"]:
    tmpdir=Path(tempfile.mkdtemp())
    local_repo=tmpdir/"repo"
    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=str(local_repo), local_dir_use_symlinks=False)
    for p in local_repo.rglob("*.pt"):
        shutil.copy2(p, root/p.name)
    shutil.rmtree(tmpdir, ignore_errors=True)
print("done")
