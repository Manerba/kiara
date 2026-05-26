"""KIara Admin - Backup & Restore API"""

from fastapi import APIRouter
from pydantic import BaseModel

backup_router = APIRouter(prefix="/api/admin/backup", tags=["backup"])

_backup_status = {"running": False, "result": None, "error": None}


class BackupCreateRequest(BaseModel):
    """Request zum Erstellen eines neuen Backups."""

    include_vectordb_data: bool = False


class BackupRestoreRequest(BaseModel):
    """Request zum Wiederherstellen eines Backups."""

    filename: str
    force: bool = False


# --- Migriert zu Admin-Registry: backup.list (Sprint 08) ---
# @backup_router.get("/list")
# async def list_backups(user=Depends(get_admin_user)):
#     cfg = load_config()
#     backup_dir = get_backup_dir(cfg)
#     backups = []
#     bd = Path(backup_dir)
#     if bd.exists():
#         for f in sorted(bd.glob("*.tar.gz"), key=lambda x: x.stat().st_mtime, reverse=True):
#             stat = f.stat()
#             backups.append({
#                 "filename": f.name,
#                 "size_bytes": stat.st_size,
#                 "created_at": __import__('datetime').datetime.fromtimestamp(stat.st_mtime).isoformat(),
#             })
#     return {"backups": backups}

# --- Migriert zu Admin-Registry: backup.create (Sprint 08) ---
# @backup_router.post("/create")
# async def create_backup_endpoint(req: BackupCreateRequest, user=Depends(get_admin_user)):
#     if _backup_status["running"]:
#         raise HTTPException(status_code=409, detail="Backup laeuft bereits")
#     _backup_status["running"] = True
#     _backup_status["result"] = None
#     _backup_status["error"] = None
#     cfg = load_config()
#     def run_backup():
#         try:
#             from kiara.backup import create_backup
#             result_path = create_backup(cfg, include_vectordb_data=req.include_vectordb_data)
#             stat = Path(result_path).stat()
#             _backup_status["result"] = {
#                 "filename": Path(result_path).name,
#                 "size_bytes": stat.st_size,
#             }
#         except Exception as exc:
#             _backup_status["error"] = str(exc)
#         finally:
#             _backup_status["running"] = False
#     threading.Thread(target=run_backup, daemon=True).start()
#     return {"message": "Backup gestartet"}

# --- Migriert zu Admin-Registry: backup.status (Sprint 08) ---
# @backup_router.get("/status")
# async def backup_status(user=Depends(get_admin_user)):
#     return {
#         "running": _backup_status["running"],
#         "result": _backup_status["result"],
#         "error": _backup_status["error"],
#     }

# --- Migriert zu Admin-Registry: backup.restore (Sprint 08) ---
# @backup_router.post("/restore")
# async def restore_backup_endpoint(req: BackupRestoreRequest, user=Depends(get_admin_user)):
#     cfg = load_config()
#     backup_dir = get_backup_dir(cfg)
#     backup_path = (Path(backup_dir) / req.filename).resolve()
#     if not str(backup_path).startswith(str(Path(backup_dir).resolve())):
#         raise HTTPException(status_code=400, detail="Ungueltiger Dateiname")
#     if not backup_path.exists():
#         raise HTTPException(status_code=404, detail="Backup-Datei nicht gefunden")
#     from kiara.backup import validate_backup, restore_backup
#     valid, manifest, errors = validate_backup(str(backup_path))
#     if not valid and not req.force:
#         raise HTTPException(status_code=400, detail=f"Validierung fehlgeschlagen: {'; '.join(errors)}")
#     result = restore_backup(str(backup_path), cfg, force=req.force)
#     return {"message": "Backup wiederhergestellt", "restored": result.get("restored", [])}
