import asyncio
import argparse
import logging
import time
import uuid
from pathlib import Path
from typing import Optional, List
from fnmatch import fnmatch

import aiosqlite
from pydantic import BaseModel, Field


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


DOMAIN_TO_CONTENT_TYPE = {
    "cord-19": "scientific_paper",
    "hotpotqa": "qa_pair",
    "wikihop": "multi_hop_qa",
    "eu_legislation": "legal_doc",
    "apt_notes": "security_report",
}


class Document(BaseModel):
    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_path: str
    filename: str
    doc_name: str
    file_extension: str
    header_exists: Optional[int] = None
    file_size: int
    domain: str
    content_type: str


def get_ignore_patterns(ignore_file: Path) -> List[str]:
    if not ignore_file.exists():
        logging.warning(f"Ignore file not found at '{ignore_file}'.")
        return []
    with open(ignore_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def is_ignored(file_path: Path, root: Path, ignore_patterns: List[str]) -> bool:
    relative_path = file_path.relative_to(root)
    for pattern in ignore_patterns:
        if fnmatch(relative_path, pattern) or fnmatch(file_path.name, pattern):
            return True
    return False


def get_domain_and_content_type(file_path: Path) -> (str, str):
    domain = file_path.parent.name
    content_type = DOMAIN_TO_CONTENT_TYPE.get(domain, "unknown")
    return domain, content_type


def get_header_exists(file_extension: str) -> Optional[int]:
    if file_extension == ".csv":
        return 1
    if file_extension == ".tsv":
        return 0
    return None


async def get_file_metadata(file_path: Path) -> Document:
    file_size = file_path.stat().st_size
    file_extension = file_path.suffix
    domain, content_type = get_domain_and_content_type(file_path)
    header_exists = get_header_exists(file_extension)
    
    return Document(
        source_path=str(file_path.resolve()),
        filename=file_path.name,
        doc_name=file_path.name,
        file_extension=file_extension,
        header_exists=header_exists,
        file_size=file_size,
        domain=domain,
        content_type=content_type,
    )


async def insert_batch(db: aiosqlite.Connection, batch: List[Document]):
    if not batch:
        return
    sql = """
        INSERT OR IGNORE INTO documents (
            doc_id, source_path, filename, doc_name, file_extension, header_exists,
            file_size, domain, content_type
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    params = [
        (
            d.doc_id, d.source_path, d.filename, d.doc_name, d.file_extension,
            d.header_exists, d.file_size, d.domain, d.content_type
        )
        for d in batch
    ]
    try:
        await db.executemany(sql, params)
    except Exception as e:
        logging.error(f"Failed to insert a batch: {e}")

async def process_folder(
    folder: str,
    db_path: str,
    ignorefile: Optional[str],
    batch_size: int,
    max_concurrency: int,
    commit_interval: int,
    progress_interval: int,
):
    start_time = time.time()
    root = Path(folder)
    db = await aiosqlite.connect(db_path)
    await db.execute("PRAGMA journal_mode=WAL;")

    ignore_patterns = get_ignore_patterns(Path(ignorefile)) if ignorefile else []
    
    semaphore = asyncio.Semaphore(max_concurrency)
    lock = asyncio.Lock()
    batch: List[Document] = []
    total_rows = 0
    
    logging.info(f"Starting scan in '{folder}' (batch_size={batch_size}, max_concurrency={max_concurrency})")

    async def process_file(file_path: Path):
        nonlocal total_rows, batch
        async with semaphore:
            try:
                metadata = await get_file_metadata(file_path)
                
                async with lock:
                    batch.append(metadata)
                    if len(batch) >= batch_size:
                        current_batch = list(batch)
                        batch.clear()
                        
                        await insert_batch(db, current_batch)
                        total_rows += len(current_batch)
                        
                        if total_rows % commit_interval == 0:
                            await db.commit()
                        if total_rows % progress_interval == 0:
                            rate = total_rows / (time.time() - start_time)
                            logging.info(f"Processed {total_rows} rows... ({rate:.2f} rows/s)")
            except Exception as e:
                logging.error(f"Failed to process file {file_path}: {e}")

    tasks = [
        process_file(p)
        for p in root.rglob("*")
        if p.is_file() and not is_ignored(p, root, ignore_patterns)
    ]
    await asyncio.gather(*tasks)

    # Insert any remaining files in the final batch
    if batch:
        await insert_batch(db, batch)
        total_rows += len(batch)
    
    await db.commit()
    await db.close()
    
    duration = time.time() - start_time
    rate = total_rows / duration if duration > 0 else 0
    logging.info(f"Completed processing. Rows inserted: {total_rows}. Time: {duration:.2f}s. Rate: {rate:.2f} rows/s")


def main():
    parser = argparse.ArgumentParser(description="Load file metadata into a SQLite database.")
    parser.add_argument("folder", help="Root folder to scan for files.")
    parser.add_argument("--db", default="file_metadata.db", help="Path to the SQLite database file.")
    parser.add_argument("--ignorefile", help="Path to a file with ignore patterns.")
    parser.add_argument("--batch-size", type=int, default=100, help="Number of files to process in a batch.")
    parser.add_argument("--max-concurrency", type=int, default=50, help="Maximum concurrent file processing tasks.")
    parser.add_argument("--commit-interval", type=int, default=2000, help="Commit to DB every N rows.")
    parser.add_argument("--progress-interval", type=int, default=1000, help="Log progress every N rows.")
    args = parser.parse_args()

    asyncio.run(
        process_folder(
            args.folder,
            args.db,
            args.ignorefile,
            batch_size=args.batch_size,
            max_concurrency=args.max_concurrency,
            commit_interval=args.commit_interval,
            progress_interval=args.progress_interval,
        )
    )

if __name__ == "__main__":
    main()
