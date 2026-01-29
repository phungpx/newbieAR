import json
from loguru import logger
from uuid import uuid4
from pathlib import Path
from deepeval.dataset.golden import Golden


def save_goldens_to_files(goldens: list[Golden], output_dir: str = "goldens"):
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {output_dir}")

    for golden in goldens:
        file_dir = output_dir / Path(golden.sourceFile).stem
        if not file_dir.exists():
            file_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {file_dir}")

        file_path = file_dir / f"{uuid4()}.json"
        golden_data = golden.model_dump(by_alias=True, exclude_none=True)

        try:
            with file_path.open(mode="w", encoding="utf-8") as f:
                json.dump(golden_data, f, indent=4, ensure_ascii=False)
            logger.info(f"Saved: {file_path}")
        except Exception as e:
            logger.error(f"Failed to save {file_path}: {e}")

    logger.info(f"\nSuccessfully saved {len(goldens)} files to '{output_dir}'.")
