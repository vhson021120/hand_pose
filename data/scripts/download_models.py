# Example usage: python data/scripts/download_models.py
import gdown
from pathlib import Path

WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
	("1hv0xwdbdf-Ym06Hpjy6wKyIklNmKmB99", "kapao_s_coco.pt"),
	("1B-QGa99n7ZrxkgKCQ1XHnm6j0Y-lvOQz", "kapao_m_coco.pt"),
	("1jYDfvRjhMoDf5xpMq1AuzdkFzUeLwt85", "kapao_l_coco.pt"),
	("1SmWwmqPwb_G6d9UPAFSUnWZ-4eaL9TTv", "kapao_s_crowdpose.pt"),
	("1IqrAk-gBdcfONrlIT6d4ndaDqnlFmT0r", "kapao_m_crowdpose.pt"),
	("146DW9ELzIBY2oDofPru446yLErxHrjJU", "kapao_l_crowdpose.pt"),
]

for file_id, filename in MODELS:
	gdown.download(
		f"https://drive.google.com/uc?id={file_id}",
		output=str(WEIGHTS_DIR / filename),
	)