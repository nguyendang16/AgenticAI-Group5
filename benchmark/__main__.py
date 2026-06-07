from dotenv import load_dotenv

from benchmark.cli import main
from benchmark.paths import REPO_ROOT

load_dotenv(REPO_ROOT / '.env', override=False)

if __name__ == '__main__':
    raise SystemExit(main())
