import os
from dotenv import load_dotenv

load_dotenv()
ACCESS = os.getenv("UPBIT_ACCESS")
SECRET = os.getenv("UPBIT_SECRET")
if not (ACCESS and SECRET):
    raise EnvironmentError(".env 에 UPBIT_ACCESS / UPBIT_SECRET 값을 설정하세요.")

MIN_CASH = 10_000
MIN_FEE_RATIO = 0.0005
