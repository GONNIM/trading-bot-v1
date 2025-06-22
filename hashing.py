# 비밀번호 해시로 변환 (최초 1회)

import yaml
from streamlit_authenticator.utilities.hasher import Hasher

with open("credentials.yaml") as file:
    config = yaml.safe_load(file)

# 패스워드 리스트 추출
passwords = [
    user_info["password"] for user_info in config["credentials"]["usernames"].values()
]

# hash_list를 사용해 해시 생성
hashed_passwords = Hasher.hash_list(passwords)

# 해시된 패스워드로 업데이트
for i, username in enumerate(config["credentials"]["usernames"]):
    config["credentials"]["usernames"][username]["password"] = hashed_passwords[i]

with open("credentials.yaml", "w") as file:
    yaml.dump(config, file, allow_unicode=True)
