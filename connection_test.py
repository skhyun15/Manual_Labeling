import paramiko
import socket


def check_server_connection(host, port):
    try:
        # 소켓을 사용해 포트가 열려 있는지 확인
        with socket.create_connection((host, port), timeout=5):
            print(f"{host}:{port} 포트 열려 있음")
    except socket.timeout:
        print(f"{host}:{port} 연결 타임아웃 (포트 닫힘 가능성)")
        return False
    except Exception as e:
        print(f"{host}:{port} 연결 실패: {e}")
        return False
    return True


def check_ssh_login(host, port, username, password=None, key_file=None):
    try:
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # 인증 방식 선택
        if key_file:
            private_key = paramiko.RSAKey.from_private_key_file(key_file)
            ssh_client.connect(host, port=port, username=username, pkey=private_key)
        else:
            ssh_client.connect(host, port=port, username=username, password=password)

        print(f"SSH 연결 성공: {host}:{port}")
        ssh_client.close()
        return True
    except paramiko.AuthenticationException:
        print("인증 실패: 사용자 이름 또는 비밀번호/키 확인 필요")
    except paramiko.SSHException as e:
        print(f"SSH 연결 오류: {e}")
    except Exception as e:
        print(f"알 수 없는 에러 발생: {e}")
    return False


def debug_ssh_connection(host, port, username, password=None, key_file=None):
    print("=== SSH 연결 디버깅 시작 ===")

    # 1. 서버 IP와 포트 확인
    if not check_server_connection(host, port):
        print("서버에 연결할 수 없습니다. 네트워크 또는 방화벽 문제일 수 있습니다.")
        return

    # 2. SSH 로그인 시도
    if check_ssh_login(host, port, username, password, key_file):
        print("SSH 연결 성공!")
    else:
        print("SSH 연결 실패: 인증 또는 서버 설정 문제일 가능성")

    print("=== 디버깅 종료 ===")


# 테스트 실행
host = "103.22.220.93"  # 원격 서버 IP
port = 22  # 원격 서버 포트
username = "nhcho"  # SSH 사용자명
password = "NhCho84869!"  # 비밀번호 (키 파일을 사용하는 경우 None)
key_file = None  # 키 파일 경로 (예: "/path/to/private_key.pem")

debug_ssh_connection(host, port, username, password, key_file)
