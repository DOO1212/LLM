import sys

from app import create_app


def main():
    app = create_app()
    client = app.test_client()
    resp = client.get("/auth/me")
    if resp.status_code != 200:
        print("healthcheck failed")
        sys.exit(1)
    print("healthcheck ok")


if __name__ == "__main__":
    main()

