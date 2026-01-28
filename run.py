import threading

from nety.main import NETYSystem
from app.main import launch_app


def start_nety():
    nety = NETYSystem()
    nety.start()
    nety.run()


def start_dashboard():
    launch_app()


if __name__ == "__main__":
    nety_thread = threading.Thread(
        target=start_nety,
        daemon=True
    )

    dashboard_thread = threading.Thread(
        target=start_dashboard,
        daemon=True
    )

    nety_thread.start()
    dashboard_thread.start()

    nety_thread.join()
    dashboard_thread.join()
