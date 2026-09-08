"""env smoke test for finn. good to run after `uv sync` to check all dependencies."""

import importlib
import platform
import sys


def check(name, fn):
    try:
        result = fn()
        print(f"  OK    {name}: {result}")
        return True
    except Exception as e:
        print(f"  FAIL  {name}: {type(e).__name__}: {e}")
        return False


def main():
    print(f"python {sys.version.split()[0]} on {platform.system()} ({platform.machine()})")
    print()

    results = []

    print("core stack:")
    results.append(check("numpy", lambda: importlib.import_module("numpy").__version__))
    results.append(check("scipy", lambda: importlib.import_module("scipy").__version__))
    results.append(check("matplotlib", lambda: importlib.import_module("matplotlib").__version__))
    results.append(check("yaml", lambda: importlib.import_module("yaml").__version__))

    print("\ncontrol math:")

    def lqr_check():
        import numpy as np
        from scipy.linalg import solve_continuous_are

        # toy double integrator: x_ddot = u for sanity
        A = np.array([[0, 1], [0, 0]])
        B = np.array([[0], [1]])
        Q = np.eye(2)
        R = np.array([[1.0]])
        P = solve_continuous_are(A, B, Q, R)
        K = np.linalg.solve(R, B.T @ P)
        return f"K = {K.flatten().round(3).tolist()}"

    results.append(check("solve_continuous_are", lqr_check))

    # Hardware tooling is optional because the simulator does not need the
    # large Qt dependency pulled in by moteus-gui.
    print("\nhardware tooling (optional, requires --extra hardware):")
    try:
        import shutil

        import moteus

        assert hasattr(moteus, "Controller") and hasattr(moteus, "Fdcanusb")
        tview = shutil.which("tview")
        mtool = shutil.which("moteus_tool")
        if not (tview and mtool):
            raise RuntimeError(f"tview={tview}, moteus_tool={mtool}")
        print("  OK    moteus: controller, fdcanusb, tview, and moteus_tool")
    except ImportError:
        print("  SKIP  moteus not installed (run `uv sync --extra hardware` to add)")

    print("\nsim:")

    def mujoco_check():
        import mujoco

        model = mujoco.MjModel.from_xml_string(
            "<mujoco><worldbody><body><geom size='1'/></body></worldbody></mujoco>"
        )
        data = mujoco.MjData(model)
        mujoco.mj_step(model, data)
        return f"version {mujoco.__version__}, stepped a model"

    results.append(check("mujoco", mujoco_check))

    print()
    failed = sum(1 for r in results if r is False)
    if failed:
        print(f"{failed} check(s) failed.")
        sys.exit(1)  # exit with non-zero status
    else:
        print("all checks passed. ready to go.")


if __name__ == "__main__":
    main()
