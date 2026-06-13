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

    # core stack
    print("core stack:")
    results.append(check("numpy", lambda: importlib.import_module("numpy").__version__))
    results.append(check("scipy", lambda: importlib.import_module("scipy").__version__))
    results.append(check("matplotlib", lambda: importlib.import_module("matplotlib").__version__))
    results.append(check("polars", lambda: importlib.import_module("polars").__version__))
    results.append(check("yaml", lambda: importlib.import_module("yaml").__version__))

    # lqr sanity check
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

    # hardware comms (imports only)
    print("\nhardware comms:")

    def moteus_check():
        import moteus

        assert hasattr(moteus, "Controller") and hasattr(moteus, "Fdcanusb")
        return "controller, fdcanusb in moteus lib"

    results.append(check("moteus", moteus_check))

    def moteus_cli_check():
        import shutil

        tview = shutil.which("tview")
        mtool = shutil.which("moteus_tool")
        if not (tview and mtool):
            raise RuntimeError(f"tview={tview}, moteus_tool={mtool}")
        return "tview + moteus_tool on PATH"

    results.append(check("moteus cli", moteus_cli_check))

    results.append(check("serial", lambda: importlib.import_module("serial").__version__))
    results.append(check("can", lambda: importlib.import_module("can").__version__))

    # sim
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

    # torch (only if ml optional dep installed)
    print("\nml (optional, requires --extra ml):")
    try:
        import torch  # pyright: ignore[reportMissingImports] -- torch is optional for now

        print(f"  OK    torch: {torch.__version__}")
        if sys.platform == "linux":
            cuda = torch.cuda.is_available()
            print(f"  {'OK   ' if cuda else 'WARN '} cuda available: {cuda}")
            if cuda:
                print(f"        device: {torch.cuda.get_device_name(0)}")
        elif sys.platform == "darwin":
            mps = torch.backends.mps.is_available()
            print(f"  {'OK   ' if mps else 'WARN '} mps available: {mps}")
        results.append(cuda if sys.platform == "linux" else mps)
    except ImportError:
        print("  SKIP  torch not installed (run `uv sync --extra ml` to add)")

    # summary
    print()
    failed = sum(1 for r in results if r is False)
    if failed:
        print(f"{failed} check(s) failed.")
        sys.exit(1)  # exit with non-zero status
    else:
        print("all checks passed. ready to go.")


if __name__ == "__main__":
    main()
