from launch import is_installed, run_pip

if not is_installed("clipseg"):
    run_pip(f"install git+https://github.com/timojl/clipseg", "clipseg")
